#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 JADDANGI AI: 3B MODEL WAKE-UP SCRIPT (CPT via 4-bit QLoRA)
Uses the EXACT TeluguGPT architecture (interleaved RoPE, tied wte/lm_head,
GQA with repeat_interleave, SwiGLU with 256-rounded hidden dim) — matched
to Telugu_Model_3B_V1.pt produced by the Net2Net width+depth expansion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, glob, gc
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
from transformers import get_cosine_schedule_with_warmup
import sentencepiece as spm
from huggingface_hub import HfApi, login, hf_hub_download

# ============================================================
# 0. AUTH & PATHS
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment. Set it as a Colab/Kaggle Secret.")
login(token=HF_TOKEN)
api = HfApi()

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
MODEL_FILE = "Telugu_Model_3B_V1.pt"
TOKENIZER_FILE = "telugu_spm.model"
DATA_FILE = "telugu_corpus.txt"

for fname in [TOKENIZER_FILE, MODEL_FILE]:
    if not os.path.exists(fname):
        print(f"📥 Downloading {fname}...")
        hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=".")

# ============================================================
# 1. Configuration — EXACT 3B architecture
# ============================================================
class Config:
    vocab_size = 32000
    d_model = 1728
    n_layers = 68
    n_heads = 24
    n_kv_heads = 6
    max_seq_len = 384
    rope_theta = 10000
    expansion_factor = 4
    dropout = 0.0
    lora_rank = 16
    lora_alpha = 32

config = Config()

# ============================================================
# 2. ARCHITECTURE — EXACT COPY of your TeluguGPT (interleaved RoPE, tied embeddings)
# ============================================================
def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot, k_rot = torch.empty_like(q), torch.empty_like(k)
    q_rot[..., 0::2] = q[..., 0::2] * cos - q[..., 1::2] * sin
    k_rot[..., 0::2] = k[..., 0::2] * cos - k[..., 1::2] * sin
    q_rot[..., 1::2] = q[..., 1::2] * cos + q[..., 0::2] * sin
    k_rot[..., 1::2] = k[..., 1::2] * cos + k[..., 0::2] * sin
    return q_rot, k_rot

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

def inject_lora(module, rank, alpha, dropout, target_names):
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, Linear4bit)) and name in target_names:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
        else:
            inject_lora(child, rank, alpha, dropout, target_names)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[:, :, :T, :], sin[:, :, :T, :])
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hd = int(2 / 3 * config.expansion_factor * config.d_model)
        hd = (hd // 256) * 256
        self.w1 = nn.Linear(config.d_model, hd, bias=False)
        self.w2 = nn.Linear(config.d_model, hd, bias=False)
        self.w3 = nn.Linear(hd, config.d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))

class TeluguGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   # TIED — same Parameter object
        head_dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(config.max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Seq len {T} exceeds {self.config.max_seq_len}"
        x = self.wte(idx)
        cos = self.cos_cached[:, :, :T, :].to(x.device)
        sin = self.sin_cached[:, :, :T, :].to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm_f(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 3. 4-BIT CONVERSION
# lm_head is EXCLUDED — it's tied to wte.weight (embedding), which must
# stay full-precision & trainable. Converting it to Linear4bit would break
# the tie and make the embedding un-trainable.
# ============================================================
def convert_to_4bit(module, exclude_names=("lm_head",)):
    for name, child in module.named_children():
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            new_layer = Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                quant_type="nf4", compute_dtype=torch.float16
            )
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()
            setattr(module, name, new_layer)
        else:
            convert_to_4bit(child, exclude_names)


# ============================================================
# 4. DATASET
# ============================================================
class TeluguTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, seq_len):
        self.seq_len = seq_len
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode_as_ids(text)
        n_chunks = len(ids) // (seq_len + 1)
        ids = ids[: n_chunks * (seq_len + 1)]
        self.chunks = torch.tensor(ids, dtype=torch.long).view(n_chunks, seq_len + 1)

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ============================================================
# 5. LOAD MODEL — pop tied/buffer keys, then re-tie explicitly (matches
#    your inference script's exact pattern)
# ============================================================
print("🔧 Initializing 3B model...")
model = TeluguGPT(config)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("📥 Loading base checkpoint...")
ckpt = torch.load(MODEL_FILE, map_location="cpu")
state_dict = ckpt.get("model_state_dict", ckpt)

for buf in ["cos_cached", "sin_cached", "lm_head.weight"]:
    state_dict.pop(buf, None)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"⚠️  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
if missing:
    print("   First few missing:", missing[:5])
if unexpected:
    print("   First few unexpected:", unexpected[:5])
model.lm_head.weight = model.wte.weight  # re-tie explicitly, belt-and-suspenders
del state_dict, ckpt
gc.collect()

print("⚙️ Converting to 4-bit NF4 (lm_head/wte excluded, stays full precision)...")
convert_to_4bit(model, exclude_names=("lm_head",))
model = model.cuda()

print("🧬 Injecting LoRA Adapters (attention + MLP)...")
targets = ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")
inject_lora(model, config.lora_rank, config.lora_alpha, 0.05, targets)

# wte.weight IS lm_head.weight (tied) — training it trains both.
model.wte.weight.requires_grad = True

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.wte.register_forward_hook(make_inputs_require_grad)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"🔢 Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================
# 6. OPTIMIZER & SCHEDULER
# embed_params holds the tied wte/lm_head weight ONCE (it's a single
# Parameter object shared by both — listing it twice would double-update it).
# ============================================================
embed_params = [model.wte.weight]
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]

optimizer = bnb.optim.AdamW8bit([
    {"params": lora_params, "lr": 2e-4},
    {"params": embed_params, "lr": 1e-5},
])

total_steps = 1000
accum_steps = 8
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
scaler = GradScaler()

# ============================================================
# 7. RESUME FROM CRASH
# ============================================================
start_step = 0
ckpts = sorted(glob.glob("cpt_3b_step_*.pt"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
if ckpts:
    latest = ckpts[-1]
    print(f"🔄 Resuming from {latest}...")
    resume_state = torch.load(latest, map_location="cuda")
    model.load_state_dict(resume_state["model_partial"], strict=False)
    optimizer.load_state_dict(resume_state["optimizer"])
    scheduler.load_state_dict(resume_state["scheduler"])
    start_step = resume_state["step"]
    print(f"✅ Resumed at step {start_step}")

# ============================================================
# 8. DATA
# ============================================================
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(TOKENIZER_FILE)
pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0

dataset = TeluguTextDataset(tokenizer, DATA_FILE, config.max_seq_len)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
print(f"📚 Dataset: {len(dataset)} sequences of length {config.max_seq_len}")

# ============================================================
# 9. TRAINING LOOP
# ============================================================
model.train()
print(f"🚀 Starting CPT to wake up the 3B model (from step {start_step})...")

step = start_step
UPLOAD_TO_HF = True

for epoch in range(100):
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.cuda()
        input_ids = batch[:, :-1].contiguous()
        labels = batch[:, 1:].contiguous()

        with autocast(dtype=torch.float16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=pad_id
            ) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item()*accum_steps:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            if step % 250 == 0:
                ckpt_name = f"cpt_3b_step_{step}.pt"
                print(f"💾 Saving checkpoint {ckpt_name}...")
                model_partial = {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad}
                torch.save({
                    "step": step,
                    "model_partial": model_partial,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss": loss.item() * accum_steps,
                }, ckpt_name)

                if UPLOAD_TO_HF:
                    api.upload_file(
                        path_or_fileobj=ckpt_name,
                        path_in_repo=f"checkpoints/{ckpt_name}",
                        repo_id=REPO_ID,
                        commit_message=f"CPT checkpoint step {step}"
                    )

            if step >= total_steps:
                break
    if step >= total_steps:
        break

print("🎉 3B Model is now Awake & Stabilized!")
final_partial = {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad}
torch.save({"step": step, "model_partial": final_partial}, "cpt_3b_final.pt")
api.upload_file(
    path_or_fileobj="cpt_3b_final.pt",
    path_in_repo="checkpoints/cpt_3b_final.pt",
    repo_id=REPO_ID,
    commit_message="Final CPT checkpoint - 3B model wake-up complete"
)
print("☁️ Final checkpoint uploaded to Hugging Face.")
