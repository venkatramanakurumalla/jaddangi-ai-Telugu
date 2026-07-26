

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
తెలుగు LoRA ఫైన్-ట్యూనింగ్ - ఫైనల్ బెస్ట్ వెర్షన్
====================================================
మార్పులు ఈ వెర్షన్‌లో:
- torch.manual_seed() బగ్ ఫిక్స్ (local torch.Generator వాడి global RNG state కాపాడబడింది)
- Held-out validation split + periodic eval loss ట్రాకింగ్ (overfitting కనిపెట్టడానికి)
- HF_TOKEN env-based (హార్డ్‌కోడ్ లేదు)
- RoPE టోగుల్ (USE_SCALED_ROPE) మీ base checkpoint కి సరిపోయేలా
- Missing-keys strict validation
- Document-boundary-aware, non-overlapping chunking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import random
import requests
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from huggingface_hub import hf_hub_download, login
from torch.amp import GradScaler, autocast

# ============================================================
# 0. AUTH & RoPE CONFIGURATION TOGGLE
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️  HF_TOKEN env లో సెట్ కాలేదు — repo private అయితే ముందు సెట్ చేయండి.")

USE_SCALED_ROPE = False  # ← మీ base model training scriptలో NTK/YaRN scaling వాడితేనే True పెట్టండి.

# ============================================================
# 1. CONFIGURATION
# ============================================================
class Config:
    vocab_size = 32000
    d_model = 1440
    n_layers = 21
    n_heads = 20
    n_kv_heads = 5
    max_seq_len = 1024
    rope_theta = 10000
    expansion_factor = 4
    dropout = 0.0

    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.05

    batch_size = 2
    gradient_accumulation_steps = 8
    learning_rate = 1e-4
    max_steps = 500          # మొదట smoke test — తర్వాత పెంచుకోండి
    warmup_steps = 50
    save_every = 100
    log_every = 25
    eval_every = 100
    max_grad_norm = 1.0

    min_doc_length = 80
    max_text_length = 10000
    val_split_ratio = 0.12   # held-out validation కోసం కేటాయించే భాగం
    eval_batches = 20        # ప్రతి eval_every దగ్గర ఎన్ని val batches వాడాలి

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
BASE_MODEL_FILE = "Telugu_Model_0.5B_V4.pt"
TOKENIZER_FILE = "telugu_spm.model"
LORA_SAVE_FILE = "Telugu_LoRA_Adapter_v5.pt"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/venkatramanakurumalla/Telugu10m/main/telugudataset"
# ⚠️ పైన URL లో username మీ HF repo owner (VenkataRamanaKurumallajaddangi) తో వేరుగా ఉంది —
#    ఇది typo కాదని, ఇది సరైన GitHub repo యేనని రన్ చేసే ముందు బ్రౌజర్‌లో ఒకసారి చెక్ చేయండి.

# ============================================================
# 2. టోకెనైజర్
# ============================================================
print("📥 టోకెనైజర్ & బేస్ మోడల్ డౌన్లోడ్...")
if not os.path.exists(TOKENIZER_FILE):
    hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_FILE, local_dir=".", local_dir_use_symlinks=False)
if not os.path.exists(BASE_MODEL_FILE):
    hf_hub_download(repo_id=REPO_ID, filename=BASE_MODEL_FILE, local_dir=".", local_dir_use_symlinks=False)

class SentencePieceTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.pad_token_id = self.sp.pad_id()
        if self.pad_token_id < 0:
            self.pad_token_id = self.sp.unk_id()
        self.eos_token_id = self.sp.eos_id()
        print(f"PAD id: {self.pad_token_id}, EOS id: {self.eos_token_id}")

    def encode(self, text, max_length=None):
        ids = self.sp.encode_as_ids(text)
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids):
        return self.sp.decode_ids(ids)

tokenizer = SentencePieceTokenizer(TOKENIZER_FILE)

# ============================================================
# 3. డేటాసెట్ — document-boundary-aware chunking + train/val split
# ============================================================
class TeluguStoryDataset(Dataset):
    def __init__(self, tokenizer, config, url=GITHUB_RAW_URL, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []
        self._prepare_data(url, split)

    def _prepare_data(self, url, split):
        print(f"📥 GitHub నుండి డేటా డౌన్లోడ్ ({split} split కోసం): {url}")
        resp = requests.get(url)
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            raise RuntimeError(f"డౌన్లోడ్ విఫలం: {resp.status_code}")
        text = resp.text

        docs = [d.strip() for d in text.split("\n\n") if len(d.strip()) > self.config.min_doc_length]
        if len(docs) < 5:
            docs = [d.strip() for d in text.split("\n") if len(d.strip()) > self.config.min_doc_length]

        rng = random.Random(1234)
        rng.shuffle(docs)
        val_count = max(1, int(len(docs) * self.config.val_split_ratio))
        if split == "val":
            docs = docs[:val_count]
        else:
            docs = docs[val_count:]

        print(f"📄 {split} split: {len(docs)} డాక్యుమెంట్లు.")
        total_tokens = 0
        seq_len = self.config.max_seq_len + 1

        for doc in docs:
            ids = self.tokenizer.encode(doc, max_length=self.config.max_text_length)
            if len(ids) < 2:
                continue
            ids = ids + [self.tokenizer.eos_token_id]
            total_tokens += len(ids)

            for start in range(0, len(ids) - 1, self.config.max_seq_len):
                chunk = ids[start:start + seq_len]
                if len(chunk) < 2:
                    continue
                if len(chunk) < seq_len:
                    pad_len = seq_len - len(chunk)
                    chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                self.data.append((input_ids, labels))

        num_sequences = len(self.data)
        eff_batch = self.config.batch_size * self.config.gradient_accumulation_steps
        steps_per_epoch = math.ceil(num_sequences / eff_batch) if eff_batch > 0 else 0
        print(f"✅ [{split}] {num_sequences} సీక్వెన్స్‌లు, మొత్తం టోకెన్లు: {total_tokens}")
        if split == "train":
            print(f"📊 Effective Batch: {eff_batch} → సుమారు {steps_per_epoch} steps/epoch")

        if len(self.data) == 0:
            raise RuntimeError(f"[{split}] డేటా ఖాళీ. min_doc_length తగ్గించండి లేదా val_split_ratio సర్దుబాటు చేయండి.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================
# 4. మోడల్
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
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

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
        # lora_B zero-init: adapter starts as a no-op

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

def inject_lora(module, rank, alpha, dropout, target_names=("q_proj","k_proj","v_proj","o_proj","w1","w2","w3")):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_names:
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
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        q, k = apply_rotary_pos_emb(q, k, cos[:,:,:T,:], sin[:,:,:T,:])
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1,2).contiguous().view(B,T,C))

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hd = int(2/3 * config.expansion_factor * config.d_model)
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
        self.lm_head.weight = self.wte.weight

        head_dim = config.d_model // config.n_heads
        if USE_SCALED_ROPE:
            scale_factor = config.max_seq_len / 384
            adjusted_theta = config.rope_theta * (scale_factor ** (head_dim / (head_dim - 2)))
            inv_freq = 1.0 / (adjusted_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            print("🔧 Scaled RoPE enabled (384→1024)")
        else:
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            print("🔧 Standard RoPE (no scaling)")

        freqs = torch.outer(torch.arange(config.max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        cos = self.cos_cached[:, :, :T, :].to(x.device)
        sin = self.sin_cached[:, :, :T, :].to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm_f(x))

# ============================================================
# 5. మోడల్ లోడ్ & LoRA ఇంజెక్షన్
# ============================================================
print("🔧 మోడల్‌ను సృష్టిస్తున్నాం...")
config = Config()
model = TeluguGPT(config)

print("📦 బేస్ చెక్‌పాయింట్ లోడ్...")
ckpt = torch.load(BASE_MODEL_FILE, map_location="cpu")
state_dict = dict(ckpt.get("model_state_dict", ckpt))
state_dict.pop("cos_cached", None)
state_dict.pop("sin_cached", None)
state_dict.pop("lm_head.weight", None)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
allowed_missing = {"cos_cached", "sin_cached", "lm_head.weight"}
real_missing = [k for k in missing if k not in allowed_missing]
if real_missing:
    print("❌ క్లిష్టమైన Missing keys:")
    for k in real_missing:
        print("  ", k)
    raise RuntimeError("Checkpoint keys mismatch. ఆపివేయబడింది.")
else:
    print("✅ All critical keys loaded.")
if unexpected:
    print("⚠️ Unexpected keys (ignored):", unexpected[:5])

model.lm_head.weight = model.wte.weight
model = model.to(model.device)

for p in model.parameters():
    p.requires_grad = False

print(f"🧩 LoRA (r={config.lora_rank}, alpha={config.lora_alpha})...")
inject_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout)
model = model.to(model.device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

# ============================================================
# 6. డేటా లోడర్లు (train + held-out val)
# ============================================================
print("📊 డేటాసెట్ తయారీ (train + val split)...")
train_dataset = TeluguStoryDataset(tokenizer, config, split="train")
val_dataset = TeluguStoryDataset(tokenizer, config, split="val")
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ============================================================
# 7. ఆప్టిమైజర్
# ============================================================
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available())

def lr_lambda(step):
    if step < config.warmup_steps:
        return float(step) / float(max(1, config.warmup_steps))
    progress = (step - config.warmup_steps) / max(1, (config.max_steps - config.warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, progress))))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# 8. జనరేషన్ — local RNG (global torch RNG ని touch చేయదు)
# ============================================================
@torch.no_grad()
def generate_sample(prompt, max_tokens=150, temperature=0.8, seed=42):
    model.eval()
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)
    ids = tokenizer.encode(prompt, max_length=config.max_seq_len)
    input_ids = torch.tensor([ids], device=model.device)
    for _ in range(max_tokens):
        logits = model(input_ids[:, -config.max_seq_len:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    model.train()
    return tokenizer.decode(input_ids[0].tolist())

# ============================================================
# 9. HELD-OUT EVAL LOSS — overfitting కనిపెట్టడానికి
# ============================================================
@torch.no_grad()
def estimate_val_loss():
    model.eval()
    total_loss = 0.0
    batches = 0
    for input_ids, labels in val_loader:
        input_ids = input_ids.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)
        with autocast(device_type=model.device.type, enabled=(model.device.type == "cuda")):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
        total_loss += loss.item()
        batches += 1
        if batches >= config.eval_batches:
            break
    model.train()
    return total_loss / max(1, batches)

# ============================================================
# 10. శిక్షణ లూప్
# ============================================================
model.train()
step = 0
optimizer_step = 0
window_loss = 0.0
loss_count = 0
last_avg_loss = None
best_val_loss = float('inf')
start_time = time.time()
data_iter = iter(train_loader)

print(f"\n🚀 TRAINING START")
print(f"   Device: {model.device.type.upper()}")
print(f"   Steps: {config.max_steps}")
print(f"   Effective Batch: {config.batch_size * config.gradient_accumulation_steps}")
print("=" * 60)

while optimizer_step < config.max_steps:
    try:
        input_ids, labels = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        input_ids, labels = next(data_iter)

    input_ids = input_ids.to(model.device, non_blocking=True)
    labels = labels.to(model.device, non_blocking=True)

    with autocast(device_type=model.device.type, enabled=(model.device.type == "cuda")):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        loss = loss / config.gradient_accumulation_steps

    scaler.scale(loss).backward()
    window_loss += loss.item() * config.gradient_accumulation_steps
    loss_count += 1

    if (step + 1) % config.gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        if optimizer_step % config.log_every == 0:
            avg_loss = window_loss / max(1, loss_count)
            last_avg_loss = avg_loss
            ppl = math.exp(min(avg_loss, 20))
            lr = scheduler.get_last_lr()[0]
            print(f"📊 Step {optimizer_step}/{config.max_steps} | Train Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")
            window_loss = 0.0
            loss_count = 0

        if optimizer_step % config.eval_every == 0:
            val_loss = estimate_val_loss()
            val_ppl = math.exp(min(val_loss, 20))
            flag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                flag = " 🏆 (best so far)"
            elif last_avg_loss is not None and val_loss > last_avg_loss * 1.3:
                flag = " ⚠️ (val loss much higher than train — possible overfitting)"
            print(f"   🧪 Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}{flag}")

            print(f"\n📝 Sample (Step {optimizer_step}):")
            prompts = ["ఒక చిన్న పల్లెటూళ్ళో ఒక రైతు", "హైదరాబాద్ నగరంలో"]
            for p in prompts:
                out = generate_sample(p, max_tokens=100, temperature=0.7, seed=42)
                print(f"   Prompt: {p}\n   → {out[:250]}...\n")

        if optimizer_step % config.save_every == 0 or optimizer_step == config.max_steps:
            lora_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if "lora_A" in k or "lora_B" in k
            }
            torch.save({
                "step": optimizer_step,
                "lora_state_dict": lora_state,
                "lora_config": {"rank": config.lora_rank, "alpha": config.lora_alpha},
                "train_loss": last_avg_loss if last_avg_loss is not None else 0.0,
                "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
            }, LORA_SAVE_FILE)
            print(f"💾 Checkpoint saved: {LORA_SAVE_FILE}")

    step += 1

print(f"\n🎉 TRAINING COMPLETE! Adapter: {LORA_SAVE_FILE}")
print(f"   Final Train Loss: {last_avg_loss}")
print(f"   Best Val Loss: {best_val_loss if best_val_loss != float('inf') else 'N/A'}")
