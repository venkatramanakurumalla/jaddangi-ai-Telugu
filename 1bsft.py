
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
తెలుగు LoRA SFT - 1B MODEL (FULLY PRODUCTION)
===============================================
Final Fixes Applied:
- Robust QA_BLOCK_RE regex parser to handle dataset formatting variations.
- Reduced LoRA rank (8) and alpha (16) for VRAM safety (~115M parameters).
- Checkpoint architecture validation to prevent config mismatches.
- Adjusted generation context truncation (preserves 1/3 head, 2/3 tail).
- Token-weighted gradient accumulation and correct answer-only loss masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import random
import requests
import re
import json
import hashlib
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from huggingface_hub import hf_hub_download, login
from torch.amp import GradScaler, autocast

# ============================================================
# 0. AUTH & CONFIG
# ============================================================
HF_TOKEN = os.environ.get("")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️  HF_TOKEN env లో సెట్ కాలేదు.")

USE_SCALED_ROPE = False

class Config:
    vocab_size = 32000
    d_model = 1440
    n_layers = 42
    n_heads = 20
    n_kv_heads = 5
    max_seq_len = 384
    rope_theta = 10000
    expansion_factor = 4
    dropout = 0.0

    # Optimized LoRA parameters for 1B model to save VRAM
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.05

    batch_size = 1
    gradient_accumulation_steps = 16

    learning_rate = 3e-5
    warmup_ratio = 0.05
    max_grad_norm = 1.0

    num_epochs = 2
    min_doc_length = 40
    min_answer_tokens = 4
    val_split_ratio = 0.10

    eval_batches = None
    save_every = 100
    log_every = 25

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
BASE_MODEL_FILE = "Telugu_Model_1B_V5.pt"
TOKENIZER_FILE = "telugu_spm.model"
LORA_BASE_DIR = "Qasft_1B"
BEST_ADAPTER_DIR = "Qasft_1B_best"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/venkatramanakurumalla/Telugu10m/main/teluguQAsft"

# ============================================================
# 1. Tokenizer & Vocabulary Assertion
# ============================================================
print("📥 Downloading Tokenizer & Base Model...")
if not os.path.exists(TOKENIZER_FILE):
    hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_FILE, local_dir=".", local_dir_use_symlinks=False)
if not os.path.exists(BASE_MODEL_FILE):
    hf_hub_download(repo_id=REPO_ID, filename=BASE_MODEL_FILE, local_dir=".", local_dir_use_symlinks=False)

class SentencePieceTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.eos_token_id = self.sp.eos_id()
        self.pad_token_id = self.sp.pad_id()
        if self.pad_token_id < 0:
            self.pad_token_id = self.eos_token_id
            print(f"⚠️  No PAD token. Using EOS (id={self.eos_token_id}) as PAD.")
        self.vocab_size = self.sp.get_piece_size()
        print(f"Tokenizer Vocab Size: {self.vocab_size}")

    def encode(self, text, max_length=None):
        ids = self.sp.encode_as_ids(text)
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids):
        return self.sp.decode_ids(ids)

tokenizer = SentencePieceTokenizer(TOKENIZER_FILE)
config = Config()
if tokenizer.vocab_size != config.vocab_size:
    raise RuntimeError(f"❌ VOCAB MISMATCH! Tokenizer has {tokenizer.vocab_size} tokens, but config expects {config.vocab_size}.")
print(f"✅ Vocabulary match: {config.vocab_size}")

# ============================================================
# 2. Data Preparation (Robust Regex Parser)
# ============================================================
QA_BLOCK_RE = re.compile(
    r"""
    ప్రశ్న\s*:\s*
    (?P<question>.*?)
    \s*
    (?:జవాబు|సమాధానం)\s*:\s*
    (?P<answer>.*?)
    (?=
        \n\s*ప్రశ్న\s*:|
        \Z
    )
    """,
    re.DOTALL | re.VERBOSE
)

def normalize_for_dedup(text):
    text = str(text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_prepare_data(url, config):
    print(f"📥 Downloading from: {url}")
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    response.encoding = "utf-8"
    text = response.text
    print(f"📄 Downloaded characters: {len(text):,}")

    matches = list(QA_BLOCK_RE.finditer(text))
    print(f"📄 Regex QA blocks found: {len(matches):,}")
    if len(matches) == 0:
        print("\n❌ Dataset format was not recognized.")
        print("First 1000 characters:")
        print(repr(text[:1000]))
        raise RuntimeError("No Telugu QA blocks were found. Check whether the file uses ప్రశ్న:/జవాబు: format.")

    parsed = []
    seen = set()
    for match in matches:
        question = normalize_for_dedup(match.group("question"))
        answer = normalize_for_dedup(match.group("answer"))
        if not question or not answer:
            continue
        if len(question) + len(answer) < config.min_doc_length:
            continue

        pair_hash = hashlib.sha256((question + "\n<ANSWER>\n" + answer).encode("utf-8")).hexdigest()
        if pair_hash in seen:
            continue

        seen.add(pair_hash)
        parsed.append((question, answer))

    print(f"✅ Parsed and deduplicated QA pairs: {len(parsed):,}")
    if len(parsed) == 0:
        raise RuntimeError("QA blocks were detected, but all were removed by filtering.")

    print("\n🔍 First parsed example:")
    print("ప్రశ్న:", parsed[0][0][:200])
    print("జవాబు:", parsed[0][1][:300])
    return parsed

# ============================================================
# 3. Dataset
# ============================================================
class TeluguQADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []
        self._build(qa_pairs)

    def _build(self, qa_pairs):
        for q_part, a_part in qa_pairs:
            prompt_text = f"ప్రశ్న: {q_part}\n\nజవాబు:"
            answer_text = f" {a_part}"

            q_ids = self.tokenizer.encode(prompt_text)
            a_ids = self.tokenizer.encode(answer_text)

            if len(a_ids) < self.config.min_answer_tokens:
                continue

            total_len = len(q_ids) + len(a_ids) + 1
            if total_len > self.config.max_seq_len:
                excess = total_len - self.config.max_seq_len
                if len(a_ids) - excess >= self.config.min_answer_tokens:
                    a_ids = a_ids[:len(a_ids) - excess]
                else:
                    max_q = max(1, self.config.max_seq_len - self.config.min_answer_tokens - 1)
                    q_ids = q_ids[:max_q]
                    avail = self.config.max_seq_len - len(q_ids) - 1
                    a_ids = a_ids[:avail]

            full_ids = q_ids + a_ids + [self.tokenizer.eos_token_id]
            q_len = len(q_ids)

            input_ids = full_ids[:-1]
            labels = full_ids[1:]

            # Answer-only loss masking
            labels[:q_len - 1] = [-100] * (q_len - 1)

            self.data.append((input_ids, labels))

        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty after tokenization.")
        print(f"✅ Final sequences: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_token_id):
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_inputs = []
    padded_labels = []

    for inp, lab in zip(input_ids_list, labels_list):
        pad_len = max_len - len(inp)
        inp_pad = inp + [pad_token_id] * pad_len
        lab_pad = lab + [-100] * pad_len
        padded_inputs.append(torch.tensor(inp_pad, dtype=torch.long))
        padded_labels.append(torch.tensor(lab_pad, dtype=torch.long))

    return torch.stack(padded_inputs), torch.stack(padded_labels)

# ============================================================
# 4. Model Architecture
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
    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

def inject_lora(module, rank, alpha, dropout, target_names=("q_proj","k_proj","v_proj","o_proj")):
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
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(config.max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        cos = self.cos_cached[:, :, :T, :].to(x.device)
        sin = self.sin_cached[:, :, :T, :].to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm_f(x))

# ============================================================
# 5. Model Load & Safety Checks
# ============================================================
print("🔧 Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TeluguGPT(config).to(device)

print("📦 Loading base checkpoint...")
ckpt = torch.load(BASE_MODEL_FILE, map_location="cpu")
state_dict = dict(ckpt.get("model_state_dict", ckpt))

# Strict checkpoint architecture verification
checkpoint_config = ckpt.get("config")
if checkpoint_config is not None:
    expected_architecture = {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "max_seq_len": config.max_seq_len,
        "expansion_factor": config.expansion_factor,
    }
    for key, expected_value in expected_architecture.items():
        if key in checkpoint_config:
            actual_value = checkpoint_config[key]
            if actual_value != expected_value:
                raise RuntimeError(f"Checkpoint architecture mismatch: {key}={actual_value}, script={expected_value}")
    print("✅ Checkpoint architecture verified.")

state_dict.pop("cos_cached", None)
state_dict.pop("sin_cached", None)
state_dict.pop("lm_head.weight", None)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
allowed_missing = {"cos_cached", "sin_cached", "lm_head.weight"}
real_missing = [k for k in missing if k not in allowed_missing]

if real_missing:
    raise RuntimeError(f"❌ Missing keys: {real_missing}")
if unexpected:
    raise RuntimeError(f"❌ Unexpected keys in checkpoint: {unexpected}")

print("✅ All critical keys loaded successfully.")

model.lm_head.weight = model.wte.weight
assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr(), "Tied weights pointer mismatch!"

for p in model.parameters():
    p.requires_grad = False

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
print(f"🧩 Injecting LoRA r={config.lora_rank}...")
inject_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout, LORA_TARGETS)
assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr(), "Tied weights broken after LoRA injection!"

model.to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

# ============================================================
# 6. Data Build & Split
# ============================================================
print("\n📊 Preparing data...")
parsed_pairs = load_and_prepare_data(GITHUB_RAW_URL, config)

rng = random.Random(1234)
rng.shuffle(parsed_pairs)

val_count = max(1, int(len(parsed_pairs) * config.val_split_ratio))
val_pairs = parsed_pairs[:val_count]
train_pairs = parsed_pairs[val_count:]

print(f"📄 Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

train_dataset = TeluguQADataset(train_pairs, tokenizer, config)
val_dataset = TeluguQADataset(val_pairs, tokenizer, config)

micro_batches_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
steps_per_epoch = math.ceil(micro_batches_per_epoch / config.gradient_accumulation_steps)
total_steps = steps_per_epoch * config.num_epochs
warmup_steps = max(1, int(total_steps * config.warmup_ratio))

def collate_wrapper(batch):
    return collate_fn(batch, tokenizer.pad_token_id)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, collate_fn=collate_wrapper)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=collate_wrapper)

# ============================================================
# 7. Optimizer & Scheduler
# ============================================================
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate, betas=(0.9, 0.95))
scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# 8. Generation & Validation
# ============================================================
@torch.no_grad()
def generate_sample(question, max_tokens=256, temperature=0.8, seed=42):
    was_training = model.training
    model.eval()
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        prompt = f"ప్రశ్న: {question}\n\nజవాబు:"
        prompt_ids = tokenizer.encode(prompt)

        # Keep 1/3 head context, preserve 2/3 tail (answer marker)
        if len(prompt_ids) >= config.max_seq_len:
            keep_len = config.max_seq_len - 1
            head_len = keep_len // 3
            tail_len = keep_len - head_len
            prompt_ids = (prompt_ids[:head_len] + prompt_ids[-tail_len:])

        prompt_len = len(prompt_ids)
        max_new_tokens = min(max_tokens, config.max_seq_len - prompt_len - 1)

        if max_new_tokens <= 0:
            return ""

        input_ids = torch.tensor([prompt_ids], device=device)
        for _ in range(max_new_tokens):
            logits = model(input_ids[:, -config.max_seq_len:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_ids = input_ids[0, prompt_len:].tolist()
        return tokenizer.decode(generated_ids).strip()
    finally:
        if was_training:
            model.train()

@torch.no_grad()
def estimate_val_loss():
    was_training = model.training
    model.eval()
    try:
        total_nll = 0.0
        total_valid_tokens = 0
        batches = 0

        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum"
                )

            valid_tokens = (labels != -100).sum().item()
            total_nll += loss.item()
            total_valid_tokens += valid_tokens
            batches += 1

            if config.eval_batches is not None and batches >= config.eval_batches:
                break

        return total_nll / max(1, total_valid_tokens)
    finally:
        if was_training:
            model.train()

# ============================================================
# 9. Adapter Saver
# ============================================================
def save_lora_adapter(model, config, step, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save(lora_state, os.path.join(save_dir, "adapter_model.bin"))

    adapter_config = {
        "adapter_type": "custom_lora",
        "base_repo": REPO_ID,
        "base_model_file": BASE_MODEL_FILE,
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "max_seq_len": config.max_seq_len,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "target_modules": list(LORA_TARGETS),
        "step": step,
        "is_best": is_best,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {save_dir}/ (Step {step})" + (" [BEST]" if is_best else ""))

# ============================================================
# 10. TRAINING LOOP
# ============================================================
model.train()
optimizer.zero_grad(set_to_none=True)

best_val_loss = float('inf')
global_step = 0
epoch = 0

print(f"\n🚀 TRAINING START")
print(f"   Device: {device.type}")
print(f"   Total Steps: {total_steps}")
print("=" * 60)

while epoch < config.num_epochs:
    epoch += 1
    print(f"\n🔄 Epoch {epoch}/{config.num_epochs}")
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    num_micro_batches = micro_batches_per_epoch
    group_token_sum = 0
    group_loss_sum = 0

    for micro_idx, (input_ids, labels) in enumerate(data_iter):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(input_ids)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum"
            )

        valid_tokens = (labels != -100).sum().item()
        group_loss_sum += loss_sum.item()
        group_token_sum += valid_tokens

        scaler.scale(loss_sum).backward()

        is_last_micro = (micro_idx == num_micro_batches - 1)
        if (micro_idx + 1) % config.gradient_accumulation_steps == 0 or is_last_micro:
            scaler.unscale_(optimizer)
            if group_token_sum > 0:
                for p in lora_params:
                    if p.grad is not None:
                        p.grad.div_(group_token_sum)

            torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % config.log_every == 0:
                avg_loss = group_loss_sum / max(1, group_token_sum)
                ppl = math.exp(min(avg_loss, 20))
                lr = scheduler.get_last_lr()[0]
                print(f"📊 Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")

            group_token_sum = 0
            group_loss_sum = 0

            if global_step % config.save_every == 0:
                save_dir = os.path.join(LORA_BASE_DIR, f"step_{global_step}")
                save_lora_adapter(model, config, global_step, save_dir, is_best=False)

            if global_step >= total_steps:
                break

    val_loss = estimate_val_loss()
    val_ppl = math.exp(min(val_loss, 20))
    print(f"   🔍 Epoch {epoch} Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_lora_adapter(model, config, global_step, BEST_ADAPTER_DIR, is_best=True)
        print(f"   🏆 New best validation loss: {best_val_loss:.4f}")

    print(f"\n📝 Samples after Epoch {epoch}:")
    test_qs = ["కాంతి పరావర్తనం అంటే ఏమిట?", "భారతదేశంలో వర్షాకాలం ఎందుకు వస్తుంది?"]
    for q in test_qs:
        ans = generate_sample(q, max_tokens=256, temperature=0.7, seed=42)
        print(f"   Q: {q}\n   A: {ans[:300]}...\n")

    if global_step >= total_steps:
        break

print(f"\n🎉 TRAINING COMPLETE!")
print(f"   Best Val Loss: {best_val_loss:.4f}")
print(f"   Best adapter: {BEST_ADAPTER_DIR}/")
print(f"   Checkpoints: {LORA_BASE_DIR}/step_*/")
