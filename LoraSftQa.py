#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
తెలుగు LoRA SFT - FINAL v6 (FULLY PRODUCTION)
===============================================
Final Fixes Applied:
- Correct token-weighted gradient accumulation (sum over group / group_token_count)
- Balanced head+tail truncation for generation (preserves both context and answer marker)
- Strict validation of unexpected checkpoint keys
- Tied-weight pointer assertion for safety
- All previous fixes (vocab assertion, safe eval mode, stable accumulation, etc.)
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
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️  HF_TOKEN env లో సెట్ కాలేదు.")

USE_SCALED_ROPE = False

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

    learning_rate = 3e-5
    warmup_ratio = 0.05
    max_grad_norm = 1.0

    num_epochs = 2
    min_doc_length = 40
    min_answer_tokens = 4
    val_split_ratio = 0.10

    eval_batches = None          # Full validation set
    save_every = 100
    log_every = 25

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
BASE_MODEL_FILE = "Telugu_Model_0.5B_V4.pt"
TOKENIZER_FILE = "telugu_spm.model"
LORA_BASE_DIR = "Qasft"
BEST_ADAPTER_DIR = "Qasft_best"
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
        print(f"PAD: {self.pad_token_id}, EOS: {self.eos_token_id}")

    def encode(self, text, max_length=None):
        ids = self.sp.encode_as_ids(text)
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids):
        return self.sp.decode_ids(ids)

tokenizer = SentencePieceTokenizer(TOKENIZER_FILE)

config = Config()
actual_vocab = tokenizer.vocab_size
if actual_vocab != config.vocab_size:
    raise RuntimeError(
        f"❌ VOCAB MISMATCH! Tokenizer has {actual_vocab} tokens, "
        f"but model config expects {config.vocab_size}."
    )
print(f"✅ Vocabulary match: {config.vocab_size}")

# ============================================================
# 2. Data Preparation (Filter BEFORE Split)
# ============================================================
def normalize_for_dedup(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def parse_and_filter_pairs(raw_pairs, config):
    parsed = []
    seen = set()
    
    for pair in raw_pairs:
        parts = re.split(r'\n+జవాబు\s*:', pair, maxsplit=1)
        if len(parts) != 2:
            continue
        q_part = parts[0].strip()
        a_part = parts[1].strip()

        if q_part.startswith("ప్రశ్న"):
            q_part = q_part[len("ప్రశ్న"):].strip()
            if q_part.startswith(":"):
                q_part = q_part[1:].strip()

        if len(q_part) + len(a_part) < config.min_doc_length:
            continue

        norm_q = normalize_for_dedup(q_part)
        norm_a = normalize_for_dedup(a_part)
        h = hashlib.sha256((norm_q + "|||" + norm_a).encode('utf-8')).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        
        parsed.append((q_part, a_part))
    
    print(f"✅ After parse & dedup: {len(parsed)} valid pairs.")
    return parsed

def load_and_prepare_data(url, config):
    print(f"📥 Downloading from: {url}")
    resp = requests.get(url)
    resp.encoding = "utf-8"
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed: {resp.status_code}")
    text = resp.text

    raw_pairs = re.split(r'\n\n(?=ప్రశ్న:)', text)
    raw_pairs = [p.strip() for p in raw_pairs if p.strip()]
    if len(raw_pairs) < 5:
        raw_pairs = [p.strip() for p in text.split("\n") if p.strip()]

    print(f"📄 Raw pairs: {len(raw_pairs)}")
    return parse_and_filter_pairs(raw_pairs, config)

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

            # Answer-only loss masking (Correctly aligned)
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
        self.lm_head.weight = self.wte.weight  # Tied weights
        head_dim = config.d_model // config.n_heads
        if USE_SCALED_ROPE:
            scale_factor = config.max_seq_len / 384
            adjusted_theta = config.rope_theta * (scale_factor ** (head_dim / (head_dim - 2)))
            inv_freq = 1.0 / (adjusted_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            print("🔧 Scaled RoPE enabled")
        else:
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            print("🔧 Standard RoPE")
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
# 5. Model Load, LoRA Injection & Safety Checks
# ============================================================
print("🔧 Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TeluguGPT(config).to(device)

print("📦 Loading base checkpoint...")
ckpt = torch.load(BASE_MODEL_FILE, map_location="cpu")
state_dict = dict(ckpt.get("model_state_dict", ckpt))
state_dict.pop("cos_cached", None)
state_dict.pop("sin_cached", None)
state_dict.pop("lm_head.weight", None)  # Tied, will be set from wte

# Strict validation of checkpoint keys
missing, unexpected = model.load_state_dict(state_dict, strict=False)
allowed_missing = {"cos_cached", "sin_cached", "lm_head.weight"}
real_missing = [k for k in missing if k not in allowed_missing]

if real_missing:
    raise RuntimeError(f"❌ Missing keys: {real_missing}")
if unexpected:
    raise RuntimeError(f"❌ Unexpected keys in checkpoint: {unexpected}")

print("✅ All critical keys loaded successfully.")

# Re-tie weights and assert pointer equality
model.lm_head.weight = model.wte.weight
assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr(), "Tied weights pointer mismatch!"

# Freeze base parameters
for p in model.parameters():
    p.requires_grad = False

# Inject LoRA
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
print(f"🧩 Injecting LoRA r={config.lora_rank}...")
inject_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout, LORA_TARGETS)

# Verify tied weights still hold after LoRA injection
assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr(), "Tied weights broken after LoRA injection!"

model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
assert trainable > 0

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

print("\n" + "="*60)
print("📈 RUN STATISTICS")
print(f"   Training examples: {len(train_dataset):,}")
print(f"   Validation examples: {len(val_dataset):,}")
print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
print(f"   Steps per epoch: {steps_per_epoch:,}")
print(f"   Total optimizer steps: {total_steps:,}")
print(f"   Warmup steps: {warmup_steps:,}")
print("="*60 + "\n")

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
scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available())

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# 8. Generation (Safe Context Manager) & Validation
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
        
        # --- FIX: Balanced head+tail truncation ---
        # Preserves both the beginning (context) and the end (జవాబు: marker)
        if len(prompt_ids) >= config.max_seq_len:
            keep_len = config.max_seq_len - 1
            half = keep_len // 2
            prompt_ids = prompt_ids[:half] + prompt_ids[-(keep_len - half):]
        
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
# 9. Adapter Saver (Versioned)
# ============================================================
def save_lora_adapter(model, config, step, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    lora_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, os.path.join(save_dir, "adapter_model.bin"))

    adapter_config = {
        "adapter_type": "custom_lora",
        "base_repo": REPO_ID,
        "base_model_file": BASE_MODEL_FILE,
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "max_seq_len": config.max_seq_len,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(LORA_TARGETS),
        "step": step,
        "is_best": is_best,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {save_dir}/ (Step {step})" + (" [BEST]" if is_best else ""))

# ============================================================
# 10. TRAINING LOOP (Correct Token-Weighted Gradient Accumulation)
# ============================================================
model.train()
optimizer.zero_grad(set_to_none=True)

best_val_loss = float('inf')
global_step = 0
epoch = 0

print(f"🚀 TRAINING START")
print(f"   Device: {device.type}")
print(f"   Total Steps: {total_steps}")
print("=" * 60)

while epoch < config.num_epochs:
    epoch += 1
    print(f"\n🔄 Epoch {epoch}/{config.num_epochs}")
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    num_micro_batches = micro_batches_per_epoch
    
    # Accumulators for the group
    group_token_sum = 0  # Total valid tokens in the group (for gradient scaling)
    group_loss_sum = 0   # Sum of NLL (for logging)

    for micro_idx, (input_ids, labels) in enumerate(data_iter):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Stable divisor per group (only for knowing when to step)
        group_start = (micro_idx // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
        group_end = min(group_start + config.gradient_accumulation_steps, num_micro_batches)
        effective_accum = group_end - group_start

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(input_ids)
            
            # --- FIX: TRUE TOKEN-WEIGHTED OBJECTIVE ---
            # Compute sum of NLL over valid answer tokens
            loss_sum = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum"
            )

        # Accumulate loss sum and token count for the group
        valid_tokens = (labels != -100).sum().item()
        group_loss_sum += loss_sum.item()
        group_token_sum += valid_tokens

        # Backward pass with loss_sum (gradients are summed across the group)
        scaler.scale(loss_sum).backward()

        # Step condition
        is_last_micro = (micro_idx == num_micro_batches - 1)
        if (micro_idx + 1) % config.gradient_accumulation_steps == 0 or is_last_micro:

            # --- FIX: Normalize gradients by total token count ---
            scaler.unscale_(optimizer)
            if group_token_sum > 0:
                for p in lora_params:
                    if p.grad is not None:
                        p.grad.div_(group_token_sum)

            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Logging: token-weighted average loss
            if global_step % config.log_every == 0:
                avg_loss = group_loss_sum / max(1, group_token_sum)
                ppl = math.exp(min(avg_loss, 20))
                lr = scheduler.get_last_lr()[0]
                print(f"📊 Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")

            # Reset group accumulators
            group_token_sum = 0
            group_loss_sum = 0

            # Save versioned checkpoint
            if global_step % config.save_every == 0:
                save_dir = os.path.join(LORA_BASE_DIR, f"step_{global_step}")
                save_lora_adapter(model, config, global_step, save_dir, is_best=False)

            if global_step >= total_steps:
                break

    # End of Epoch: Validation & Best Save
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
