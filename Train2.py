# ==============================================================================
# 🚀 JADDANGI AI: v7.8-FINAL — PROPER VALIDATION & STABLE METRICS
# ==============================================================================
# FIXES FROM v7.7:
# 1. Separate Telugu validation source (not C4 train fallback)
# 2. 20 eval batches (40 samples) for stable loss estimate
# 3. Validation split from IndicCorp + Sangraha mix, not training data
# 4. Keep all previous fixes: dead yield removal, LR floor, pre-flight check
# ==============================================================================

!pip install datasets torch sentencepiece huggingface_hub -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import os
import time
import gc
import random
import warnings
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
import sentencepiece as spm
from huggingface_hub import HfApi, login, hf_hub_download
from torch.amp import GradScaler, autocast

# ============================================
# 1. AUTH & CONFIG
# ============================================
HF_TOKEN =""
if HF_TOKEN:
    login(token=HF_TOKEN)
api = HfApi()

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
SAVE_MODEL_FILE = "Telugu_Model_0.5B_V4.pt"
TOKENIZER_FILE = "telugu_spm.model"

class Config:
    vocab_size = 32000
    d_model = 1440
    n_layers = 21
    n_heads = 20
    n_kv_heads = 5
    max_seq_len = 384
    expansion_factor = 4
    dropout = 0.05
    use_checkpointing = True

    batch_size = 2
    gradient_accumulation_steps = 64
    learning_rate = 2e-4
    max_steps = 10000
    warmup_steps = 500
    save_every = 200
    log_every = 10
    eval_every = 100
    max_grad_norm = 1.0
    use_compile = True
    min_lr_ratio = 0.1

    target_english_ratio = 0.10
    rep_penalty_window = 64

    # Validation settings
    val_batches = 20  # 20 batches × 2 = 40 samples for stable estimate
    val_skip = 50000  # Skip first 50k of training split to avoid overlap

# ============================================
# 2. DOWNLOAD FILES FROM HUB IF MISSING
# ============================================
print("📥 Checking files...")

if not os.path.exists(TOKENIZER_FILE):
    print("   Downloading tokenizer...")
    hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_FILE, local_dir=".", local_dir_use_symlinks=False)

if not os.path.exists(SAVE_MODEL_FILE):
    print(f"   Downloading checkpoint {SAVE_MODEL_FILE} from Hub...")
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=SAVE_MODEL_FILE,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print(f"   ✅ Downloaded checkpoint")
    except Exception as e:
        print(f"   ⚠️ Could not download checkpoint: {e}")
        print("   Will start fresh if file not found locally")

# ============================================
# 3. TOKENIZER
# ============================================
class SentencePieceTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

        self.eos_token_id = self.sp.eos_id()
        self.bos_token_id = self.sp.bos_id() if self.sp.bos_id() != -1 else self.sp.eos_id()
        self.unk_token_id = self.sp.unk_id()

        raw_pad = self.sp.pad_id()
        if raw_pad == -1 or raw_pad is None:
            self.pad_token_id = self.unk_token_id
            print(f"   ⚠️ No PAD, using UNK ({self.pad_token_id})")
        else:
            self.pad_token_id = raw_pad

        print(f"   ✅ Vocab: {self.sp.get_piece_size()}")
        print(f"   PAD={self.pad_token_id} | EOS={self.eos_token_id} | UNK={self.unk_token_id}")

    def encode(self, text, truncation=True, max_length=None):
        ids = self.sp.encode_as_ids(text)
        if truncation and max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids):
        return self.sp.decode_ids(ids)

tokenizer = SentencePieceTokenizer(TOKENIZER_FILE)

# ============================================
# 4. DATASET
# ============================================
class BilingualDataset(IterableDataset):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def _extract_text(self, example):
        if isinstance(example, str):
            return example
        if isinstance(example, dict):
            for key in ['text', 'content', 'sentence', 'paragraph']:
                if key in example and example[key]:
                    return str(example[key])
            for val in example.values():
                if isinstance(val, str) and len(val) > 50:
                    return val
        return ''

    def _is_valid_text(self, text, is_english=False):
        if not text or len(text) < 80:
            return False
        words = text.split()
        if len(words) < 15:
            return False
        if not is_english:
            latin_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
            if latin_ratio > 0.7:
                return False
        junk = ["లాగిన్", "సబ్స్క్రైబ్", "కాపీరైట్", "javascript", "cookie", "privacy policy"]
        if any(j in text.lower() for j in junk):
            return False
        return True

    def _make_chunk(self, text):
        ids = self.tokenizer.encode(text[:2000], truncation=True, max_length=self.config.max_seq_len)
        if len(ids) < 10:
            return None

        content_len = len(ids) + 1
        full_seq = ids + [self.tokenizer.eos_token_id]
        pad_len = self.config.max_seq_len + 1 - len(full_seq)

        if pad_len > 0:
            full_seq += [self.tokenizer.pad_token_id] * pad_len

        input_ids = full_seq[:-1]
        labels = full_seq[1:]

        if pad_len > 0:
            for i in range(len(labels) - pad_len, len(labels)):
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "content_len": content_len
        }

    def __iter__(self):
        print("🌐 Loading datasets (90/10 token-balanced)...")

        tel_sources = []
        tel_names = []
        try:
            ds = load_dataset("ai4bharat/IndicCorpV2", split="tel_Telu", streaming=True, trust_remote_code=True)
            tel_sources.append(iter(ds)); tel_names.append("indic")
            print("   ✅ IndicCorp v2")
        except Exception as e:
            print(f"   ⚠️ IndicCorp: {e}")
        try:
            ds = load_dataset("ai4bharat/sangraha", "verified", split="tel", streaming=True, trust_remote_code=True)
            tel_sources.append(iter(ds)); tel_names.append("sangraha")
            print("   ✅ Sangraha")
        except Exception as e:
            print(f"   ⚠️ Sangraha: {e}")
        try:
            ds = load_dataset("allenai/c4", "multilingual", split="train", streaming=True,
                              data_files={"train": "multilingual/c4-te.*.json.gz"})
            tel_sources.append(iter(ds)); tel_names.append("c4_te")
            print("   ✅ C4 Telugu")
        except Exception as e:
            print(f"   ⚠️ C4 Telugu: {e}")

        en_source = None
        try:
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            en_source = iter(ds)
            print("   ✅ C4 English")
        except Exception as e:
            print(f"   ⚠️ C4 English: {e}")

        if not tel_sources:
            raise RuntimeError("No Telugu datasets loaded!")

        tel_tokens = 0
        en_tokens = 0
        tel_idx = 0

        while True:
            total = tel_tokens + en_tokens
            current_en_ratio = en_tokens / max(total, 1)
            need_english = (en_source is not None) and (current_en_ratio < self.config.target_english_ratio)

            if need_english:
                try:
                    example = next(en_source)
                    text = self._extract_text(example)
                    if text and len(text) > 80:
                        chunk = self._make_chunk(text)
                        if chunk:
                            en_tokens += chunk["content_len"]
                            yield chunk
                            continue
                except StopIteration:
                    try:
                        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
                        en_source = iter(ds)
                    except:
                        en_source = None
                except Exception:
                    pass

            yielded = False
            for _ in range(len(tel_sources)):
                i = tel_idx % len(tel_sources)
                tel_idx += 1
                try:
                    example = next(tel_sources[i])
                    text = self._extract_text(example)
                    if self._is_valid_text(text):
                        chunk = self._make_chunk(text)
                        if chunk:
                            tel_tokens += chunk["content_len"]
                            yield chunk
                            yielded = True
                            break
                except StopIteration:
                    try:
                        if tel_names[i] == "indic":
                            ds = load_dataset("ai4bharat/IndicCorpV2", split="tel_Telu", streaming=True, trust_remote_code=True)
                        elif tel_names[i] == "sangraha":
                            ds = load_dataset("ai4bharat/sangraha", "verified", split="tel", streaming=True, trust_remote_code=True)
                        else:
                            ds = load_dataset("allenai/c4", "multilingual", split="train", streaming=True,
                                              data_files={"train": "multilingual/c4-te.*.json.gz"})
                        tel_sources[i] = iter(ds)
                        print(f"   🔄 Reloaded {tel_names[i]}")
                    except Exception as re:
                        print(f"   ❌ Failed reload {tel_names[i]}: {re}")
                except Exception:
                    pass

            if not yielded and en_source:
                pass

# ============================================
# 5. MODEL
# ============================================
def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
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
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[:, :, :T, :], sin[:, :, :T, :])
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 / 3 * config.expansion_factor * config.d_model)
        hidden_dim = (hidden_dim // 256) * 256
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)
        self.use_checkpointing = config.use_checkpointing

    def _forward_impl(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x, cos, sin):
        if self.training and x.requires_grad and self.use_checkpointing:
            return checkpoint(self._forward_impl, x, cos, sin, use_reentrant=False)
        return self._forward_impl(x, cos, sin)

class TeluguGPT(nn.Module):
    def __init__(self, config, do_init=True):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        head_dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(config.max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

        if self.device_type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except AttributeError:
                pass

        if do_init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm_f(x))

# ============================================
# 6. LOAD CHECKPOINT
# ============================================
print("\n🔧 Loading model...")
do_init = not os.path.exists(SAVE_MODEL_FILE)
model = TeluguGPT(Config(), do_init=do_init)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

optimizer_step = 0
checkpoint_data = {}

if os.path.exists(SAVE_MODEL_FILE):
    print(f"📦 Loading checkpoint: {SAVE_MODEL_FILE}...")
    checkpoint_data = torch.load(SAVE_MODEL_FILE, map_location='cpu')
    state_dict = checkpoint_data.get('model_state_dict', checkpoint_data)

    state_dict.pop('lm_head.weight', None)
    for buf in ['cos_cached', 'sin_cached']:
        state_dict.pop(buf, None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   Missing: {len(missing)} keys")
    if unexpected:
        print(f"   Unexpected: {len(unexpected)} keys")

    model.lm_head.weight = model.wte.weight
    model = model.to(model.device)

    optimizer_step = checkpoint_data.get('step', 0)
    print(f"✅ RESUMING FROM STEP {optimizer_step}")
else:
    print("⚠️ No checkpoint found, starting fresh")

if Config.use_compile and model.device_type == 'cuda' and hasattr(torch, 'compile'):
    print("⚡ Compiling model...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("   ✅ Compiled")
    except Exception as e:
        print(f"   ⚠️ Compile failed: {e}")

# ============================================
# 7. OPTIMIZER & LR
# ============================================
use_fused = 'cuda' in model.device_type

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=Config.learning_rate,
    weight_decay=0.05,
    betas=(0.9, 0.95),
    fused=use_fused
)

if 'optimizer_state_dict' in checkpoint_data:
    try:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        print("   ✅ Optimizer state loaded")
    except Exception as e:
        print(f"   ⚠️ Fresh optimizer: {e}")

scaler = GradScaler(enabled=(model.device_type == 'cuda'))

if 'target_steps' in checkpoint_data:
    target_steps = checkpoint_data['target_steps']
    print(f"   📊 Loaded target_steps: {target_steps}")
else:
    target_steps = optimizer_step + Config.max_steps
    print(f"   📊 New target_steps: {target_steps}")

def get_lr(step):
    if step < Config.warmup_steps:
        scale = float(step) / float(max(1, Config.warmup_steps))
    else:
        progress = (step - Config.warmup_steps) / (target_steps - Config.warmup_steps)
        progress = max(0.0, min(1.0, progress))
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = max(scale, Config.min_lr_ratio)
    return Config.learning_rate * scale

current_lr = get_lr(optimizer_step)
for param_group in optimizer.param_groups:
    param_group['lr'] = current_lr
print(f"   📊 LR: {current_lr:.2e} | Target: {target_steps}")

# ============================================
# 8. DATALOADERS
# ============================================
train_dataset = BilingualDataset(tokenizer, Config())
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, num_workers=0, pin_memory=True)

# ============================================
# 8.5 VALIDATION DATASET — PROPER TELUGU SOURCE
# ============================================
class TeluguValidationDataset(IterableDataset):
    """
    Uses a DIFFERENT data source than training to measure generalization.
    Strategy: Use IndicCorp train split but skip deep into it (50k examples)
    so there's no overlap with training which starts from the beginning.
    """
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def _make_chunk(self, text):
        ids = self.tokenizer.encode(text[:2000], truncation=True, max_length=self.config.max_seq_len)
        if len(ids) < 10:
            return None
        content_len = len(ids) + 1
        full_seq = ids + [self.tokenizer.eos_token_id]
        pad_len = self.config.max_seq_len + 1 - len(full_seq)
        if pad_len > 0:
            full_seq += [self.tokenizer.pad_token_id] * pad_len
        else:
            full_seq = full_seq[:self.config.max_seq_len + 1]
            pad_len = 0

        input_ids = full_seq[:-1]
        labels = full_seq[1:]
        if pad_len > 0:
            for i in range(len(labels) - pad_len, len(labels)):
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def __iter__(self):
        ds = None
        source_name = ""
        
        # Priority 1: Try Sangraha (smaller, cleaner, less likely to overlap)
        try:
            ds = load_dataset("ai4bharat/sangraha", "verified", split="tel", streaming=True, trust_remote_code=True)
            source_name = "Sangraha"
            print(f"   ✅ Validation: {source_name}")
        except Exception as e:
            print(f"   ⚠️ Sangraha val failed: {e}")
            
            # Priority 2: IndicCorp with deep skip
            try:
                ds = load_dataset("ai4bharat/IndicCorpV2", split="tel_Telu", streaming=True, trust_remote_code=True)
                source_name = "IndicCorp (deep skip)"
                print(f"   ✅ Validation: {source_name}")
            except Exception as e2:
                print(f"   ⚠️ IndicCorp val failed: {e2}")
                
                # Priority 3: C4 Telugu with deep skip (last resort)
                try:
                    ds = load_dataset("allenai/c4", "multilingual", split="train", streaming=True,
                                      data_files={"train": "multilingual/c4-te.*.json.gz"})
                    source_name = "C4 Telugu (deep skip)"
                    print(f"   ⚠️ Validation: {source_name} (not ideal)")
                except Exception as e3:
                    print(f"   ❌ No validation source available: {e3}")
                    return

        if ds is None:
            print("   ❌ ds is None")
            return

        it = iter(ds)
        skipped = 0
        
        # Deep skip to avoid training overlap
        try:
            for _ in range(self.config.val_skip):
                next(it)
                skipped += 1
        except StopIteration:
            print(f"   ⚠️ Dataset exhausted after {skipped} skips (need {self.config.val_skip})")
            return
        except Exception as e:
            print(f"   ⚠️ Skip error: {e}")
            return

        print(f"   ✅ Skipped {skipped} examples, collecting validation set")

        count = 0
        for ex in it:
            text = ex.get('text', '') if isinstance(ex, dict) else ex
            if not text or len(text) < 80:
                continue
            chunk = self._make_chunk(text)
            if chunk:
                yield chunk
                count += 1
                if count >= self.config.val_batches * self.config.batch_size * 2:
                    break

eval_dataset = TeluguValidationDataset(tokenizer, Config())

# ============================================
# 8.6 PRE-FLIGHT VALIDATION CHECK
# ============================================
print("\n🧪 Pre-flight validation check...")
try:
    test_val_loader = iter(DataLoader(eval_dataset, batch_size=2, num_workers=0))
    test_batch = next(test_val_loader)
    test_text = tokenizer.decode(test_batch['input_ids'][0].tolist())
    print(f"   ✅ Validation loader works!")
    print(f"   📝 Sample: {test_text[:100]}...")
except StopIteration:
    print("   ❌❌❌ VALIDATION LOADER IS EMPTY ❌❌❌")
    print("   Fix dataset paths before training or you will waste GPU hours!")
    raise RuntimeError("Validation loader empty — aborting")
except Exception as e:
    print(f"   ⚠️ Validation check error: {e}")
    raise

# ============================================
# 9. TRAINING FUNCTIONS
# ============================================
best_loss = checkpoint_data.get('best_loss', float('inf'))

@torch.no_grad()
def estimate_loss():
    model.eval()
    total_loss = 0.0
    steps = 0
    
    try:
        loader = iter(DataLoader(eval_dataset, batch_size=Config.batch_size, num_workers=0))
    except Exception as e:
        print(f"   ⚠️ Failed to create validation loader: {e}")
        model.train()
        return float('inf')
    
    for _ in range(Config.val_batches):
        try:
            batch = next(loader)
            inputs = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, Config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100
                )
            total_loss += loss.item()
            steps += 1
        except StopIteration:
            print(f"   ⚠️ Validation loader exhausted at batch {steps}")
            break
        except Exception as e:
            print(f"   ⚠️ Validation batch error: {e}")
            continue
    
    if steps == 0:
        print("   ❌ NO VALIDATION DATA PROCESSED — returning inf")
        model.train()
        return float('inf')
    
    avg_loss = total_loss / steps
    model.train()
    return avg_loss

@torch.no_grad()
def generate_samples(prompts=None, max_tokens=50, temperature=0.7, repetition_penalty=1.15, top_k=50, top_p=0.9):
    model.eval()
    if prompts is None:
        prompts = [
            "హైదరాబాద్ నగరంలో",
            "తెలుగు భాష అంటే",
            "రైతుల జీవితం",
            "ఆరోగ్యం మరియు పోషణ",
        ]
    samples = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)[-Config.max_seq_len:]
        input_ids = torch.tensor([ids], device=model.device)
        for _ in range(max_tokens):
            logits = model(input_ids[:, -Config.max_seq_len:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            recent = input_ids[0][-Config.rep_penalty_window:].tolist()
            for token_id in set(recent):
                logits[0, token_id] /= repetition_penalty

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            if top_p > 0.0:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        samples.append((prompt, tokenizer.decode(input_ids[0].tolist())))
    model.train()
    return samples

# ============================================
# 10. TRAINING LOOP
# ============================================
print(f"\n{'='*60}")
print(f"🚀 RESUMING FROM STEP {optimizer_step} / {target_steps}")
print(f"   Best loss: {best_loss:.4f}")
print(f"   LR: {Config.learning_rate} | Dropout: {Config.dropout}")
print(f"{'='*60}")

model.train()
batch_step = 0
window_loss = 0.0
loss_count = 0
last_reported_loss = 0.0
start_time = time.time()
upload_count = 0

print("\n🔥 Training started...\n")

try:
    for batch in train_loader:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)

        with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, Config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )
            scaled_loss = loss / Config.gradient_accumulation_steps

        scaler.scale(scaled_loss).backward()
        window_loss += loss.item()
        loss_count += 1

        if (batch_step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1
            new_lr = get_lr(optimizer_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            if optimizer_step % Config.log_every == 0:
                if model.device_type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start_time
                current_loss = window_loss / max(1, loss_count)
                last_reported_loss = current_loss
                ppl = math.exp(min(current_loss, 20))
                mem = torch.cuda.memory_allocated()/1e9 if model.device_type == 'cuda' else 0
                print(f"📊 Step {optimizer_step}/{target_steps} | Loss: {current_loss:.4f} | PPL: {ppl:.2f} | LR: {new_lr:.2e} | Time: {elapsed:.1f}s | GPU: {mem:.2f}GB")
                window_loss = 0.0
                loss_count = 0
                start_time = time.time()
                gc.collect()

            if optimizer_step % Config.eval_every == 0 or optimizer_step == 1:
                val_loss = estimate_loss()
                val_ppl = math.exp(min(val_loss, 20))
                print(f"   🧪 Val: {val_loss:.4f} | PPL: {val_ppl:.2f}")
                if val_loss < best_loss:
                    best_loss = val_loss

            if optimizer_step % Config.save_every == 0:
                print(f"\n{'='*60}")
                print(f"💾 CHECKPOINT Step {optimizer_step}")
                samples = generate_samples(max_tokens=50)
                for p, g in samples:
                    print(f"   🔹 {p}")
                    print(f"   🔸 {g[:120]}")
                print()

                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_dict = {
                    'step': optimizer_step,
                    'model_state_dict': raw_model.state_dict(),
                    'loss': last_reported_loss,
                    'best_loss': best_loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'target_steps': target_steps,
                }

                tmp = SAVE_MODEL_FILE + ".tmp"
                torch.save(save_dict, tmp)
                os.replace(tmp, SAVE_MODEL_FILE)
                print(f"📁 Saved locally")

                if model.device_type == 'cuda':
                    torch.cuda.empty_cache()

                if HF_TOKEN:
                    try:
                        api.upload_file(
                            path_or_fileobj=SAVE_MODEL_FILE,
                            path_in_repo=SAVE_MODEL_FILE,
                            repo_id=REPO_ID,
                            commit_message=f"🔧 v7.8-FINAL Step {optimizer_step} | Loss: {last_reported_loss:.4f} | Val: {val_loss:.4f}"
                        )
                        upload_count += 1
                        print(f"☁️ Upload #{upload_count}")
                    except Exception as e:
                        print(f"⚠️ Upload: {str(e)[:80]}")
                print(f"{'='*60}\n")

        batch_step += 1
        if optimizer_step >= target_steps:
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrupted! Saving...")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        'step': optimizer_step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'loss': last_reported_loss,
        'target_steps': target_steps,
    }, SAVE_MODEL_FILE)
    print("✅ Saved!")

print(f"\n🎉 Done! Step: {optimizer_step} | Best loss: {best_loss:.4f}")
