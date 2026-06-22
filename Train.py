# ==============================================================================
# 🚀 JADDANGI AI: PHASE-4 v7.1 FINAL - 3 DATASETS FIXED
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
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
import sentencepiece as spm
from huggingface_hub import HfApi, login, hf_hub_download
from torch.amp import GradScaler, autocast

# ============================================
# 1. AUTH & CONFIG
# ============================================
# 🔴 SECURITY: Replace with Colab secrets or env var!
# from google.colab import userdata
# HF_TOKEN = userdata.get('HF_TOKEN')
HF_TOKEN = ""
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
    dropout = 0.2
    use_checkpointing = True
    
    batch_size = 2
    gradient_accumulation_steps = 64
    learning_rate = 3e-4
    max_steps = 10000
    warmup_steps = 500
    save_every = 200          # Save, Upload & Generate Sample every 200 steps
    log_every = 10
    eval_every = 100
    max_grad_norm = 1.0
    use_compile = False
    save_optimizer_every = 1000
    
    # Dataset ratios
    indicorp_ratio = 0.4      # 40% IndicCorp v2
    c4_ratio = 0.2            # 20% C4 Telugu
    sangraha_ratio = 0.4      # 40% Sangraha verified

# ============================================
# 2. TOKENIZER & ULTRA-FAST DATASET
# ============================================
print("📥 Loading tokenizer and model...")
if not os.path.exists(TOKENIZER_FILE):
    hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER_FILE, local_dir=".", local_dir_use_symlinks=False)
if not os.path.exists(SAVE_MODEL_FILE):
    print(f"   Downloading checkpoint...")
    hf_hub_download(repo_id=REPO_ID, filename=SAVE_MODEL_FILE, local_dir=".", local_dir_use_symlinks=False)

class SentencePieceTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.pad_token_id = self.sp.pad_id()
        self.eos_token_id = self.sp.eos_id()
    
    def encode(self, text, truncation=True, max_length=None):
        ids = self.sp.encode_as_ids(text)
        if truncation and max_length:
            ids = ids[:max_length]
        return ids
    
    def decode(self, ids):
        return self.sp.decode_ids(ids)

tokenizer = SentencePieceTokenizer(TOKENIZER_FILE)

class UltraFastTeluguDataset(IterableDataset):
    """Maximum speed dataset - IndicCorp 40%, C4 20%, Sangraha 40%"""
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.buffer = []
        self.total_yielded = 0
    
    def _extract_text(self, example):
        """Fast text extraction from any dataset format"""
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
    
    def _is_valid_telugu(self, text):
        """Ultra-fast Telugu validation"""
        if len(text) < 80:
            return False
        sample = text[:500]
        telugu_chars = sum(1 for c in sample if '\u0C00' <= c <= '\u0C7F')
        return telugu_chars > 30
    
    def __iter__(self):
        print("⚡ Loading 3 Telugu datasets...")
        print(f"   📊 Ratios: IndicCorp v2 ({self.config.indicorp_ratio*100:.0f}%) | C4 ({self.config.c4_ratio*100:.0f}%) | Sangraha ({self.config.sangraha_ratio*100:.0f}%)")
        
        # Test and load each source exactly as specified
        print("1. IndicCorp v2 (telugu)...")
        ds1 = load_dataset("ai4bharat/IndicCorpV2", split="tel_Telu", streaming=True)
        print("   ✅ OK")
        
        print("2. C4 (te)...")
        ds2 = load_dataset("allenai/c4", "te", split="train", streaming=True)
        print("   ✅ OK")
        
        print("3. Sangraha (te)...")
        ds3 = load_dataset("ai4bharat/sangraha", "verified", split="tel", streaming=True)
        print("   ✅ OK")
        
        # Interleave datasets
        combined = interleave_datasets(
            [ds1, ds2, ds3],
            probabilities=[self.config.indicorp_ratio, self.config.c4_ratio, self.config.sangraha_ratio],
            seed=random.randint(0, 10000),
            stopping_strategy="all_exhausted"
        )
        
        processed = 0
        last_log_time = time.time()
        
        for example in combined:
            text = self._extract_text(example)
            
            if not self._is_valid_telugu(text):
                continue
            
            ids = self.tokenizer.encode(
                text[:2000],
                truncation=True,
                max_length=self.config.max_seq_len
            )
            
            self.buffer.extend(ids + [self.tokenizer.eos_token_id])
            
            while len(self.buffer) >= self.config.max_seq_len + 1:
                chunk = self.buffer[:self.config.max_seq_len + 1]
                self.buffer = self.buffer[self.config.max_seq_len + 1:]
                self.total_yielded += 1
                
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long)
                }
            
            processed += 1
            
            if processed % 500 == 0 and time.time() - last_log_time > 30:
                print(f"   ⚡ {processed} texts → {self.total_yielded} chunks | Buffer: {len(self.buffer)}")
                last_log_time = time.time()

class FastEvalDataset(IterableDataset):
    """Fast evaluation dataset"""
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
    
    def __iter__(self):
        ds_eval = load_dataset("allenai/c4", "te", split="validation", streaming=True)
        buffer = []
        for example in ds_eval:
            text = example.get('text', '')
            if len(text) < 80:
                continue
            ids = self.tokenizer.encode(text[:2000], truncation=True, max_length=self.config.max_seq_len)
            buffer.extend(ids + [self.tokenizer.eos_token_id])
            while len(buffer) >= self.config.max_seq_len + 1:
                chunk = buffer[:self.config.max_seq_len + 1]
                buffer = buffer[self.config.max_seq_len + 1:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long)
                }

# ============================================
# 3. MODEL ARCHITECTURE
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
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
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
    def __init__(self, config):
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
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        
        cos = self.cos_cached[:, :, :T, :].to(x.device)
        sin = self.sin_cached[:, :, :T, :].to(x.device)
        
        for block in self.blocks:
            x = block(x, cos, sin)
        
        return self.lm_head(self.norm_f(x))

# ============================================
# 4. LOAD CHECKPOINT
# ============================================
print("\n🔧 Initializing model...")
model = TeluguGPT(Config())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

use_fused = 'cuda' in model.device_type
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=Config.learning_rate,
    weight_decay=0.05,
    betas=(0.9, 0.95),
    fused=use_fused
)
scaler = GradScaler(model.device_type, enabled=(model.device_type == 'cuda'))
optimizer_step = 0

if os.path.exists(SAVE_MODEL_FILE):
    print(f"📦 Loading checkpoint: {SAVE_MODEL_FILE}...")
    checkpoint_data = torch.load(SAVE_MODEL_FILE, map_location='cpu')
    state_dict = checkpoint_data.get('model_state_dict', checkpoint_data)
    
    state_dict.pop('lm_head.weight', None)
    for buf in ['cos_cached', 'sin_cached']:
        if buf in state_dict:
            print(f"   Removing old {buf}")
            state_dict.pop(buf)
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   Missing keys: {len(missing)}")
    if unexpected:
        print(f"   Unexpected keys: {len(unexpected)}")
    
    model.lm_head.weight = model.wte.weight
    model = model.to(model.device)
    
    if 'optimizer_state_dict' in checkpoint_data:
        try:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            print("   ✅ Optimizer loaded")
        except:
            print("   ⚠️ Fresh optimizer")
    
    optimizer_step = checkpoint_data.get('step', 0)
    print(f"✅ Resuming from Step {optimizer_step}")
else:
    print("⚠️ Starting from scratch")

if Config.use_compile and hasattr(torch, 'compile') and model.device_type == 'cuda':
    print("⚡ Compiling model...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except:
        pass

# ============================================
# 5. DATALOADERS & SCHEDULER
# ============================================
print("\n📊 Setting up data pipeline...")
train_dataset = UltraFastTeluguDataset(tokenizer, Config())
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    num_workers=0,
    pin_memory=True
)

eval_dataset = FastEvalDataset(tokenizer, Config())

def lr_lambda(step):
    if step < Config.warmup_steps:
        return float(step) / float(max(1, Config.warmup_steps))
    progress = (step - Config.warmup_steps) / (Config.max_steps - Config.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, progress))))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

for _ in range(optimizer_step):
    scheduler.step()

# ============================================
# 6. TRAINING LOOP
# ============================================
print(f"\n🚀 TRAINING START: 0.5B Telugu Model")
print(f"   Device: {model.device_type.upper()}")
print(f"   Datasets: IndicCorp v2 (40%) + C4 (20%) + Sangraha (40%)")
print(f"   Resume Step: {optimizer_step}/{Config.max_steps}")
print(f"   Save & Upload + Sample: Every {Config.save_every} steps")
print("=" * 60)

@torch.no_grad()
def estimate_loss():
    model.eval()
    total_loss = 0.0
    steps = 0
    
    temp_loader = iter(DataLoader(eval_dataset, batch_size=Config.batch_size))
    for _ in range(10):
        try:
            batch = next(temp_loader)
            inputs = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, Config.vocab_size), labels.reshape(-1))
            
            total_loss += loss.item()
            steps += 1
        except StopIteration:
            break
    
    model.train()
    return total_loss / max(1, steps)

@torch.no_grad()
def generate_samples(prompts=None, max_tokens=50, temperature=0.8):
    """Generate multiple text samples"""
    model.eval()
    
    if prompts is None:
        prompts = [
            "హైదరాబాద్ నగరంలో",
            "తెలుగు భాష అంటే",
            "భారతదేశంలో విద్య",
            "రైతుల జీవితం",
            "ఆరోగ్యం మరియు పోషణ"
        ]
    
    samples = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)[-Config.max_seq_len:]
        input_ids = torch.tensor([ids], device=model.device)
        
        for _ in range(max_tokens):
            logits = model(input_ids[:, -Config.max_seq_len:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        generated = tokenizer.decode(input_ids[0].tolist())
        samples.append((prompt, generated))
    
    model.train()
    return samples

model.train()
batch_step = 0
window_loss = 0.0
loss_count = 0
last_reported_loss = 0.0
start_time = time.time()
best_loss = float('inf')
upload_count = 0

print("\n🔥 Starting training loop...\n")

for batch in train_loader:
    input_ids = batch["input_ids"].to(model.device, non_blocking=True)
    labels = batch["labels"].to(model.device, non_blocking=True)
    
    with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, Config.vocab_size),
            labels.reshape(-1)
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
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        optimizer_step += 1
        
        # ============ LOGGING ============
        if optimizer_step % Config.log_every == 0:
            if model.device_type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            current_loss = window_loss / max(1, loss_count)
            last_reported_loss = current_loss
            ppl = math.exp(min(current_loss, 20))
            lr = scheduler.get_last_lr()[0]
            
            if model.device_type == 'cuda':
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"📊 Step {optimizer_step}/{Config.max_steps} | Loss: {current_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | Time: {elapsed:.1f}s | GPU: {mem_alloc:.2f}GB")
            else:
                print(f"📊 Step {optimizer_step}/{Config.max_steps} | Loss: {current_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")
            
            window_loss = 0.0
            loss_count = 0
            start_time = time.time()
            
            gc.collect()
            if model.device_type == 'cuda':
                torch.cuda.empty_cache()
        
        # ============ EVALUATION (Every 100 steps) ============
        if optimizer_step % Config.eval_every == 0 or optimizer_step == 1:
            val_loss = estimate_loss()
            val_ppl = math.exp(min(val_loss, 20))
            
            print(f"   🧪 Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"   🏆 New best val loss: {best_loss:.4f}")
        
        # ============ SAVE, UPLOAD & SAMPLE (Every 200 steps) ============
        if optimizer_step % Config.save_every == 0:
            print(f"\n{'='*60}")
            print(f"💾 CHECKPOINT at Step {optimizer_step}")
            print(f"{'='*60}")
            
            # Generate samples
            print(f"📝 Generating samples...")
            samples = generate_samples(max_tokens=50, temperature=0.8)
            for prompt, generated in samples:
                print(f"   🔹 Prompt: {prompt}")
                print(f"   🔸 Generated: {generated}")
                print()
            
            # Save locally
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_dict = {
                'step': optimizer_step,
                'model_state_dict': raw_model.state_dict(),
                'loss': last_reported_loss,
                'best_loss': best_loss,
            }
            
            if optimizer_step % Config.save_optimizer_every == 0:
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            
            temp_path = SAVE_MODEL_FILE + ".tmp"
            torch.save(save_dict, temp_path)
            os.replace(temp_path, SAVE_MODEL_FILE)
            print(f"📁 Local save complete")
            
            # Upload to HuggingFace Hub
            try:
                api.upload_file(
                    path_or_fileobj=SAVE_MODEL_FILE,
                    path_in_repo=SAVE_MODEL_FILE,
                    repo_id=REPO_ID,
                    commit_message=f"🧠 Step {optimizer_step} | Loss: {last_reported_loss:.4f} | PPL: {math.exp(min(last_reported_loss, 20)):.2f}"
                )
                upload_count += 1
                print(f"☁️ Upload #{upload_count} to HuggingFace Hub complete!")
            except Exception as e:
                print(f"⚠️ Upload failed (local file safe): {str(e)[:100]}")
            
            print(f"{'='*60}\n")
    
    batch_step += 1
    
    if optimizer_step >= Config.max_steps:
        break

# ============ FINAL SAVE, SAMPLE & UPLOAD ============
print(f"\n{'='*60}")
print(f"🎉 TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"   Final Step: {optimizer_step}")
print(f"   Best Val Loss: {best_loss:.4f}")
print(f"   Best PPL: {math.exp(min(best_loss, 20)):.2f}")
print(f"   Total Uploads: {upload_count}")

# Final samples
print(f"\n📝 Final Sample Generation:")
samples = generate_samples(max_tokens=60, temperature=0.8)
for prompt, generated in samples:
    print(f"   🔹 Prompt: {prompt}")
    print(f"   🔸 Generated: {generated}")
    print()

# Final save
raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
final_dict = {
    'step': optimizer_step,
    'model_state_dict': raw_model.state_dict(),
    'final_loss': last_reported_loss,
    'best_loss': best_loss,
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(final_dict, SAVE_MODEL_FILE)
print("📁 Final local save complete")

# Final upload
try:
    api.upload_file(
        path_or_fileobj=SAVE_MODEL_FILE,
        path_in_repo=SAVE_MODEL_FILE,
        repo_id=REPO_ID,
        commit_message=f"✅ Training Complete: Step {optimizer_step} | Final Loss: {last_reported_loss:.4f}"
    )
    print("☁️ Final model uploaded to HuggingFace Hub!")
except Exception as e:
    print(f"⚠️ Final upload failed: {e}")

print(f"\n📊 Training Summary:")
print(f"   Total Steps: {optimizer_step}")
print(f"   Hub Uploads: {upload_count} (every 200 steps)")
print(f"   Best Loss: {best_loss:.4f}")
print(f"   Best PPL: {math.exp(min(best_loss, 20)):.2f}")
print(f"   Dataset Mix: IndicCorp 40% | C4 20% | Sangraha 40%")
print("\n🎊 Jaddangi AI - Telugu 0.5B Model Ready!")
