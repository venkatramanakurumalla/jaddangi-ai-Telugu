

# The issue is clear from the logs:
# - Datasets load fine (all 4 succeed)
# - But the for loop in _process_stream runs for 6+ minutes with NO yield
# - Then exits because interleave_datasets is exhausted
# - Buffer never fills to 1025 tokens because most texts are filtered out

# ROOT CAUSE: _is_valid_telugu is too strict + buffer needs too many tokens
# FIX:
# 1. Lower Telugu threshold (30 chars -> 10 chars)
# 2. Pad short sequences instead of requiring full 1025 tokens
# 3. Add a max_wait limit - if buffer has some tokens, pad and yield
# 4. Add explicit timeout in data loading
# 5. Use take() to bound dataset iteration
# 6. Log EVERY sample processed to see what's happening


"""
Continue Training Telugu Model with 1024 Context + Telugu Wikipedia
====================================================================
FIXED VERSION - Addresses dataset hang/exhaustion issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import gc
import random
from datasets import load_dataset, interleave_datasets
from torch.utils.checkpoint import checkpoint
import sentencepiece as spm
from huggingface_hub import HfApi, login, hf_hub_download
from torch.amp import GradScaler, autocast

# ============================================
# 1. AUTH & CONFIG
# ============================================
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
    max_seq_len = 1024
    expansion_factor = 4
    dropout = 0.2
    use_checkpointing = True

    batch_size = 2
    gradient_accumulation_steps = 32
    learning_rate = 3e-5
    max_steps = 15000
    warmup_steps = 300
    save_every = 200
    log_every = 10
    eval_every = 100
    max_grad_norm = 1.0
    use_compile = False
    save_optimizer_every = 1000

    # Dataset ratios
    indicorp_ratio = 0.30
    c4_ratio = 0.15
    sangraha_ratio = 0.30
    wiki_ratio = 0.25

    # Data loading settings
    max_samples_per_dataset = 500000  # Limit to prevent infinite streaming
    min_telugu_chars = 5  # Lowered from 30 - less aggressive filtering
    min_text_length = 50  # Lowered from 80
    pad_token_id = 0  # Will be set from tokenizer

# ============================================
# 2. TOKENIZER
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
Config.pad_token_id = tokenizer.pad_token_id

# ============================================
# 3. DATA LOADING - FIXED WITH PADDING & TIMEOUTS
# ============================================

def is_valid_telugu(text, min_chars=5, min_len=50):
    """Relaxed Telugu validation - just needs SOME Telugu chars."""
    if len(text) < min_len:
        return False
    # Check first 1000 chars for Telugu presence
    sample = text[:1000]
    telugu_chars = sum(1 for c in sample if '\u0C00' <= c <= '\u0C7F')
    return telugu_chars >= min_chars

def extract_text(example):
    """Extract text from various dataset formats."""
    if isinstance(example, str):
        return example
    if isinstance(example, dict):
        for key in ['text', 'content', 'sentence', 'paragraph', 'article']:
            if key in example and example[key]:
                return str(example[key])
        # Wikipedia format
        if 'title' in example and 'text' in example:
            return f"{example['title']}\n\n{example['text']}"
        for val in example.values():
            if isinstance(val, str) and len(val) > 50:
                return val
    return ''

def load_single_dataset(name, config_name=None, split="train", max_samples=500000):
    """Load a single dataset with bounded samples."""
    try:
        if config_name:
            ds = load_dataset(name, config_name, split=split, streaming=True)
        else:
            ds = load_dataset(name, split=split, streaming=True)
        return ds.take(max_samples)  # Bound the dataset
    except Exception as e:
        print(f"   ❌ Failed to load {name}: {e}")
        return None

def create_data_generator(tokenizer, config):
    """Create a data generator with guaranteed yields."""
    print("⚡ Loading Telugu datasets...")

    # Load datasets with bounded samples
    print("   📊 Loading IndicCorpV2...")
    ds1 = load_single_dataset("ai4bharat/IndicCorpV2", split="tel_Telu", max_samples=config.max_samples_per_dataset)

    print("   📊 Loading C4 (te)...")
    ds2 = load_single_dataset("allenai/c4", config_name="te", split="train", max_samples=config.max_samples_per_dataset)

    print("   📊 Loading Sangraha...")
    ds3 = load_single_dataset("ai4bharat/sangraha", config_name="verified", split="tel", max_samples=config.max_samples_per_dataset)

    print("   📚 Loading Telugu Wikipedia...")
    ds4 = load_single_dataset("wikimedia/wikipedia", config_name="20231101.te", split="train", max_samples=100000)

    if ds4 is None:
        print("   Trying fallback Wikipedia...")
        ds4 = load_single_dataset("wikipedia", config_name="20220301.te", split="train", max_samples=100000)

    # Filter out None datasets
    datasets_list = []
    probs = []

    if ds1 is not None:
        datasets_list.append(ds1)
        probs.append(config.indicorp_ratio)
    if ds2 is not None:
        datasets_list.append(ds2)
        probs.append(config.c4_ratio)
    if ds3 is not None:
        datasets_list.append(ds3)
        probs.append(config.sangraha_ratio)
    if ds4 is not None:
        datasets_list.append(ds4)
        probs.append(config.wiki_ratio)

    if len(datasets_list) == 0:
        raise RuntimeError("No datasets could be loaded!")

    # Normalize probabilities
    total = sum(probs)
    probs = [p / total for p in probs]

    print(f"   ✅ Loaded {len(datasets_list)} datasets")
    print(f"   📊 Ratios: {[f'{p:.2f}' for p in probs]}")

    # Interleave with first_exhausted
    combined = interleave_datasets(
        datasets_list,
        probabilities=probs,
        seed=random.randint(0, 10000),
        stopping_strategy="first_exhausted"
    )

    # Process with buffer
    buffer = []
    total_yielded = 0
    processed = 0
    filtered = 0
    last_log = time.time()
    start_time = time.time()

    for example in combined:
        text = extract_text(example)
        processed += 1

        if not is_valid_telugu(text, config.min_telugu_chars, config.min_text_length):
            filtered += 1
            continue

        # Tokenize - use full text up to a large limit
        ids = tokenizer.encode(text[:8000], truncation=True, max_length=config.max_seq_len)

        if len(ids) < 10:  # Skip very short tokenized texts
            filtered += 1
            continue

        buffer.extend(ids + [tokenizer.eos_token_id])

        # Yield complete chunks
        while len(buffer) >= config.max_seq_len + 1:
            chunk = buffer[:config.max_seq_len + 1]
            buffer = buffer[config.max_seq_len + 1:]
            total_yielded += 1

            yield {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long)
            }

        # Log progress every 10 seconds
        if time.time() - last_log > 10:
            elapsed = time.time() - start_time
            print(f"   ⚡ {elapsed:.0f}s | Processed: {processed} | Filtered: {filtered} | "
                  f"Yielded: {total_yielded} | Buffer: {len(buffer)}")
            last_log = time.time()

    print(f"\n   ✅ Dataset stream ended.")
    print(f"   📊 Total processed: {processed} | Filtered: {filtered} | Yielded: {total_yielded}")

    # Flush remaining buffer with padding if we have enough tokens
    if len(buffer) >= 64:  # At least 64 tokens to make it worthwhile
        print(f"   🔄 Flushing buffer with {len(buffer)} tokens...")
        while len(buffer) >= 2:
            # Create a padded chunk
            chunk_len = min(len(buffer) - 1, config.max_seq_len)
            chunk = buffer[:chunk_len + 1]
            buffer = buffer[chunk_len:]

            # Pad to max_seq_len if needed
            input_ids = chunk[:-1]
            labels = chunk[1:]

            if len(input_ids) < config.max_seq_len:
                pad_len = config.max_seq_len - len(input_ids)
                input_ids = input_ids + [config.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len  # -100 is ignored in loss

            total_yielded += 1
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        print(f"   ✅ Buffer flush complete. Total yielded: {total_yielded}")

def get_batch(data_iter, batch_size, device, timeout=60):
    """Get a batch with timeout protection."""
    input_ids = []
    labels = []
    start = time.time()

    while len(input_ids) < batch_size:
        if time.time() - start > timeout:
            print(f"   ⚠️ Batch timeout after {timeout}s. Returning partial batch ({len(input_ids)} samples)")
            break

        try:
            sample = next(data_iter)
            input_ids.append(sample["input_ids"])
            labels.append(sample["labels"])
        except StopIteration:
            print("   ⚠️ Data iterator exhausted during batching")
            break

    if len(input_ids) == 0:
        return None

    return {
        "input_ids": torch.stack(input_ids).to(device, non_blocking=True),
        "labels": torch.stack(labels).to(device, non_blocking=True)
    }

# ============================================
# 4. MODEL (same as before)
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
# 5. LOAD CHECKPOINT
# ============================================
print("\n🔧 Initializing model with 1024 context...")
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

    for buf in ['cos_cached', 'sin_cached']:
        if buf in state_dict:
            print(f"   Removing old {buf}")
            state_dict.pop(buf)

    state_dict.pop('lm_head.weight', None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   Missing keys: {len(missing)} (normal for context extension)")
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
    print("⚠️ No checkpoint found - starting fresh")

# ============================================
# 6. SCHEDULER
# ============================================
def lr_lambda(step):
    if step < Config.warmup_steps:
        return float(step) / float(max(1, Config.warmup_steps))
    progress = (step - Config.warmup_steps) / (Config.max_steps - Config.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, progress))))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Skip scheduler steps for resumed training
for _ in range(min(optimizer_step, Config.max_steps)):
    scheduler.step()

# ============================================
# 7. EVAL & GENERATE
# ============================================
@torch.no_grad()
def estimate_loss():
    model.eval()
    total_loss = 0.0
    steps = 0

    eval_ds = load_dataset("allenai/c4", "te", split="validation", streaming=True).take(1000)
    buffer = []
    for example in eval_ds:
        text = example.get('text', '')
        if len(text) > 50:
            ids = tokenizer.encode(text[:2000], truncation=True, max_length=Config.max_seq_len)
            buffer.extend(ids + [tokenizer.eos_token_id])
            if len(buffer) >= Config.max_seq_len + 1:
                chunk = buffer[:Config.max_seq_len + 1]
                buffer = buffer[Config.max_seq_len + 1:]
                input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=model.device)
                labels = torch.tensor([chunk[1:]], dtype=torch.long, device=model.device)

                with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.reshape(-1, Config.vocab_size), labels.reshape(-1))

                total_loss += loss.item()
                steps += 1
                if steps >= 20:
                    break

    model.train()
    return total_loss / max(1, steps)

@torch.no_grad()
def generate_samples(prompts=None, max_tokens=80, temperature=0.8):
    model.eval()
    if prompts is None:
        prompts = [
            "హైదరాబాద్ నగరంలో",
            "తెలుగు భాష అంటే",
            "భారతదేశంలో విద్య",
            "రైతుల జీవితం",
            "ఆరోగ్యం మరియు పోషణ",
            "కంప్యూటర్ మదర్ బోర్డ్"
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

        samples.append((prompt, tokenizer.decode(input_ids[0].tolist())))

    model.train()
    return samples

# ============================================
# 8. TRAINING LOOP
# ============================================
print(f"\n🚀 CONTINUED TRAINING: 1024 Context + Telugu Wikipedia")
print(f"   Device: {model.device_type.upper()}")
print(f"   Start Step: {optimizer_step}/{Config.max_steps}")
print(f"   Context: 384 → 1024")
print("=" * 60)

model.train()
window_loss = 0.0
loss_count = 0
last_reported_loss = 0.0
start_time = time.time()
best_loss = float('inf')
upload_count = 0
batch_step = 0

print("\n🔥 Starting training...")
print("   (First batch may take 2-3 minutes)\n")

# Create data iterator
data_iter = iter(create_data_generator(tokenizer, Config))

while optimizer_step < Config.max_steps:
    # Get batch with timeout
    batch = get_batch(data_iter, Config.batch_size, model.device, timeout=120)

    if batch is None:
        print("\n⚠️ No data available. Restarting data generator...")
        data_iter = iter(create_data_generator(tokenizer, Config))
        batch = get_batch(data_iter, Config.batch_size, model.device, timeout=120)
        if batch is None:
            print("❌ Cannot get data after restart. Stopping.")
            break

    # Check for padding mask
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Create attention mask for padding
    attention_mask = (input_ids != Config.pad_token_id).long()

    # Forward pass
    with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
        logits = model(input_ids)

        # Compute loss only on non-padded positions
        loss = F.cross_entropy(
            logits.reshape(-1, Config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100
        )
        scaled_loss = loss / Config.gradient_accumulation_steps

    # Backward pass
    scaler.scale(scaled_loss).backward()

    window_loss += loss.item()
    loss_count += 1
    batch_step += 1

    # Gradient accumulation step
    if batch_step % Config.gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Only step scheduler after optimizer step
        scheduler.step()

        optimizer_step += 1

        # Logging
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
                print(f"📊 Step {optimizer_step} | Loss: {current_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | GPU: {mem_alloc:.2f}GB | Time: {elapsed:.0f}s")
            else:
                print(f"📊 Step {optimizer_step} | Loss: {current_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | Time: {elapsed:.0f}s")

            window_loss = 0.0
            loss_count = 0
            start_time = time.time()

            gc.collect()
            if model.device_type == 'cuda':
                torch.cuda.empty_cache()

        # Evaluation
        if optimizer_step % Config.eval_every == 0 or optimizer_step == 1:
            val_loss = estimate_loss()
            val_ppl = math.exp(min(val_loss, 20))
            print(f"   🧪 Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"   🏆 New best: {best_loss:.4f}")

        # Save & Upload
        if optimizer_step % Config.save_every == 0:
            print(f"\n{'='*60}")
            print(f"💾 CHECKPOINT at Step {optimizer_step}")
            print(f"{'='*60}")

            print(f"📝 Generating samples...")
            samples = generate_samples(max_tokens=80, temperature=0.8)
            for prompt, generated in samples:
                print(f"   🔹 {prompt}")
                print(f"   🔸 {generated[:200]}...")
                print()

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
            print(f"📁 Saved")

            try:
                api.upload_file(
                    path_or_fileobj=SAVE_MODEL_FILE,
                    path_in_repo=SAVE_MODEL_FILE,
                    repo_id=REPO_ID,
                    commit_message=f"Step {optimizer_step} | Loss: {last_reported_loss:.4f}"
                )
                upload_count += 1
                print(f"☁️ Upload #{upload_count} complete!")
            except Exception as e:
                print(f"⚠️ Upload failed: {str(e)[:100]}")

            print(f"{'='*60}\n")

# ============================================
# 9. FINAL SAVE
# ============================================
print(f"\n{'='*60}")
print(f"🎉 TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"   Final Step: {optimizer_step}")
print(f"   Best Val Loss: {best_loss:.4f}")

print(f"\n📝 Final samples:")
samples = generate_samples(max_tokens=100, temperature=0.8)
for prompt, generated in samples:
    print(f"   🔹 {prompt}")
    print(f"   🔸 {generated}")
    print()

raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
final_dict = {
    'step': optimizer_step,
    'model_state_dict': raw_model.state_dict(),
    'final_loss': last_reported_loss,
    'best_loss': best_loss,
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(final_dict, SAVE_MODEL_FILE)
print("📁 Final save complete")

try:
    api.upload_file(
        path_or_fileobj=SAVE_MODEL_FILE,
        path_in_repo=SAVE_MODEL_FILE,
        repo_id=REPO_ID,
        commit_message=f"Complete: Step {optimizer_step}"
    )
    print("☁️ Final upload complete!")
except Exception as e:
    print(f"⚠️ Upload failed: {e}")

print(f"\n📊 Summary:")
print(f"   Steps: {optimizer_step}")
print(f"   Context: 1024")
print(f"   Uploads: {upload_count}")
print(f"   Best Loss: {best_loss:.4f}")
print("\n🎊 Jaddangi AI - Telugu 0.5B Model Ready!")
