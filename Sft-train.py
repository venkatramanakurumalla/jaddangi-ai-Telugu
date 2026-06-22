# ==============================================================================
# 🚀 PHASE-4 v10.6 ENTERPRISE - FINAL PRODUCTION SFT ENGINE (UNK PURGE ENABLED)
# ==============================================================================
!pip install datasets torch sentencepiece huggingface_hub -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import gc
import json
import hashlib
import random
import csv
import re
import numpy as np
import threading
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
import sentencepiece as spm
from huggingface_hub import HfApi, login, hf_hub_download
from torch.amp import GradScaler, autocast

# ============================================
# 1. AUTH, CONFIG & TELEMETRY
# ============================================
# 👇 Paste your HF Write Token inside the quotes 👇
HF_TOKEN = "YOUR_HUGGINGFACE_WRITE_TOKEN_HERE"

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_token = user_secrets.get_secret("HF_TOKEN")
    if secret_token: HF_TOKEN = secret_token
except Exception:
    pass

if HF_TOKEN and HF_TOKEN != "YOUR_HUGGINGFACE_WRITE_TOKEN_HERE":
    login(token=HF_TOKEN)
else:
    print("⚠️ CRITICAL: HF_TOKEN is missing! Please paste your token into the HF_TOKEN variable.")

api = HfApi()

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
BASE_MODEL_FILE = "Telugu_Model_0.5B_V4.pt"
SFT_MODEL_LATEST = "Telugu_Model_0.5B_V4_SFT_latest.pt"
SFT_MODEL_BEST = "Telugu_Model_0.5B_V4_SFT_best.pt"
TOKENIZER_FILE = "telugu_spm.model"
TOKENIZER_CONFIG = "tokenizer_config.json"
LOG_FILE = "training_telemetry.csv"

class SFTConfig:
    vocab_size = 32000
    d_model = 1440
    n_layers = 21
    n_heads = 20
    n_kv_heads = 5
    max_seq_len = 512             
    expansion_factor = 4

    use_checkpointing = True
    dropout = 0.10              
    weight_decay = 0.02         
    learning_rate = 2e-5
    max_steps = 4000            
    warmup_steps = 150          

    batch_size = 2
    gradient_accumulation_steps = 32
    shuffle_buffer = 5000        

    save_every = 200            
    save_opt_every = 1000       
    log_every = 10
    eval_every = 100
    eval_steps = 100            
    max_grad_norm = 1.0
    use_compile = True          
    label_smoothing = 0.05        
    pad_token_id = 0

    rope_cache_size = 4096
    eval_seed = 42
    adam_beta2 = 0.98
    early_stopping_patience = 5 
    
    max_input_len = 512       
    max_response_len = 256    
    drop_path_prob = 0.05     
    ema_update_every = 8
    
    neftune_alpha = 5.0       

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(["step", "train_loss", "raw_val_loss", "ema_val_loss", "lr", "tok_per_sec", "grad_norm"])

# ============================================
# 2. TOKENIZER EXTENSION
# ============================================
print("📥 Synchronizing core weights and tokenizers...")
for f in [TOKENIZER_FILE, BASE_MODEL_FILE]:
    if not os.path.exists(f):
        hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=".", local_dir_use_symlinks=False)

LOAD_CHECKPOINT = SFT_MODEL_LATEST if os.path.exists(SFT_MODEL_LATEST) else BASE_MODEL_FILE

class JaddangiTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.pad_token_id = self.sp.pad_id() if self.sp.pad_id() != -1 else self.sp.eos_id()
        self.eos_token_id = self.sp.eos_id()
        self.base_vocab_size = self.sp.get_piece_size()
        self.special_tokens = {"<user>": self.base_vocab_size, "<assistant>": self.base_vocab_size + 1}
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.extended_vocab_size = self.base_vocab_size + len(self.special_tokens)
        self.user_token_id = self.special_tokens["<user>"]
        self.asst_token_id = self.special_tokens["<assistant>"]

    def encode(self, text): return self.sp.encode_as_ids(text)
    def decode(self, ids):
        if torch.is_tensor(ids): ids = ids.tolist()
        text, buffer = "", []
        for i in ids:
            if i in self.inverse_special_tokens:
                if buffer: text += self.sp.decode_ids(buffer); buffer = []
                text += self.inverse_special_tokens[i]
            else: buffer.append(i)
        if buffer: text += self.sp.decode_ids(buffer)
        return text
    def save_config(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"base_vocab_size": self.base_vocab_size, "extended_vocab_size": self.extended_vocab_size, 
                       "special_tokens": self.special_tokens, "pad_token_id": self.pad_token_id, 
                       "eos_token_id": self.eos_token_id}, f, indent=2)

tokenizer = JaddangiTokenizer(TOKENIZER_FILE)
tokenizer.save_config(TOKENIZER_CONFIG)
SFTConfig.vocab_size = tokenizer.extended_vocab_size
SFTConfig.pad_token_id = tokenizer.pad_token_id

# ============================================
# 3. STATIC & STREAMING DATASETS
# ============================================
class IsolatedTeluguSFTDataset(IterableDataset):
    def __init__(self, tokenizer, config, split="train", val_ratio=0.04, shuffle_seed=42):
        self.tokenizer, self.config = tokenizer, config
        self.split, self.val_ratio = split, val_ratio
        self.shuffle_seed = shuffle_seed if split == "train" else 42 
        self.ignore_index = -100
        self._base_ds = load_dataset("ai4bharat/indic-align", "Indic_ShareLlama", split="train", streaming=True)

    def __iter__(self):
        shuffled_ds = self._base_ds.shuffle(buffer_size=self.config.shuffle_buffer, seed=self.shuffle_seed)
        global_input_pool, global_label_pool, global_seq_ids, current_seq_id = [], [], [], 1

        for example in shuffled_ds:
            conv_pairs = example.get('tel_Telu', [])
            if not conv_pairs or not isinstance(conv_pairs, list): continue
            
            normalized_pairs = []
            if isinstance(conv_pairs[0], list):
                for pair in conv_pairs:
                    if len(pair) >= 2: normalized_pairs.append([str(pair[0]), str(pair[1])])
            elif len(conv_pairs) >= 2 and isinstance(conv_pairs[0], str):
                normalized_pairs.append([str(conv_pairs[0]), str(conv_pairs[1])])
            if not normalized_pairs: continue

            serialized_conv = json.dumps(normalized_pairs, ensure_ascii=False).encode('utf-8')
            is_val_sample = (int(hashlib.md5(serialized_conv).hexdigest(), 16) % 100) < (self.val_ratio * 100)

            if self.split == "train" and is_val_sample: continue
            if self.split == "val" and not is_val_sample: continue

            for u_text, a_text in normalized_pairs:
                # 🛡️ 100% Native Telugu Enforcement: Drop samples containing Latin characters
                if bool(re.search(r'[a-zA-Z]', u_text)) or bool(re.search(r'[a-zA-Z]', a_text)):
                    continue

                u_ids = [self.tokenizer.user_token_id] + self.tokenizer.encode(u_text)[:self.config.max_input_len] + [self.tokenizer.asst_token_id]
                a_ids = self.tokenizer.encode(a_text)[:self.config.max_response_len] + [self.tokenizer.eos_token_id]
                
                # 🚨 THE NEW PURGE FILTER: If the tokenizer generates an <unk> token, throw the whole conversation away!
                if self.tokenizer.sp.unk_id() in u_ids or self.tokenizer.sp.unk_id() in a_ids:
                    continue
                
                turn_inputs = u_ids + a_ids
                global_input_pool.extend(turn_inputs)
                global_label_pool.extend([self.ignore_index] * len(u_ids) + a_ids)
                global_seq_ids.extend([current_seq_id] * len(turn_inputs))
                current_seq_id += 1

            while len(global_input_pool) >= self.config.max_seq_len:
                yield {
                    "input_ids": torch.tensor(global_input_pool[:self.config.max_seq_len], dtype=torch.long),
                    "labels": torch.tensor(global_label_pool[:self.config.max_seq_len], dtype=torch.long),
                    "seq_ids": torch.tensor(global_seq_ids[:self.config.max_seq_len], dtype=torch.long)
                }
                global_input_pool = global_input_pool[self.config.max_seq_len:]
                global_label_pool = global_label_pool[self.config.max_seq_len:]
                global_seq_ids = global_seq_ids[self.config.max_seq_len:]

            if len(global_input_pool) > self.config.max_seq_len * 4:
                global_input_pool, global_label_pool, global_seq_ids = [], [], []

dl_kwargs = {"num_workers": 0}

# ============================================
# 4. MODEL ARCHITECTURE 
# ============================================
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    return x.div(keep_prob) * random_tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x, cos, sin, attention_mask=None, is_causal=True, use_cache=False, past_key_value=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present_kv = (k, v) if use_cache else None

        k_rep, v_rep = k.repeat_interleave(self.n_rep, dim=1), v.repeat_interleave(self.n_rep, dim=1)
        
        # Native un-wrapped PyTorch call triggers Flash Attention dynamically
        y = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
            
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C)), present_kv

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = (int(2 / 3 * config.expansion_factor * config.d_model) // 256) * 256
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x): return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.d_model), RMSNorm(config.d_model)
        self.attn, self.mlp = GroupedQueryAttention(config), SwiGLU(config)
        self.use_checkpointing = config.use_checkpointing
        self.drop_path_prob = config.drop_path_prob * (layer_idx / max(1, config.n_layers - 1))

    def _forward_impl(self, x, cos, sin, attention_mask, is_causal, use_cache, past_key_value):
        attn_out, present_kv = self.attn(self.norm1(x), cos, sin, attention_mask, is_causal, use_cache, past_key_value)
        x = x + drop_path(attn_out, self.drop_path_prob, self.training)
        mlp_out = self.mlp(self.norm2(x))
        x = x + drop_path(mlp_out, self.drop_path_prob, self.training)
        return x, present_kv

    def forward(self, x, cos, sin, attention_mask=None, is_causal=True, use_cache=False, past_key_value=None):
        if self.training and x.requires_grad and self.use_checkpointing and not use_cache:
            return checkpoint(self._forward_impl, x, cos, sin, attention_mask, is_causal, use_cache, past_key_value, use_reentrant=False)
        return self._forward_impl(x, cos, sin, attention_mask, is_causal, use_cache, past_key_value)

class TeluguGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self.lm_head.weight = self.wte.weight

        head_dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(config.rope_cache_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, seq_ids=None, use_cache=False, past_key_values=None):
        B, T = idx.shape
        x = self.wte(idx)
        
        if self.training and self.config.neftune_alpha > 0:
            mag = self.config.neftune_alpha / math.sqrt(x.size(-1))
            noise = torch.empty_like(x).uniform_(-mag, mag)
            x = x + noise

        seq_offset = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cos, sin = self.cos_cached[:, :, seq_offset:seq_offset+T, :], self.sin_cached[:, :, seq_offset:seq_offset+T, :]

        if past_key_values is not None:
            attention_mask, is_causal = None, False
        elif seq_ids is not None:
            seq_ids = seq_ids.unsqueeze(0).expand(B, -1) if seq_ids.dim() == 1 else seq_ids
            valid_mask = seq_ids != 0
            block_mask = (seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)) & valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            combined_mask = block_mask & causal_mask.unsqueeze(0)
            attention_mask = torch.zeros_like(combined_mask, dtype=x.dtype).masked_fill_(~combined_mask, float('-inf')).unsqueeze(1)
            is_causal = False
        else:
            attention_mask, is_causal = None, True

        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            x, present_kv = block(x, cos, sin, attention_mask, is_causal, use_cache, past_key_values[i] if past_key_values is not None else None)
            if use_cache: present_key_values.append(present_kv)

        return (self.lm_head(self.norm_f(x)), present_key_values) if use_cache else self.lm_head(self.norm_f(x))

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=100, temperature=0.7, top_k=50):
        self.eval()
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                logits, past_key_values = self(idx[:, -1:], use_cache=True, past_key_values=past_key_values)
            else:
                logits, past_key_values = self(idx, use_cache=True)
            
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_token), dim=1)
            if next_token.item() == self.config.pad_token_id or next_token.item() == tokenizer.eos_token_id:
                break
                
        self.train()
        return idx[0].tolist()

# ============================================
# 5. EMA 
# ============================================
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {name: param.data.clone().detach().float() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data.float(), 1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

# ============================================
# 6. INITIALIZATION & STATE RECOVERY
# ============================================
print("\n🔧 Initializing structural footprint...")
config = SFTConfig()
raw_model = TeluguGPT(config)  

decay_params, no_decay_params = [], []
for name, param in raw_model.named_parameters():
    if param.requires_grad:
        (no_decay_params if 'norm' in name or 'bias' in name or 'embedding' in name else decay_params).append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': config.weight_decay},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=config.learning_rate, betas=(0.9, config.adam_beta2), fused=('cuda' in raw_model.device_type))

def lr_lambda(step):
    if step < config.warmup_steps: return float(step) / float(max(1, config.warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)))))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler(device=raw_model.device_type, enabled=(raw_model.device_type == 'cuda'))
optimizer_step, best_val_loss, evals_without_improvement = 0, float('inf'), 0

print(f"📦 Interrogating checkpoint route: {LOAD_CHECKPOINT}...")
checkpoint_data = torch.load(LOAD_CHECKPOINT, map_location='cpu') if os.path.exists(LOAD_CHECKPOINT) else {}
state_dict = checkpoint_data.get('model_state_dict', checkpoint_data)
state_dict.pop('lm_head.weight', None)

vocab_expanded = False
wte_key = 'wte.weight'
if wte_key in state_dict and state_dict[wte_key].shape[0] < config.vocab_size:
    print("   ✨ Applying Statistical Vocabulary Expansion...")
    old_wte = state_dict[wte_key]
    expansion_size = config.vocab_size - old_wte.shape[0]
    new_wte = torch.cat([old_wte, torch.zeros(expansion_size, config.d_model, dtype=old_wte.dtype, device=old_wte.device)], dim=0)
    nn.init.normal_(new_wte[-expansion_size:], mean=old_wte.mean().item(), std=old_wte.std().item())
    state_dict[wte_key] = new_wte
    vocab_expanded = True

if state_dict:
    raw_model.load_state_dict(state_dict, strict=False)
raw_model.lm_head.weight = raw_model.wte.weight

ema = ModelEMA(raw_model, decay=0.9999)

if config.use_compile and hasattr(torch, 'compile'):
    print("   ⚡ Compiling execution graph for fused kernels...")
    model = torch.compile(raw_model)
else:
    model = raw_model

if 'rng_state' in checkpoint_data:
    try:
        torch.set_rng_state(checkpoint_data['rng_state']['torch'])
        if torch.cuda.is_available() and len(checkpoint_data['rng_state']['cuda']) > 0:
            torch.cuda.set_rng_state_all(checkpoint_data['rng_state']['cuda'])
        random.setstate(checkpoint_data['rng_state']['python'])
        np.random.set_state(checkpoint_data['rng_state']['numpy'])
    except Exception: pass

if LOAD_CHECKPOINT == SFT_MODEL_LATEST:
    if 'optimizer_state_dict' in checkpoint_data and not vocab_expanded:
        try:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor): state[k] = v.to(raw_model.device)
        except Exception: pass
    if 'scheduler_state_dict' in checkpoint_data:
        try: scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        except Exception: pass
    
    optimizer_step = checkpoint_data.get('step', 0)
    best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
    evals_without_improvement = checkpoint_data.get('evals_without_improvement', 0)

    if 'ema_shadow' in checkpoint_data:
        try:
            ema.shadow = {k: v.to(raw_model.device).float() for k, v in checkpoint_data['ema_shadow'].items()}
        except Exception: pass

del checkpoint_data, state_dict
gc.collect()

# ============================================
# 7. STATIC VALIDATION CACHE & INFERENCE
# ============================================
print("\n📊 Locking Deterministic Validation Set...")
val_dataset = IsolatedTeluguSFTDataset(tokenizer, config, split="val")
val_loader_temp = DataLoader(val_dataset, batch_size=config.batch_size, **dl_kwargs)

cached_val_batches = []
for batch in val_loader_temp:
    cached_val_batches.append({k: v for k, v in batch.items()})
    if len(cached_val_batches) >= config.eval_steps: break
print(f"   ✅ Locked {len(cached_val_batches)} val blocks in memory.")

EVAL_PROMPTS = [
    "హైదరాబాద్ నగరం గురించి ఐదు వాక్యాలు చెప్పండి.",
    "కృత్రిమ మేధస్సు భవిష్యత్తులో మన జీవితాలను ఎలా మారుస్తుంది?"
]

@torch.no_grad()
def estimate_validation_loss():
    def _compute_loss():
        total_val_loss, total_val_tokens = 0.0, 0
        for batch in cached_val_batches:
            inputs, labels, seq_ids = batch["input_ids"].to(model.device), batch["labels"].to(model.device), batch["seq_ids"].to(model.device)
            with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
                logits = model(inputs, seq_ids=seq_ids)
                shift_logits, shift_labels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.reshape(-1, config.vocab_size), shift_labels.reshape(-1), ignore_index=-100, reduction='none')
                valid_mask = (shift_labels != -100)
                weighted_loss = (loss * valid_mask.view(-1).float()).sum()

            total_val_loss += weighted_loss.item()
            total_val_tokens += valid_mask.sum().item()
        return total_val_loss / total_val_tokens if total_val_tokens > 0 else float('inf')

    model.eval()
    raw_val_loss = _compute_loss()
    
    ema.apply_shadow(raw_model)
    ema_val_loss = _compute_loss()
    
    print("\n   [Qualitative Generation Check]")
    for p in EVAL_PROMPTS:
        prompt_ids = [tokenizer.user_token_id] + tokenizer.encode(p) + [tokenizer.asst_token_id]
        out_ids = raw_model.generate(prompt_ids, max_new_tokens=60, temperature=0.7)
        print(f"   User: {p}")
        print(f"   Asst: {tokenizer.decode(out_ids[len(prompt_ids):])}")
        print("   " + "-"*40)
        
    ema.restore(raw_model)
    model.train()
    return raw_val_loss, ema_val_loss

# ============================================
# 8. ASYNCHRONOUS BACKGROUND UPLOADER
# ============================================
class AsyncUploader:
    def __init__(self, api, repo_id):
        self.api, self.repo_id = api, repo_id
    
    def background_upload(self, local_path, repo_path, commit_msg):
        def _upload():
            attempt = 0
            while True:
                try:
                    self.api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path, repo_id=self.repo_id, commit_message=commit_msg)
                    print(f"\n✅ [Background Sync] {repo_path} pushed.")
                    break
                except Exception as e:
                    attempt += 1
                    wait = min(300, 2 ** attempt)
                    print(f"\n⚠️ [Background Sync] Upload failed. Retrying in {wait}s...")
                    time.sleep(wait)
        threading.Thread(target=_upload, daemon=True).start()

uploader = AsyncUploader(api, REPO_ID)

# ============================================
# 9. TRAINING EXECUTION LOOP
# ============================================
print(f"\n🚀 PRODUCTION SFT ALIGNMENT ENGINE ACTIVE")
print("=" * 60)

train_dataset = IsolatedTeluguSFTDataset(tokenizer, config, split="train")
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, **dl_kwargs)

model.train()
batch_step, window_loss, loss_count, window_tokens = 0, 0.0, 0, 0
start_time = time.time()
optimizer.zero_grad(set_to_none=True)

while optimizer_step < config.max_steps:
    for batch in train_loader:
        if optimizer_step >= config.max_steps: break

        inputs, labels, seq_ids = batch["input_ids"].to(model.device), batch["labels"].to(model.device), batch["seq_ids"].to(model.device)

        with autocast(device_type=model.device_type, enabled=(model.device_type == 'cuda')):
            logits = model(inputs, seq_ids=seq_ids)
            shift_logits, shift_labels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.reshape(-1, config.vocab_size), shift_labels.reshape(-1), ignore_index=-100, label_smoothing=config.label_smoothing)
            scaled_loss = loss / config.gradient_accumulation_steps

        if not torch.isfinite(scaled_loss):
            optimizer.zero_grad(set_to_none=True)
            batch_step += 1
            continue

        scaler.scale(scaled_loss).backward()
        window_loss += loss.item()
        loss_count += 1
        window_tokens += (shift_labels != -100).sum().item()
        del logits, loss, shift_logits, shift_labels, scaled_loss

        if (batch_step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.max_grad_norm)

            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                
                if (optimizer_step + 1) % config.ema_update_every == 0:
                    ema.update(raw_model)
                    
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step % config.log_every == 0:
                    elapsed = time.time() - start_time
                    current_loss = window_loss / max(1, loss_count)
                    tok_per_sec = window_tokens / max(elapsed, 0.001)
                    current_lr = scheduler.get_last_lr()[0]
                    
                    print(f"📊 Step {optimizer_step} | Loss: {current_loss:.4f} | LR: {current_lr:.2e} | Tok/s: {tok_per_sec:.0f} | Time: {elapsed:.1f}s")
                    
                    with open(LOG_FILE, mode='a', newline='') as f:
                        csv.writer(f).writerow([optimizer_step, current_loss, "", "", current_lr, tok_per_sec, grad_norm.item()])
                    
                    window_loss, loss_count, window_tokens, start_time = 0.0, 0, 0, time.time()

                if optimizer_step % config.eval_every == 0:
                    raw_val_loss, ema_val_loss = estimate_validation_loss()
                    is_new_best = ema_val_loss < best_val_loss
                    
                    print(f"\n🧪 [EVAL] Step {optimizer_step} | Raw Val: {raw_val_loss:.4f} | EMA Val: {ema_val_loss:.4f} | Best: {best_val_loss:.4f}")
                    
                    if is_new_best:
                        best_val_loss = ema_val_loss
                        evals_without_improvement = 0
                        print(f"🏆 NEW BEST VAL - Isolating weights...")
                        
                        best_dict = {
                            'step': optimizer_step,
                            'model_state_dict': {k: v.detach().cpu().half() for k, v in raw_model.state_dict().items()},
                            'best_val_loss': best_val_loss
                        }
                        torch.save(best_dict, SFT_MODEL_BEST + ".tmp")
                        os.replace(SFT_MODEL_BEST + ".tmp", SFT_MODEL_BEST)
                        uploader.background_upload(SFT_MODEL_BEST, SFT_MODEL_BEST, f"🏆 Best Val: {best_val_loss:.4f} @ Step {optimizer_step}")
                        del best_dict
                    else:
                        evals_without_improvement += 1

                    with open(LOG_FILE, mode='a', newline='') as f:
                        csv.writer(f).writerow([optimizer_step, "", raw_val_loss, ema_val_loss, "", "", ""])

                    if evals_without_improvement >= config.early_stopping_patience:
                        print(f"\n⏹️ EARLY STOPPING triggered. Pipeline converged.")
                        break

                if optimizer_step % config.save_every == 0:
                    print(f"\n💾 SERIALIZING LATEST WEIGHTS: Step {optimizer_step}")
                    
                    rng_state = {
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        'python': random.getstate(),
                        'numpy': np.random.get_state()
                    }

                    latest_dict = {
                        'step': optimizer_step,
                        'model_state_dict': {k: v.detach().cpu().half() for k, v in raw_model.state_dict().items()},
                        'best_val_loss': best_val_loss,
                        'scheduler_state_dict': scheduler.state_dict(),
                        'rng_state': rng_state,
                        'ema_shadow': {k: v.cpu().half() for k, v in ema.shadow.items()}
                    }
                    
                    if optimizer_step % config.save_opt_every == 0:
                        print("   -> Streaming optimizer payload to CPU...")
                        cpu_opt_state = {"state": {}, "param_groups": optimizer.param_groups}
                        for param_id, state in optimizer.state.items():
                            cpu_opt_state["state"][param_id] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state.items()}
                        latest_dict['optimizer_state_dict'] = cpu_opt_state

                    torch.save(latest_dict, SFT_MODEL_LATEST + ".tmp")
                    os.replace(SFT_MODEL_LATEST + ".tmp", SFT_MODEL_LATEST)
                    
                    if 'optimizer_state_dict' in latest_dict:
                        del cpu_opt_state
                    del latest_dict
                    gc.collect()

                    uploader.background_upload(SFT_MODEL_LATEST, SFT_MODEL_LATEST, f"🧠 Weights Step {optimizer_step}")

            else:
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        batch_step += 1

    if evals_without_improvement >= config.early_stopping_patience: break
