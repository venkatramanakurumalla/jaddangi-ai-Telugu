"""
🧠 Jaddangi AI — Telugu 0.5B GPT — Long-Form Inference Script
   - HF token loaded from environment (never hardcoded).
   - Vectorized token blocking + repetition penalty (fast, no Python loops per step).
   - True sliding-window generation for arbitrarily long output (>> 1024 tokens).
   - Per-token progress + throughput reporting (no silent hangs).
   - Device/GPU check with clear warning if running on CPU.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import math, os, re, time
import sentencepiece as spm
from huggingface_hub import hf_hub_download, login
from dataclasses import dataclass

# ============================================================
# 1. Authentication & File Download
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN")  # set via Colab/Kaggle secrets, never hardcode
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️  HF_TOKEN not set in environment — set it before running if the repo is private.")

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
MODEL_FILE = "Telugu_Model_0.5B_V4.pt"
TOKENIZER_FILE = "telugu_spm.model"

for fname in [TOKENIZER_FILE, MODEL_FILE]:
    if not os.path.exists(fname):
        hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=".")
        print(f"✅ Downloaded {fname}")
    else:
        print(f"✅ {fname} found locally")

# ============================================================
# 2. Configuration
# ============================================================
class Config:
    vocab_size = 32000
    d_model = 1440
    n_layers = 21
    n_heads = 20
    n_kv_heads = 5
    max_seq_len = 1024
    expansion_factor = 4
    dropout = 0.0

@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 60
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    repetition_window: int = 80
    context_reserve: int = 128      # tokens of headroom kept free before re-sliding window
    print_every: int = 10           # print progress every N generated tokens
    min_new_tokens: int = 0         # EOS is suppressed until this many tokens are generated

# ============================================================
# 3. Tokenizer
# ============================================================
class SentencePieceTokenizer:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.pad_token_id = self.sp.pad_id()
        self.eos_token_id = self.sp.eos_id()
        self.bos_token_id = self.sp.bos_id()
        self.unk_token_id = self.sp.unk_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def id_to_piece(self, id):
        return self.sp.id_to_piece(id)

    def __len__(self):
        return self.vocab_size

tokenizer = SentencePieceTokenizer(TOKENIZER_FILE)

# ============================================================
# 4. Blocked-token mask (precomputed once, vectorized at use)
# ============================================================
def build_blocked_mask(tokenizer, device) -> torch.Tensor:
    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    english_junk = {'the','and','for','are','but','not','you','all','can','had',
                     'her','was','one','our','out','day','get','has','him','his',
                     'how','man','new','now','old','see','two','way','who','boy',
                     'did','its','let','put','say','she','too','use'}
    for token_id in range(tokenizer.vocab_size):
        piece = tokenizer.id_to_piece(token_id)
        clean = piece.replace('▁', '')
        if token_id == tokenizer.unk_token_id:
            mask[token_id] = True
        elif re.match(r'^[a-zA-Z]+$', clean):
            mask[token_id] = True
        elif any(x in piece for x in ['<', '>', '/', '=', 'http', 'www', '.com', '@', '#']):
            mask[token_id] = True
        elif len(re.findall(r'\d', piece)) >= 2:
            mask[token_id] = True
        elif clean.lower() in english_junk:
            mask[token_id] = True
    return mask.to(device)

# ============================================================
# 5. Model definition
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
        self.lm_head.weight = self.wte.weight

        head_dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(config.max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        x = self.wte(idx)
        cos = self.cos_cached[:, :, :T, :].to(x.device)
        sin = self.sin_cached[:, :, :T, :].to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm_f(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

# ============================================================
# 6. Load model
# ============================================================
import gc
print("\n🔧 Initializing model...")
model = TeluguGPT(Config())
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("🔍 Loading checkpoint...")
ckpt = torch.load(MODEL_FILE, map_location='cpu')
state_dict = ckpt['model_state_dict']
for buf in ['cos_cached', 'sin_cached', 'lm_head.weight']:
    state_dict.pop(buf, None)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
model.lm_head.weight = model.wte.weight
model = model.to(model.device)
model.eval()

BLOCKED_MASK = build_blocked_mask(tokenizer, model.device)

print(f"✅ Model loaded — {model.get_num_params()/1e6:.1f}M params on {model.device}")
print(f"🚫 Blocked {BLOCKED_MASK.sum().item()} tokens ({100*BLOCKED_MASK.float().mean().item():.1f}% of vocab)")
if model.device.type == "cpu":
    print("⚠️  Running on CPU — generation will be SLOW (no KV cache, ~0.5–3 tok/s for this model size).")
    print("   If on Colab/Kaggle: Runtime → Change runtime type → GPU, then rerun this cell.")

# ============================================================
# 7. Core single-window sampler
# ============================================================
@torch.no_grad()
def _sample_next_token(model, context_ids, blocked_mask, generated_tail, gen_cfg: GenerationConfig,
                        suppress_eos_id: int = None):
    """context_ids: 1D list, len <= max_seq_len. Returns next token id.
    If suppress_eos_id is given, that token is masked out entirely (used to
    enforce a minimum generation length before EOS is allowed)."""
    context_tensor = torch.tensor([context_ids], device=model.device)
    logits = model(context_tensor)[0, -1, :].clone()

    logits.masked_fill_(blocked_mask, float('-inf'))
    if suppress_eos_id is not None and 0 <= suppress_eos_id < logits.size(0):
        logits[suppress_eos_id] = float('-inf')

    if generated_tail:
        tail = torch.tensor(list(set(generated_tail)), device=model.device, dtype=torch.long)
        vals = logits[tail]
        penalized = torch.where(vals < 0, vals * gen_cfg.repetition_penalty, vals / gen_cfg.repetition_penalty)
        logits[tail] = penalized

    logits = logits / max(gen_cfg.temperature, 1e-6)

    if gen_cfg.top_k > 0:
        k = min(gen_cfg.top_k, logits.size(-1))
        topk_vals, topk_idx = torch.topk(logits, k)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[topk_idx] = False
        logits[mask] = float('-inf')

    if gen_cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cum_probs > gen_cfg.top_p
        sorted_remove[1:] = sorted_remove[:-1].clone()
        sorted_remove[0] = False
        remove_idx = sorted_remove.scatter(0, sorted_idx, sorted_remove)
        logits[remove_idx] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or probs.sum().item() < 0.99:
        return torch.argmax(logits).item()
    return torch.multinomial(probs, 1).item()

# ============================================================
# 8. Long-form generator with sliding window + live progress
# ============================================================
@torch.no_grad()
def generate_large_text(prompt: str, total_new_tokens: int = 300, gen_cfg: GenerationConfig = None,
                          print_progress: bool = True) -> str:
    """
    Generates long Telugu text by sliding the 1024-token context window
    forward as generation proceeds. Coherence over very long spans will
    drift (the model can't see beyond its window), but this streams
    progress continuously instead of hanging silently.
    """
    gen_cfg = gen_cfg or GenerationConfig()
    max_len = Config.max_seq_len
    window_budget = max_len - gen_cfg.context_reserve

    generated = tokenizer.encode(prompt)
    new_tokens = 0
    hit_eos = False
    eos_id = tokenizer.eos_token_id
    start = time.time()

    while new_tokens < total_new_tokens and not hit_eos:
        context_ids = generated[-window_budget:]

        while new_tokens < total_new_tokens and len(context_ids) < max_len:
            tail = generated[-gen_cfg.repetition_window:]
            suppress_eos = eos_id if new_tokens < gen_cfg.min_new_tokens else None
            next_id = _sample_next_token(model, context_ids, BLOCKED_MASK, tail, gen_cfg,
                                          suppress_eos_id=suppress_eos)

            if next_id == eos_id:
                hit_eos = True
                break

            generated.append(next_id)
            context_ids.append(next_id)
            new_tokens += 1

            if print_progress and new_tokens % gen_cfg.print_every == 0:
                elapsed = time.time() - start
                rate = new_tokens / elapsed if elapsed > 0 else 0
                eta = (total_new_tokens - new_tokens) / rate if rate > 0 else float('inf')
                print(f"  {new_tokens}/{total_new_tokens} tokens | {rate:.2f} tok/s | "
                      f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s")

        if hit_eos:
            break

    elapsed = time.time() - start
    if print_progress:
        if hit_eos:
            print(f"⏹️  Model emitted EOS after {new_tokens}/{total_new_tokens} tokens "
                  f"({elapsed:.0f}s) — stopped early, not an error.")
        else:
            print(f"✅ Done: {new_tokens} tokens in {elapsed:.0f}s")

    return tokenizer.decode(generated)

# ============================================================
# 9. Convenience wrapper
# ============================================================
def generate_epic_story(prompt: str, total_tokens: int = 300, temperature: float = 0.8,
                          top_k: int = 60, top_p: float = 0.9, repetition_penalty: float = 1.5,
                          print_every: int = 10, min_new_tokens: int = None) -> str:
    """
    min_new_tokens: EOS is suppressed until this many tokens are generated.
    Defaults to total_tokens (i.e. EOS fully disabled) if not set, since the
    model has been observed to emit EOS very early. Pass 0 to allow natural
    early stopping.
    """
    if min_new_tokens is None:
        min_new_tokens = total_tokens
    cfg = GenerationConfig(temperature=temperature, top_k=top_k, top_p=top_p,
                            repetition_penalty=repetition_penalty, print_every=print_every,
                            min_new_tokens=min_new_tokens)
    return generate_large_text(prompt, total_new_tokens=total_tokens, gen_cfg=cfg)

print("\n🎉 Ready. Recommended: test small first, then scale up.")
print("   story = generate_epic_story('ఒక చిన్న పల్లెటూళ్ళో...', total_tokens=20)  # quick sanity check")
print("   print(story)")
print("   # once confirmed working:")
print("   story = generate_epic_story('ఒక చిన్న పల్లెటూళ్ళో...', total_tokens=1000)")
