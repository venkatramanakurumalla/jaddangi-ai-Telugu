
# ==============================================================================
# 🧬 JADDANGI AI: 1B -> 3B BALANCED NET2NET EXPANSION (WIDTH + DEPTH)
# MEMORY-SAFE VERSION — fixes Colab RAM crash
# Source: Telugu_Model_1B_V5.pt  (d_model=1440, 42 layers, ~1.0B params)
# Target: ~3.0B params            (d_model=1728, 68 layers)
# ==============================================================================
!pip install torch huggingface_hub -q

import torch
import os
import gc
from huggingface_hub import HfApi, login, hf_hub_download

print("🧬 Initiating Memory-Safe Net2Net Expansion (1B -> 3B)...")

# ---------------------------------------------------------
# 1. AUTH & DOWNLOAD
# ---------------------------------------------------------
HF_TOKEN =""
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment. Set it as a Colab/Kaggle Secret.")
login(token=HF_TOKEN)
api = HfApi()

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
OLD_FILE = "Telugu_Model_1B_V5.pt"
NEW_FILE = "Telugu_Model_3B_V1.pt"

if not os.path.exists(OLD_FILE):
    print(f"📥 Downloading {OLD_FILE} from Hugging Face...")
    hf_hub_download(repo_id=REPO_ID, filename=OLD_FILE, local_dir=".")

# ---------------------------------------------------------
# 2. ARCHITECTURE DEFINITIONS
# ---------------------------------------------------------
class OldConfig:
    vocab_size = 32000
    d_model = 1440
    n_layers = 42
    n_heads = 20
    n_kv_heads = 5
    expansion_factor = 4

class NewConfig:
    vocab_size = 32000
    d_model = 1728          # 1440 x 1.2  (head_dim = 72, unchanged)
    n_layers = 68            # 42 x 1.62
    n_heads = 24              # 1728 / 72
    n_kv_heads = 6            # 4:1 ratio maintained
    expansion_factor = 4
    max_seq_len = 384

old_head_dim = OldConfig.d_model // OldConfig.n_heads
new_head_dim = NewConfig.d_model // NewConfig.n_heads
assert old_head_dim == new_head_dim, "head_dim must stay constant for safe expansion"

old_hd = int(2/3 * OldConfig.expansion_factor * OldConfig.d_model)
new_hd = int(2/3 * NewConfig.expansion_factor * NewConfig.d_model)

NOISE_SCALE = 1e-4  # small enough to not disturb old function, large enough to break symmetry

# ---------------------------------------------------------
# 3. HELPERS — build directly in FP16, pop source tensors immediately
# ---------------------------------------------------------
def expand_linear_fp16(old_w, new_out, new_in, noise_scale=NOISE_SCALE):
    """Builds the expanded tensor directly in FP16 to halve peak RAM vs FP32
    intermediates. Old weights copied in; new rows/cols get small noise."""
    old_out, old_in = old_w.shape
    new_w = torch.zeros((new_out, new_in), dtype=torch.float16)
    new_w[:old_out, :old_in] = old_w.half()
    if new_out > old_out:
        new_w[old_out:, :] = (torch.randn(new_out - old_out, new_in) * noise_scale).half()
    if new_in > old_in:
        new_w[:, old_in:] = (torch.randn(new_out, new_in - old_in) * noise_scale).half()
    return new_w

def expand_norm_fp16(old_norm, new_dim, noise_scale=NOISE_SCALE):
    new_norm = torch.ones(new_dim, dtype=torch.float16)
    new_norm[:old_norm.shape[0]] = old_norm.half()
    if new_dim > old_norm.shape[0]:
        new_norm[old_norm.shape[0]:] += (torch.randn(new_dim - old_norm.shape[0]) * noise_scale).half()
    return new_norm

def pop_expand_linear(old_state, key, new_out, new_in):
    """Pops the source tensor out of old_state (freeing it from the dict) right
    after use, and deletes the local reference. This is the key RAM fix —
    old_state shrinks continuously instead of staying at full size until the end."""
    old_w = old_state.pop(key)
    result = expand_linear_fp16(old_w, new_out, new_in)
    del old_w
    return result

def pop_expand_norm(old_state, key, new_dim):
    old_n = old_state.pop(key)
    result = expand_norm_fp16(old_n, new_dim)
    del old_n
    return result

# ---------------------------------------------------------
# 4. LOAD CHECKPOINT
# ---------------------------------------------------------
print("📥 Loading 1B checkpoint...")
old_ckpt = torch.load(OLD_FILE, map_location="cpu")
old_state = old_ckpt.get("model_state_dict", old_ckpt)
old_loss = old_ckpt.get("loss", float("inf"))
del old_ckpt  # drop the wrapper dict, keep only the tensors we need
gc.collect()

new_state = {}

# ---------------------------------------------------------
# 5. EMBEDDINGS + FINAL NORM
# ---------------------------------------------------------
print("🔬 Expanding Embeddings and Final Norm...")
new_state["wte.weight"] = pop_expand_linear(old_state, "wte.weight", NewConfig.vocab_size, NewConfig.d_model)
new_state["lm_head.weight"] = pop_expand_linear(old_state, "lm_head.weight", NewConfig.vocab_size, NewConfig.d_model)
new_state["norm_f.weight"] = pop_expand_norm(old_state, "norm_f.weight", NewConfig.d_model)
gc.collect()

# ---------------------------------------------------------
# 6. TRANSFORMER LAYERS
#    Layers 0-41  : WIDTH-expanded copies of the old 42 layers (trained knowledge)
#    Layers 42-67 : NEW depth, TRUE zero-identity blocks (fresh, to be trained)
# ---------------------------------------------------------
prefix = "blocks."
print(f"🔬 Width-expanding {OldConfig.n_layers} existing layers "
      f"(d_model {OldConfig.d_model} -> {NewConfig.d_model})...")

for i in range(OldConfig.n_layers):
    p = f"{prefix}{i}."

    new_state[p + "norm1.weight"] = pop_expand_norm(old_state, p + "norm1.weight", NewConfig.d_model)
    new_state[p + "norm2.weight"] = pop_expand_norm(old_state, p + "norm2.weight", NewConfig.d_model)

    new_state[p + "attn.q_proj.weight"] = pop_expand_linear(
        old_state, p + "attn.q_proj.weight", NewConfig.n_heads * new_head_dim, NewConfig.d_model
    )
    new_state[p + "attn.k_proj.weight"] = pop_expand_linear(
        old_state, p + "attn.k_proj.weight", NewConfig.n_kv_heads * new_head_dim, NewConfig.d_model
    )
    new_state[p + "attn.v_proj.weight"] = pop_expand_linear(
        old_state, p + "attn.v_proj.weight", NewConfig.n_kv_heads * new_head_dim, NewConfig.d_model
    )
    new_state[p + "attn.o_proj.weight"] = pop_expand_linear(
        old_state, p + "attn.o_proj.weight", NewConfig.d_model, NewConfig.n_heads * new_head_dim
    )

    new_state[p + "mlp.w1.weight"] = pop_expand_linear(old_state, p + "mlp.w1.weight", new_hd, NewConfig.d_model)
    new_state[p + "mlp.w2.weight"] = pop_expand_linear(old_state, p + "mlp.w2.weight", new_hd, NewConfig.d_model)
    new_state[p + "mlp.w3.weight"] = pop_expand_linear(old_state, p + "mlp.w3.weight", NewConfig.d_model, new_hd)

    # Free per-layer garbage every few layers to keep RAM flat, not just at the end
    if i % 5 == 0:
        gc.collect()
    print(f"  ✅ Layer {i+1}/{OldConfig.n_layers} width-expanded | old_state keys remaining: {len(old_state)}")

# old_state should be empty (or near-empty) now — free it fully
del old_state
gc.collect()
print("🧹 old_state fully cleared from RAM.")

print(f"🔬 Appending {NewConfig.n_layers - OldConfig.n_layers} new depth layers "
      f"(true zero-identity, FP16, at new width {NewConfig.d_model})...")

for i in range(OldConfig.n_layers, NewConfig.n_layers):
    p = f"{prefix}{i}."

    new_state[p + "norm1.weight"] = torch.ones(NewConfig.d_model, dtype=torch.float16)
    new_state[p + "norm2.weight"] = torch.ones(NewConfig.d_model, dtype=torch.float16)

    new_state[p + "attn.q_proj.weight"] = torch.zeros(NewConfig.n_heads * new_head_dim, NewConfig.d_model, dtype=torch.float16)
    new_state[p + "attn.k_proj.weight"] = torch.zeros(NewConfig.n_kv_heads * new_head_dim, NewConfig.d_model, dtype=torch.float16)
    new_state[p + "attn.v_proj.weight"] = torch.zeros(NewConfig.n_kv_heads * new_head_dim, NewConfig.d_model, dtype=torch.float16)
    new_state[p + "attn.o_proj.weight"] = torch.zeros(NewConfig.d_model, NewConfig.n_heads * new_head_dim, dtype=torch.float16)

    new_state[p + "mlp.w1.weight"] = torch.zeros(new_hd, NewConfig.d_model, dtype=torch.float16)
    new_state[p + "mlp.w2.weight"] = torch.zeros(new_hd, NewConfig.d_model, dtype=torch.float16)
    new_state[p + "mlp.w3.weight"] = torch.zeros(NewConfig.d_model, new_hd, dtype=torch.float16)

    if i % 10 == 0:
        gc.collect()

print(f"✅ Expansion complete. Total layers: {NewConfig.n_layers}, d_model: {NewConfig.d_model}")

# ---------------------------------------------------------
# 7. NaN / Inf SAFETY GUARD
# ---------------------------------------------------------
print("🔍 Running NaN/Inf guard on expanded state...")
bad_keys = [k for k, v in new_state.items() if torch.isnan(v).any() or torch.isinf(v).any()]
if bad_keys:
    raise ValueError(f"NaN/Inf detected in: {bad_keys[:5]} ... aborting save")
print("✅ All tensors clean.")

# ---------------------------------------------------------
# 8. SAVE (already FP16) & UPLOAD
# ---------------------------------------------------------
print(f"💾 Saving to {NEW_FILE} (already FP16, no extra conversion pass needed)...")
torch.save({
    "step": 0,
    "model_state_dict": new_state,
    "loss": old_loss,
    "config": {
        "vocab_size": NewConfig.vocab_size,
        "d_model": NewConfig.d_model,
        "n_layers": NewConfig.n_layers,
        "n_heads": NewConfig.n_heads,
        "n_kv_heads": NewConfig.n_kv_heads,
        "max_seq_len": NewConfig.max_seq_len,
        "expansion_factor": NewConfig.expansion_factor,
    },
    "expansion_metadata": {
        "source": OLD_FILE,
        "type": "Net2Net-Balanced (Width+Depth), Memory-Safe",
        "old_d_model": OldConfig.d_model,
        "new_d_model": NewConfig.d_model,
        "old_layers": OldConfig.n_layers,
        "new_layers": NewConfig.n_layers,
        "noise_scale": NOISE_SCALE,
    }
}, NEW_FILE)
print("✅ 3B checkpoint saved locally (FP16).")

del new_state
gc.collect()

print(f"☁️ Uploading {NEW_FILE} to Hugging Face...")
api.upload_file(
    path_or_fileobj=NEW_FILE,
    path_in_repo=NEW_FILE,
    repo_id=REPO_ID,
    commit_message="🚀 1B -> 3B Balanced Net2Net Expansion (Width 1440->1728, Depth 42->68), Memory-Safe"
)
print("🎉 Upload Complete! The 3B Telugu model is now on Hugging Face.")
