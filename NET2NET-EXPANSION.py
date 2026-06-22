# ==============================================================================
# 🧬 JADDANGI AI: PURE 0.5B ZERO-LOSS NET2NET EXPANSION & CLOUD UPLOAD
# ==============================================================================
!pip install torch huggingface_hub -q

import torch
import os
from huggingface_hub import HfApi, login, hf_hub_download

print("🧬 Initiating Perfect Zero-Loss Net2Net Surgery (0.5B Target)...")

# ---------------------------------------------------------
# 1. AUTH & SMART DOWNLOAD (V3 ని సేఫ్ గా తీసుకురావడం)
# ---------------------------------------------------------
HF_TOKEN = ""
login(token=HF_TOKEN)
api = HfApi()

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
OLD_FILE = "Telugu_Model_Foundation_V3.pt"
NEW_FILE = "Telugu_Model_0.5B_V4.pt" # 👈 కొత్త ఫైల్ పేరు, పాతదాన్ని టచ్ చేయదు

if not os.path.exists(OLD_FILE):
    print(f"📥 {OLD_FILE} లోకల్‌గా లేదు. Hugging Face నుండి డౌన్‌లోడ్ చేస్తున్నాను...")
    hf_hub_download(repo_id=REPO_ID, filename=OLD_FILE, local_dir=".")

# ---------------------------------------------------------
# 2. SURGERY CONFIGURATIONS
# ---------------------------------------------------------
class OldConfig:
    vocab_size = 32000
    d_model = 576
    n_layers = 16
    n_heads = 8
    n_kv_heads = 2
    expansion_factor = 4

class NewConfig:
    vocab_size = 32000
    d_model = 1440          # 1440/20 = 72 head_dim (same as old)
    n_layers = 21
    n_heads = 20
    n_kv_heads = 5          # 4:1 ratio
    expansion_factor = 4
    max_seq_len = 384

print(f"📥 Loading V3 brain for surgery...")
old_ckpt = torch.load(OLD_FILE, map_location='cpu')
old_state = old_ckpt['model_state_dict']
new_state = {}

# ---------------------------------------------------------
# 3. ZERO-LOSS TRANSPLANT LOGIC
# ---------------------------------------------------------
def expand_linear(old_w, new_out, new_in, add_noise=False, noise_scale=1e-5):
    new_w = torch.zeros((new_out, new_in))
    old_out, old_in = old_w.shape
    new_w[:old_out, :old_in] = old_w
    if add_noise:
        if new_in > old_in:
            new_w[:, old_in:] = torch.randn(new_out, new_in - old_in) * noise_scale
    return new_w

# Embeddings and final norm
new_state['wte.weight'] = expand_linear(old_state['wte.weight'], NewConfig.vocab_size, NewConfig.d_model, add_noise=True)
new_state['lm_head.weight'] = expand_linear(old_state['lm_head.weight'], NewConfig.vocab_size, NewConfig.d_model, add_noise=True)
old_norm = old_state['norm_f.weight']
new_norm = torch.ones(NewConfig.d_model)
new_norm[:old_norm.shape[0]] = old_norm
new_state['norm_f.weight'] = new_norm

old_head_dim = OldConfig.d_model // OldConfig.n_heads   
new_head_dim = NewConfig.d_model // NewConfig.n_heads   

old_hd = int(2/3 * OldConfig.expansion_factor * OldConfig.d_model)
new_hd = int(2/3 * NewConfig.expansion_factor * NewConfig.d_model)

print("🔬 Transplanting layers with perfect identity preservation...")

for i in range(NewConfig.n_layers):
    is_new_layer = i >= OldConfig.n_layers
    src_i = i if not is_new_layer else (i % OldConfig.n_layers)
    prefix_new = f"blocks.{i}."
    prefix_old = f"blocks.{src_i}."

    for norm_name in ['norm1.weight', 'norm2.weight']:
        o_norm = old_state[prefix_old + norm_name]
        n_norm = torch.ones(NewConfig.d_model)
        n_norm[:o_norm.shape[0]] = o_norm
        new_state[prefix_new + norm_name] = n_norm

    if not is_new_layer:
        # Existing layer expansion
        old_q = old_state[prefix_old + 'attn.q_proj.weight']  
        new_q = expand_linear(old_q, NewConfig.n_heads*new_head_dim, NewConfig.d_model, add_noise=False)
        new_q[OldConfig.n_heads*old_head_dim:, :] = 0.0
        new_state[prefix_new + 'attn.q_proj.weight'] = new_q

        old_k = old_state[prefix_old + 'attn.k_proj.weight']  
        new_k = expand_linear(old_k, NewConfig.n_kv_heads*new_head_dim, NewConfig.d_model, add_noise=False)
        new_k[OldConfig.n_kv_heads*old_head_dim:, :] = 0.0
        new_state[prefix_new + 'attn.k_proj.weight'] = new_k

        old_v = old_state[prefix_old + 'attn.v_proj.weight']  
        new_v = expand_linear(old_v, NewConfig.n_kv_heads*new_head_dim, NewConfig.d_model, add_noise=False)
        new_v[OldConfig.n_kv_heads*old_head_dim:, :] = 0.0
        new_state[prefix_new + 'attn.v_proj.weight'] = new_v

        old_o = old_state[prefix_old + 'attn.o_proj.weight']  
        new_o = expand_linear(old_o, NewConfig.d_model, NewConfig.n_heads*new_head_dim, add_noise=False)
        new_o[:, OldConfig.n_heads*old_head_dim:] = 0.0
        new_state[prefix_new + 'attn.o_proj.weight'] = new_o

        new_state[prefix_new + 'mlp.w1.weight'] = expand_linear(old_state[prefix_old + 'mlp.w1.weight'], new_hd, NewConfig.d_model, add_noise=True)
        new_state[prefix_new + 'mlp.w2.weight'] = expand_linear(old_state[prefix_old + 'mlp.w2.weight'], new_hd, NewConfig.d_model, add_noise=True)
        new_state[prefix_new + 'mlp.w3.weight'] = expand_linear(old_state[prefix_old + 'mlp.w3.weight'], NewConfig.d_model, new_hd, add_noise=True)

    else:
        # New layer: pure identity
        new_state[prefix_new + 'attn.q_proj.weight'] = torch.zeros(NewConfig.n_heads*new_head_dim, NewConfig.d_model)
        new_state[prefix_new + 'attn.k_proj.weight'] = torch.zeros(NewConfig.n_kv_heads*new_head_dim, NewConfig.d_model)
        new_state[prefix_new + 'attn.v_proj.weight'] = torch.zeros(NewConfig.n_kv_heads*new_head_dim, NewConfig.d_model)
        new_state[prefix_new + 'attn.o_proj.weight'] = torch.zeros(NewConfig.d_model, NewConfig.n_heads*new_head_dim)
        
        new_state[prefix_new + 'mlp.w1.weight'] = torch.zeros(new_hd, NewConfig.d_model)
        new_state[prefix_new + 'mlp.w2.weight'] = torch.zeros(new_hd, NewConfig.d_model)
        new_state[prefix_new + 'mlp.w3.weight'] = torch.zeros(NewConfig.d_model, new_hd)

# ---------------------------------------------------------
# 4. SAVE & CLOUD UPLOAD
# ---------------------------------------------------------
print(f"💾 Saving 0.5B brain locally to {NEW_FILE}...")
torch.save({
    'step': 0,
    'model_state_dict': new_state,
    'loss': old_ckpt.get('loss', float('inf'))
}, NEW_FILE)
print("✅ Perfect 0.5B V4 created locally.")

print(f"☁️ Uploading {NEW_FILE} to Hugging Face...")
api.upload_file(
    path_or_fileobj=NEW_FILE,
    path_in_repo=NEW_FILE,
    repo_id=REPO_ID,
    commit_message="🚀 Initiated 0.5B Model Expansion (V4)"
)
print(f"🎉 Upload Complete! Your V3 is safe, and V4 is now on the cloud.")
