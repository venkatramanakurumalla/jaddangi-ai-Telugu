# ==============================================================================
# 🧬 JADDANGI AI: LOW-RAM NET2DEEPER EXPANSION (0.5B -> 1B)
# Optimized to prevent Colab/Notebook runtime disconnects
# ==============================================================================
!pip install torch huggingface_hub -q

import torch
import os
import gc
from datetime import datetime, timezone
from huggingface_hub import HfApi, login, hf_hub_download

print("🧬 Initiating Low-RAM Net2Deeper Expansion...")

# ---------------------------------------------------------
# 1. CONFIGURATION & CLOUD FETCH
# ---------------------------------------------------------
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # 👈 PLACE YOUR HUGGING FACE TOKEN HERE
login(token=HF_TOKEN)
api = HfApi()

REPO_ID = "VenkataRamanaKurumallajaddangi/Telugu"
OLD_FILE = "Telugu_Model_0.5B_V4.pt"
NEW_FILE = "Telugu_Model_1B_V5.pt"
EXPANSION_FACTOR = 2 

if not os.path.exists(OLD_FILE):
    print(f"📥 Downloading {OLD_FILE} from Hugging Face...")
    hf_hub_download(repo_id=REPO_ID, filename=OLD_FILE, local_dir=".")

# ---------------------------------------------------------
# 2. LOAD & BUILD ARCHITECTURE MAP
# ---------------------------------------------------------
print(f"📥 Loading source state dictionary...")
# Load directly to CPU, avoiding GPU RAM spikes
old_ckpt = torch.load(OLD_FILE, map_location="cpu")
old_state = old_ckpt.get("model_state_dict", old_ckpt)
new_state = {}

prefix_str = "blocks." if any(k.startswith("blocks.") for k in old_state.keys()) else "model.layers."
OLD_LAYERS = len(set(int(k.split(prefix_str)[1].split(".")[0]) for k in old_state.keys() if k.startswith(prefix_str)))
NEW_LAYERS = OLD_LAYERS * EXPANSION_FACTOR

blueprint = {k[len(f"{prefix_str}0."):]: v for k, v in old_state.items() if k.startswith(f"{prefix_str}0.")}
print(f"🔍 Discovered Structure: {OLD_LAYERS} layers. Target: {NEW_LAYERS} layers.")

# ---------------------------------------------------------
# 3. CLONING & IDENTITY INITIALIZATION
# ---------------------------------------------------------
print(f"🧬 Executing Depth Expansion (Append Strategy)...")

# A. Copy non-layer parameters
for key in [k for k in old_state.keys() if not k.startswith(prefix_str)]:
    new_state[key] = old_state[key].clone()

# B. Copy original layers
for i in range(OLD_LAYERS):
    for sub_k in blueprint:
        new_state[f"{prefix_str}{i}.{sub_k}"] = old_state[f"{prefix_str}{i}.{sub_k}"].clone()
        
# C. Append identity layers
for i in range(OLD_LAYERS, NEW_LAYERS):
    for sub_k, template in blueprint.items():
        key_lower = sub_k.lower()
        full_key = f"{prefix_str}{i}.{sub_k}"
        
        if "norm" in key_lower and sub_k.endswith(".weight"): 
            new_state[full_key] = torch.ones_like(template)
        else:
            new_state[full_key] = torch.zeros_like(template)

# ---------------------------------------------------------
# 4. FREE UP RAM BEFORE SAVING (CRITICAL FIX)
# ---------------------------------------------------------
print("🧹 Clearing old memory to prevent crashes...")
del old_state
del old_ckpt
gc.collect()  # Force garbage collection to reclaim RAM

# ---------------------------------------------------------
# 5. METADATA & LOCAL SAVE
# ---------------------------------------------------------
old_config = {
    'vocab_size': 32000, 'd_model': 1440, 'n_layers': OLD_LAYERS,
    'n_heads': 20, 'n_kv_heads': 5, 'expansion_factor': 4, 'max_seq_len': 384
}
new_config = old_config.copy()
new_config["n_layers"] = NEW_LAYERS

export_payload = {
    "step": 0,
    "model_state_dict": new_state,
    "loss": float("inf"),
    "config": new_config,
    "expansion_metadata": {
        "source": OLD_FILE,
        "type": "Net2Deeper",
        "original_layers": OLD_LAYERS,
        "expanded_layers": NEW_LAYERS,
    }
}

print(f"💾 Writing massive 1B checkpoint to disk (this will take a minute)...")
torch.save(export_payload, NEW_FILE)

# Free up the new state from RAM too, before uploading!
del new_state
del export_payload
gc.collect()

# ---------------------------------------------------------
# 6. DIRECT UPLOAD TO CLOUD
# ---------------------------------------------------------
print(f"☁️ Uploading {NEW_FILE} to Hugging Face...")
api.upload_file(
    path_or_fileobj=NEW_FILE,
    path_in_repo=NEW_FILE,
    repo_id=REPO_ID,
    commit_message=f"🚀 Low-RAM Net2Deeper Expansion ({OLD_LAYERS} -> {NEW_LAYERS} Layers)"
)
print("🎉 Success! The 1B Model is safely uploaded.")
