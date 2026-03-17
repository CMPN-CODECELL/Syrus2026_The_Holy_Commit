#!/bin/bash
set -e

WEIGHTS_DIR=~/jewelforge/backend/weights
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  JewelForge v2 — Downloading model weights"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Grounding DINO ─────────────────────────────────────────────────────────
echo ""
echo "Downloading Grounding DINO weights..."
wget -q --show-progress \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# ── SAM 2 ──────────────────────────────────────────────────────────────────
echo ""
echo "Downloading SAM 2.1 (hiera_small) weights..."
wget -q --show-progress \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# ── TripoSG / TripoSR via Hugging Face ────────────────────────────────────
echo ""
echo "Downloading TripoSR weights from Hugging Face..."
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

dest = os.path.join(os.environ.get("HOME", ""), "jewelforge", "backend", "weights", "TripoSR")
print(f"  Destination: {dest}")
snapshot_download(
    repo_id="stabilityai/TripoSR",
    local_dir=dest,
    local_dir_use_symlinks=False,
)
print("  TripoSR download complete.")
PYEOF

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All weights downloaded!"
echo ""
ls -lh "$WEIGHTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Clone model repos:"
echo "     cd ~/jewelforge/backend/models"
echo "     git clone https://github.com/IDEA-Research/GroundingDINO.git"
echo "     git clone https://github.com/facebookresearch/sam2.git"
echo "  2. Run verify_setup.py to confirm everything is ready."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
