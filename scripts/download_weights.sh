#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)/backend"
WEIGHTS_DIR="$BACKEND_DIR/weights"

mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

echo "Downloading Grounding DINO weights..."
wget -q --show-progress \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

echo "Downloading SAM 2.1 weights..."
wget -q --show-progress \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

echo "Downloading SAM 2.1 config..."
wget -q --show-progress \
  https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
  -O sam2.1_hiera_s.yaml

echo "Downloading TripoSR weights from HuggingFace..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/TripoSR', local_dir='TripoSR')
print('TripoSR downloaded.')
"

echo ""
echo "All weights downloaded to: $WEIGHTS_DIR"
ls -lh "$WEIGHTS_DIR"
