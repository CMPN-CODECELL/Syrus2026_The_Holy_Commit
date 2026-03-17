# JewelForge v2

Agentic AI-Driven 2D → 3D Jewelry Customization Pipeline

Upload a 2D jewelry image → AI segments it into components, generates a 3D model, bakes textures, and lets you customize materials via natural language chat with real-time price estimation.

## Quick Start

```bash
# 1. Start backend
cd backend
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
uvicorn main:app --reload --port 8000

# 2. Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Setup

### Prerequisites

- Python 3.10+
- Node.js v20+
- CUDA 12.1+ with 8GB+ VRAM (RTX 3070 or better)
- NVIDIA driver 535+

### Installation

```bash
# Clone model repos
cd backend/models
git clone https://github.com/IDEA-Research/GroundingDINO.git
git clone https://github.com/facebookresearch/sam2.git
git clone https://github.com/VAST-AI-Research/TripoSG.git

# Install Python dependencies
cd backend
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart rembg[gpu] Pillow numpy trimesh \
    opencv-python-headless scikit-learn anthropic xatlas einops huggingface_hub
pip install -e models/GroundingDINO
pip install -e models/sam2
pip install -r models/TripoSG/requirements.txt
pip install nvdiffrast  # optional GPU rasterization

# Download weights
bash scripts/download_weights.sh

# Verify setup
cd backend && python verify_setup.py
```

## Architecture

Three autonomous behaviors:

1. **Pipeline self-correction** — Segmentation runs, checks its own results, retries with different prompts/thresholds, falls back to color-based segmentation.

2. **Mesh quality validation** — After 3D generation, the system checks if the mesh is collapsed/flat, retries with a rotated input image, or falls back to a cached demo mesh.

3. **Multi-step goal planning** — User says "make this premium under Rs. 80000" → LLM plans a sequence (white gold + CZ), executes multiple tool calls, verifies the budget constraint, then responds.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload image, run full pipeline |
| POST | `/api/chat/{job_id}` | NL chat (agentic LLM loop) |
| POST | `/api/swap/{job_id}` | Direct material swap (no LLM) |
| GET  | `/api/files/{id}/{file}` | Serve generated files |
| GET  | `/api/materials` | Material presets |
| GET  | `/api/health` | Health check |

## VRAM Sequence

Models are loaded/unloaded sequentially to stay within 8GB:

```
GDINO+SAM2 (2.5GB) → unload → TripoSG (6GB) → unload → nvdiffrast (0.5GB)
```

## File Structure

```
jewelforge/
├── backend/
│   ├── agent/          # Orchestrator, segmentation agent, mesh validator, prompts
│   ├── segment/        # Grounding DINO + SAM 2 wrapper
│   ├── gen3d/          # TripoSG wrapper
│   ├── texture/        # Texture baking + mask projection
│   ├── pricing/        # Price engine + comparator
│   ├── materials/      # PBR presets (source of truth)
│   ├── main.py         # FastAPI server
│   └── preprocess.py   # rembg preprocessing
├── frontend/
│   └── src/
│       ├── components/ # React components
│       ├── hooks/      # Custom React hooks
│       ├── materials/  # Auto-generated JS presets
│       └── api/        # API client
└── scripts/
    ├── download_weights.sh
    ├── sync_presets_to_frontend.py
    └── test_agentic.py
```

## Testing

```bash
# Test all 3 agentic behaviors (no GPU needed for most tests)
cd backend && python ../scripts/test_agentic.py

# Verify full setup
cd backend && python verify_setup.py
```
