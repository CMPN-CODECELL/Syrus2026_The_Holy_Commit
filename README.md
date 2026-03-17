# JewelForge v2

**AI-powered 2D → 3D jewelry customization with agentic orchestration**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                               │
│  ┌───────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Upload   │  │  Three.js Viewer │  │   Agent Chat     │  │
│  │  Dropzone │  │  (GLB + PBR mats)│  │   (Claude API)   │  │
│  └─────┬─────┘  └────────┬────────┘  └────────┬─────────┘  │
│        │                 │                     │             │
└────────┼─────────────────┼─────────────────────┼────────────┘
         │    REST/JSON     │                     │
┌────────▼─────────────────▼─────────────────────▼────────────┐
│                    FastAPI Backend                            │
│  POST /api/upload    POST /api/chat/{id}   GET /api/files    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │             JewelForgeAgent (Claude + tools)          │   │
│  └──────────────────────────────────────────────────────┘   │
│         │              │             │             │          │
│  ┌──────▼───┐  ┌───────▼──┐  ┌──────▼───┐  ┌────▼──────┐  │
│  │ preproc  │  │  GDINO   │  │  TripoSG │  │  Texture  │  │
│  │  (rembg) │  │ + SAM 2  │  │   mesh   │  │   baker   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Terminal 1 — Backend
cd ~/jewelforge/backend
cp .env.example .env          # add your ANTHROPIC_API_KEY
pip install -r requirements.txt
bash ../scripts/download_weights.sh
python verify_setup.py
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd ~/jewelforge/frontend
npm install
npm run dev                   # http://localhost:5173
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| CUDA | 11.8+ (GPU required for GDINO + SAM 2 + TripoSG) |
| Node.js | 18+ |
| GPU VRAM | ≥ 10 GB recommended |

**Python packages** (installed via `requirements.txt`):
- FastAPI, uvicorn, rembg, Pillow, numpy, trimesh, opencv-python-headless
- anthropic, scikit-learn, xatlas, pygltflib, huggingface_hub

**Model repos** (clone into `backend/models/`):
```bash
cd backend/models
git clone https://github.com/IDEA-Research/GroundingDINO.git
git clone https://github.com/facebookresearch/sam2.git
```

**Optional projector** (better texture quality):
```bash
pip install nvdiffrast        # NVIDIA only
# or
pip install pytorch3d         # see pytorch3d install docs for CUDA version
```

---

## Key Files

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI server + endpoints |
| `backend/agent/orchestrator.py` | Claude-powered agent with tool-calling |
| `backend/pipeline.py` | Sequential pipeline orchestration |
| `backend/segment/gdino_sam2.py` | Grounding DINO + SAM 2 wrapper |
| `backend/gen3d/triposg.py` | TripoSG/TripoSR 3D generation wrapper |
| `backend/texture/bake_and_project.py` | Texture baking + face labeling |
| `backend/pricing/engine.py` | Price estimation + budget alternatives |
| `frontend/src/components/JewelryViewer.jsx` | React Three Fiber 3D canvas |
| `frontend/src/components/AgentChat.jsx` | Chat UI |
| `scripts/test_pipeline.py` | End-to-end pipeline test |

---

## Team

- **Person A** — Pipeline (preprocess → segment → 3D → texture)
- **Person B** — Agent (orchestrator, tools, Claude API integration)
- **Person C** — Frontend (Three.js viewer, material panel, chat UI)
