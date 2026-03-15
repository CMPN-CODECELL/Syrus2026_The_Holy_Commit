# Jewelry 3D Demo — Run Instructions

## Prerequisites

```bash
pip install fastapi uvicorn python-multipart
# All other deps are in requirements.txt
pip install -r requirements.txt
```

GPU with CUDA is required (the TripoSG pipeline uses CUDA throughout).

---

## Terminal 1 — Backend

Run from the **repo root** (`TripoSG/`):

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

On first run the server downloads pretrained weights (~2 GB) from HuggingFace.
Wait for the log line: **"Models loaded. Server ready."**

Health check: http://localhost:8000/health

---

## Terminal 2 — Frontend

```bash
cd frontend
python -m http.server 3000
```

Open **http://localhost:3000** in your browser.

---

## Using the Demo

1. **Upload** a jewelry image (JPG/PNG).
2. Click **Generate 3D Model** — segmentation + mesh generation runs (~1–3 min on GPU).
3. **Segmentation overlay** appears in the left panel showing the detected jewelry region.
4. The **3D model** loads in the right viewer — drag to rotate, scroll to zoom.
5. Click **Gold / Silver / Ruby / Sapphire / Emerald / Platinum** to switch materials instantly.

---

## Architecture

```
Image upload
    │
    ▼
server/main.py          (FastAPI, POST /generate)
    │
    ▼
pipeline/run_pipeline.py
    ├── utils/visualize_masks.py   → segmented_overlay.png  (RMBG mask + OpenCV overlay)
    └── scripts/inference_triposg.py → run_triposg()        → generated_mesh.glb
                                         └── scripts/image_process.py (RMBG preprocessing)
    │
    ▼
outputs/
    ├── segmented_overlay.png
    └── generated_mesh.glb

frontend/
    ├── index.html    (UI, file upload, material switcher)
    └── viewer.js     (Three.js GLTFLoader + OrbitControls + MeshPhysicalMaterial)
```

## Notes

- Segmentation uses **RMBG-1.4** (existing background removal model), which cleanly separates jewelry from background and generates the colored mask overlay.
- SAM2 / GroundingDINO are not in this repo; RMBG serves the same purpose for clean jewelry images.
- Each `/generate` call overwrites `outputs/segmented_overlay.png` and `outputs/generated_mesh.glb`.
