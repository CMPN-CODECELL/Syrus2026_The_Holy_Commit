"""
FastAPI backend for the TripoSG Jewelry 3D Demo.

Endpoints:
  POST /generate   — Upload an image, run the full pipeline, return paths.
  GET  /outputs/*  — Static file serving for generated outputs.
  GET  /health     — Health check.
  GET  /*          — Serves the frontend (index.html, viewer.js) so that
                     window.location.origin in the browser always resolves
                     to this server's host:port, eliminating any port mismatch.

Run from the TripoSG directory:
  uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
"""
import io
import os
import shutil
import sys
import uuid
import logging
from contextlib import asynccontextmanager

# Make CUDA errors surface at the exact kernel that caused them rather than
# at a later async call. Also serialises kernel launches which prevents the
# "illegal memory access" caused by two back-to-back RMBG runs racing each other.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
# Reduce VRAM fragmentation — critical on ≤8 GB GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image as PILImage
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# --- Path setup -----------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from huggingface_hub import snapshot_download
from briarmbg import BriaRMBG
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from pipeline.run_pipeline import run_pipeline
from utils.segment_ring import load_dino, load_sam2

# --- Config ---------------------------------------------------------------
TRIPOSG_WEIGHTS = os.path.join(REPO_ROOT, "pretrained_weights", "TripoSG")
RMBG_WEIGHTS    = os.path.join(REPO_ROOT, "pretrained_weights", "RMBG-1.4")
OUTPUTS_DIR     = os.path.join(REPO_ROOT, "outputs")
UPLOADS_DIR     = os.path.join(REPO_ROOT, "uploads")
FRONTEND_DIR    = os.path.join(REPO_ROOT, "frontend")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("triposg-server")

# --- Model state (loaded once at startup) ---------------------------------
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once on startup."""
    log.info("Downloading / loading pretrained weights...")

    # Download weights if not present
    if not os.path.isfile(os.path.join(TRIPOSG_WEIGHTS, "model_index.json")):
        log.info("Downloading TripoSG weights from HuggingFace...")
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=TRIPOSG_WEIGHTS)

    if not os.path.isfile(os.path.join(RMBG_WEIGHTS, "config.json")):
        log.info("Downloading RMBG-1.4 weights from HuggingFace...")
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=RMBG_WEIGHTS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # float16 is required to fit TripoSG in ≤8 GB VRAM.
    # The prior "illegal memory access" was caused by the tensor-squeeze bug
    # in visualize_masks.py (now fixed), not by float16 itself.
    dtype = torch.float16 if device == "cuda" else torch.float32
    log.info(f"Using device={device}, dtype={dtype}")

    rmbg_net = BriaRMBG.from_pretrained(RMBG_WEIGHTS).to(device)
    rmbg_net.eval()
    models["rmbg_net"] = rmbg_net

    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(TRIPOSG_WEIGHTS).to(device, dtype)
    models["pipe"] = pipe

    # Pre-load DINO + SAM2 onto CPU to avoid per-request HF download overhead.
    # They are moved to CUDA only during the segmentation step (run_pipeline.py),
    # then moved back to CPU so they don't occupy VRAM during TripoSG diffusion.
    dino_processor, dino_model = load_dino("cpu")
    models["dino_processor"] = dino_processor
    models["dino_model"]     = dino_model

    sam2_model = load_sam2("cpu")
    models["sam2_model"] = sam2_model

    log.info("Models loaded. Server ready.")
    yield
    models.clear()


# --- App ------------------------------------------------------------------
app = FastAPI(title="TripoSG Jewelry Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static output files so the frontend can load images and GLBs
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# --- Endpoints ------------------------------------------------------------

# Serve frontend files explicitly so API routes are never shadowed.
# Mounting StaticFiles at "/" would intercept /health and /generate.
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/viewer.js")
def serve_viewer_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "viewer.js"), media_type="application/javascript")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    seed: int = 42,
    num_steps: int = 50,
    guidance_scale: float = 7.0,
):
    """
    Upload a jewelry image and receive segmentation overlay + GLB mesh paths.

    Returns:
        {
            "segmentation": "/outputs/segmented_overlay.png",
            "mesh":         "/outputs/generated_mesh.glb"
        }
    """
    if "pipe" not in models:
        raise HTTPException(503, "Models not yet loaded")

    # Validate MIME type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image, got {file.content_type}")

    # Read the file bytes once so we can validate size and decodability
    file_bytes = await file.read()

    # Validate file size (≤ 20 MB)
    MAX_BYTES = 20 * 1024 * 1024
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(
            413,
            f"File too large: {len(file_bytes) / (1024*1024):.1f} MB. Maximum is 20 MB."
        )

    # Validate that PIL can actually decode the image
    try:
        PILImage.open(io.BytesIO(file_bytes)).verify()
    except Exception:
        raise HTTPException(422, "File could not be decoded as an image.")

    # Save uploaded image with unique name to avoid collisions
    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    unique_id = uuid.uuid4().hex[:8]
    upload_path = os.path.join(UPLOADS_DIR, f"{unique_id}{ext}")

    with open(upload_path, "wb") as f:
        f.write(file_bytes)

    log.info(f"Received upload: {upload_path}")

    try:
        result = run_pipeline(
            image_path=upload_path,
            output_dir=OUTPUTS_DIR,
            pipe=models["pipe"],
            rmbg_net=models["rmbg_net"],
            seed=seed,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            dino_processor=models.get("dino_processor"),
            dino_model=models.get("dino_model"),
            sam2_model=models.get("sam2_model"),
        )
    except Exception as exc:
        log.exception("Pipeline failed")
        raise HTTPException(500, f"Pipeline error: {exc}")

    src = "/home/saranshlulla/Documents/Syrus/output_boxes.png"
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(OUTPUTS_DIR, "seg_boxes.png"))

    # Return URL paths relative to this server.
    # seg_boxes / seg_masks are only present when DINO+SAM2 ran (not HSV fallback).
    resp = {
        "segmentation": "/outputs/seg_boxes.png",
        "mesh":         "/outputs/generated_mesh.glb",
        "labels":       "/outputs/generated_mesh_labels.json",
        "seg_debug":    "/outputs/seg_debug.png",
    }
    if result.get("seg_boxes"):
        resp["seg_boxes"] = "/outputs/seg_boxes.png"
    if result.get("seg_masks"):
        resp["seg_masks"] = "/outputs/seg_masks.png"
    return JSONResponse(resp)
