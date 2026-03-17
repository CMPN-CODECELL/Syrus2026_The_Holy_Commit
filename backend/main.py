"""
JewelForge v2 — FastAPI server.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.orchestrator import JewelForgeAgent
from materials.presets import GEMSTONE_PRESETS, METAL_PRESETS
from pipeline import generate_job_id

app = FastAPI(title="JewelForge v2", version="2.0.0")

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store ──────────────────────────────────────────────────────────
# Maps job_id → JewelForgeAgent instance
_sessions: Dict[str, JewelForgeAgent] = {}

_OUTPUTS_DIR = Path("outputs")
_UPLOADS_DIR = Path("uploads")
_OUTPUTS_DIR.mkdir(exist_ok=True)
_UPLOADS_DIR.mkdir(exist_ok=True)


# ── Pydantic models ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "JewelForge v2"}


@app.get("/api/materials")
async def get_materials():
    """Return all PBR material presets."""
    return {"metals": METAL_PRESETS, "gemstones": GEMSTONE_PRESETS}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Accept a jewelry image, create a job, run the full pipeline via the agent,
    and return the resulting mesh URL, labels URL, components, and price.
    """
    job_id = generate_job_id()
    upload_dir = _UPLOADS_DIR / job_id
    output_dir = _OUTPUTS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    image_path = upload_dir / f"original{suffix}"
    with open(image_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Create agent for this session
    agent = JewelForgeAgent()
    _sessions[job_id] = agent

    # Run pipeline via agent (agent tracks job state internally)
    try:
        result = agent.process_message(
            user_message=f"I just uploaded a jewelry image. Please run the full pipeline on it.",
            image_path=str(image_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    state = agent.current_state
    mesh_filename = Path(state.get("mesh_path", "")).name if state.get("mesh_path") else None
    labels_filename = Path(state.get("labels_path", "")).name if state.get("labels_path") else None

    return {
        "job_id": job_id,
        "mesh_url": f"/api/files/{job_id}/{mesh_filename}" if mesh_filename else None,
        "labels_url": f"/api/files/{job_id}/{labels_filename}" if labels_filename else None,
        "components": state.get("components", []),
        "price": state.get("price_estimate"),
        "agent_greeting": result.get("response_text", ""),
    }


@app.post("/api/chat/{job_id}")
async def chat(job_id: str, request: ChatRequest):
    """
    Send a chat message to the agent for an existing job.
    Returns the agent's text response and any actions executed.
    """
    agent = _sessions.get(job_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    try:
        result = agent.process_message(user_message=request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state = agent.current_state
    return {
        "response_text": result.get("response_text", ""),
        "actions": result.get("actions", []),
        "price": state.get("price_estimate"),
        "materials_applied": state.get("materials_applied", {}),
    }


@app.get("/api/files/{job_id}/{filename}")
async def serve_file(job_id: str, filename: str):
    """Serve a generated file (GLB, JSON, PNG) for a given job."""
    # Sanitise: prevent path traversal
    if ".." in job_id or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path.")

    file_path = _OUTPUTS_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".glb": "model/gltf-binary",
        ".obj": "model/obj",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".stl": "model/stl",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(str(file_path), media_type=media_type)
