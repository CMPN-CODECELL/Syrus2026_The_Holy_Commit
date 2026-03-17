# JewelForge v2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi import HTTPException
import shutil
import uuid
import os
import traceback
import numpy as np


def _np_safe(obj):
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _np_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_np_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

from agent.orchestrator import JewelForgeAgent
from materials.presets import METAL_PRESETS, GEMSTONE_PRESETS

app = FastAPI(title="JewelForge v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, JewelForgeAgent] = {}


@app.post("/api/upload")
async def upload(image: UploadFile = File(...), jewelry_type: str = "auto"):
    try:
        job_id = str(uuid.uuid4())[:8]
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{job_id}.png"
        with open(path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        agent = JewelForgeAgent()
        result = agent.run_pipeline(path, jewelry_type, job_id=job_id)
        sessions[job_id] = agent
        return JSONResponse(content=_np_safe({"job_id": job_id, **result}))
    except Exception as e:
        traceback.print_exc()
        fallback_job_id = locals().get("job_id", "unknown")
        return JSONResponse(
            status_code=500,
            content=_np_safe({
                "error": str(e),
                "job_id": fallback_job_id,
                "message": "Upload failed before a usable pipeline result was created.",
            }),
        )


@app.post("/api/chat/{job_id}")
async def chat(job_id: str, body: dict):
    agent = sessions.get(job_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return JSONResponse(content=_np_safe(agent.chat(body.get("text", ""))))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Chat request failed."},
        )


@app.post("/api/swap/{job_id}")
async def swap(job_id: str, body: dict):
    """Direct material swap from UI click — no LLM."""
    agent = sessions.get(job_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    return agent.direct_swap(body["component"], body["material"])


@app.get("/api/files/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    path = f"outputs/{job_id}/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/api/materials")
async def materials():
    return {"metals": METAL_PRESETS, "gemstones": GEMSTONE_PRESETS}


@app.get("/api/health")
async def health():
    return {"status": "ok"}
