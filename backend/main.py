# JewelForge v2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import HTTPException
import shutil
import uuid
import os
import json

from agent.orchestrator import JewelForgeAgent
from materials.presets import METAL_PRESETS, GEMSTONE_PRESETS

app = FastAPI(title="JewelForge v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, JewelForgeAgent] = {}


@app.post("/api/upload")
async def upload(image: UploadFile = File(...), jewelry_type: str = "auto"):
    job_id = str(uuid.uuid4())[:8]
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{job_id}.png"
    with open(path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    agent = JewelForgeAgent()
    sessions[job_id] = agent
    result = agent.run_pipeline(path, jewelry_type)
    return {"job_id": job_id, **result}


@app.post("/api/chat/{job_id}")
async def chat(job_id: str, body: dict):
    agent = sessions.get(job_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    return agent.chat(body.get("text", ""))


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
