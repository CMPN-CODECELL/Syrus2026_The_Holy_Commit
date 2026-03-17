"""
TripoSG / TripoSR 3D mesh generation wrapper for JewelForge v2.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Optional

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_WEIGHTS_DIR = _BACKEND_DIR / "weights"
_TRIPOSR_DIR = _WEIGHTS_DIR / "TripoSR"


def generate_3d_mesh(
    image_path: str,
    output_dir: str,
    resolution: int = 256,
    chunk_size: int = 4096,
) -> str:
    """
    Generate a 3D mesh from a single jewelry image using TripoSG/TripoSR.

    Steps:
        1. Clear CUDA cache.
        2. Load TripoSR model from weights dir.
        3. Run inference with the input image.
        4. Marching-cubes mesh extraction at the given resolution.
        5. Export to .glb in output_dir.
        6. Unload model and clear VRAM.

    Args:
        image_path: Path to the preprocessed RGB image (512×512 PNG).
        output_dir: Directory to write the exported mesh file.
        resolution: Marching-cubes grid resolution (higher = more detail, more VRAM).
        chunk_size: Chunk size for batched NeRF field queries.

    Returns:
        Absolute path to the exported .glb file.
    """
    import sys
    import torch
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    # ── Clear CUDA cache before loading a large model ──────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load model ────────────────────────────────────────────────────────
    model = _load_triposr()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # ── Preprocess image ──────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB").resize((512, 512))

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        scene_codes = model([image], device=device)

    # ── Mesh extraction ───────────────────────────────────────────────────
    meshes = model.extract_mesh(
        scene_codes,
        resolution=resolution,
        chunk_size=chunk_size,
    )
    mesh = meshes[0]

    # ── Export ────────────────────────────────────────────────────────────
    stem = Path(image_path).stem
    glb_path = os.path.join(output_dir, f"{stem}_mesh.glb")
    mesh.export(glb_path)

    # ── Cleanup VRAM ──────────────────────────────────────────────────────
    del model, scene_codes, meshes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return glb_path


def _load_triposr():
    """
    Import and instantiate the TripoSR model.

    Tries the installed `tsr` package first (pip install tsr),
    then falls back to a local clone in models/TripoSR.
    """
    import sys

    # Try installed package
    try:
        from tsr.system import TSR  # type: ignore

        model = TSR.from_pretrained(
            str(_TRIPOSR_DIR),
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        return model
    except ImportError:
        pass

    # Try local clone
    local_triposr = _BACKEND_DIR / "models" / "TripoSR"
    if local_triposr.exists():
        sys.path.insert(0, str(local_triposr))
        from tsr.system import TSR  # type: ignore

        model = TSR.from_pretrained(
            str(_TRIPOSR_DIR),
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        return model

    raise RuntimeError(
        "TripoSR is not installed and not found in models/TripoSR. "
        "Run scripts/download_weights.sh first."
    )
