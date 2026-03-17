"""
End-to-end pipeline orchestration for JewelForge v2.
Called by the agent's tool implementations.
"""

from __future__ import annotations

import gc
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def generate_job_id() -> str:
    """Return a random 8-character hex job ID."""
    return uuid.uuid4().hex[:8]


def run_full_pipeline(
    image_path: str,
    output_dir: str,
    jewelry_type: str = "auto",
) -> Dict:
    """
    Orchestrate all pipeline stages with sequential VRAM management.

    Stages:
        1. Preprocess (rembg + center-crop + resize)
        2. Segmentation (Grounding DINO + SAM 2)
        3. 3D generation (TripoSG/TripoSR)
        4. Texture baking + mask projection (nvdiffrast/trimesh)
        5. Price estimation

    Args:
        image_path:   Path to the raw uploaded image.
        output_dir:   Directory for all generated outputs.
        jewelry_type: "ring" | "necklace" | "earring" | "bracelet" | "auto"

    Returns:
        Dict with keys: mesh_path, labels_path, texture_path,
                        components, face_to_component, price, job_id
    """
    os.makedirs(output_dir, exist_ok=True)
    _clear_cuda()

    # ── Stage 1: Preprocess ────────────────────────────────────────────────
    from preprocess import preprocess_image

    prep = preprocess_image(image_path, output_dir, target_size=512)
    rgb_path = prep["rgb"]
    rgba_path = prep["rgba"]

    # ── Stage 2: Segment ───────────────────────────────────────────────────
    from segment.gdino_sam2 import JewelrySegmenter

    segmenter = JewelrySegmenter(device=_device())
    segmenter.load_models()
    segments = segmenter.segment(rgba_path, jewelry_type=jewelry_type)
    segmenter.unload_models()
    _clear_cuda()

    component_names = list({s["label"] for s in segments})

    # ── Stage 3: 3D generation ────────────────────────────────────────────
    from gen3d.triposg import generate_3d_mesh

    mesh_path = generate_3d_mesh(rgb_path, output_dir, resolution=256)
    _clear_cuda()

    # ── Stage 4: Texture + mask projection ────────────────────────────────
    from texture.bake_and_project import MeshProjector

    projector = MeshProjector(device=_device())
    projector.load_mesh(mesh_path)

    texture_path = os.path.join(output_dir, "texture.png")
    projector.bake_texture(rgb_path, texture_path, texture_size=1024)

    face_to_component = projector.project_masks_to_faces(segments, image_size=512)

    glb_path = os.path.join(output_dir, "mesh_labeled.glb")
    glb_path, labels_path = projector.export_labeled_glb(
        face_to_component, texture_path, glb_path
    )
    _clear_cuda()

    # ── Stage 5: Price estimation ─────────────────────────────────────────
    from pricing.engine import estimate_price

    inferred_config = _infer_config_from_components(component_names, jewelry_type)
    price = estimate_price(inferred_config)

    return {
        "mesh_path": glb_path,
        "labels_path": labels_path,
        "texture_path": texture_path,
        "components": component_names,
        "face_to_component": {str(k): v for k, v in face_to_component.items()},
        "price": price,
        "jewelry_type": inferred_config["jewelry_type"],
    }


def build_label_map(segments: List[Dict], height: int, width: int) -> np.ndarray:
    """
    Combine a list of segment dicts into a single (H, W) integer label map.

    Background = -1. Higher segment_id wins on overlap.

    Args:
        segments: List of segment dicts from JewelrySegmenter.segment().
        height: Output map height in pixels.
        width: Output map width in pixels.

    Returns:
        np.ndarray of shape (H, W), dtype int32. Value is segment_id or -1.
    """
    import cv2

    label_map = np.full((height, width), -1, dtype=np.int32)
    for seg in sorted(segments, key=lambda s: s.get("area_fraction", 0)):
        mask: np.ndarray = seg["mask"]
        seg_id: int = seg["segment_id"]

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        label_map[mask] = seg_id

    return label_map


# ── Private helpers ────────────────────────────────────────────────────────

def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _infer_config_from_components(
    component_names: List[str], jewelry_type: str
) -> Dict:
    """Map detected component names to a pricing config dict."""
    # Detect jewelry type from components if "auto"
    if jewelry_type == "auto":
        if "shank" in component_names or "band" in " ".join(component_names):
            jewelry_type = "ring"
        elif "chain" in component_names:
            jewelry_type = "necklace"
        elif "hoop" in component_names or "stud" in component_names:
            jewelry_type = "earring"
        else:
            jewelry_type = "ring"

    # Default config — the agent will update this via apply_material
    return {
        "jewelry_type": jewelry_type,
        "metal": "yellow_gold",
        "center_stone": "diamond",
        "accent_stone": "cubic_zirconia",
    }
