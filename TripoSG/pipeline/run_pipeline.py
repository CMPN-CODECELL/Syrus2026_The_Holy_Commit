"""
Full image-to-3D pipeline for the jewelry demo.

Steps:
  1. RMBG segmentation overlay  (segmented_overlay.png)
  2. Ring component segmentation (band + gemstone masks + seg_debug.png)
  3. TripoSG mesh generation     (generated_mesh.glb)
  4. Transfer segments → 3D      (generated_mesh_labels.json)

Reuses:
  - scripts/inference_triposg.py  → run_triposg()
  - utils/visualize_masks.py      → create_segmentation_overlay()
  - utils/segment_ring.py         → segment_ring_components()
  - utils/transfer_segments.py    → transfer_segmentation_to_3d()
"""
import logging
import os
import sys

import torch

# Ensure repo root and scripts/ are on the path so existing modules resolve.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference_triposg import run_triposg                       # noqa: E402  (existing)
from utils.transfer_segments import transfer_segmentation_to_3d  # noqa: E402  (new)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../test'))
from test_segmentation import detect_components, segment_with_sam2, visualize_and_save
import test_segmentation as _test_seg
from constants import JEWELRY_LABELS
_test_seg.JEWELRY_LABELS = JEWELRY_LABELS

from PIL import Image as _PIL_Image
import numpy as _np

log = logging.getLogger(__name__)


def run_pipeline(
    image_path: str,
    output_dir: str,
    pipe,
    rmbg_net,
    seed: int = 42,
    num_steps: int = 50,
    guidance_scale: float = 7.0,
    dino_processor=None,
    dino_model=None,
    sam2_model=None,
) -> dict:
    """
    Run the full jewelry → segmentation → 3D mesh pipeline.

    Returns:
        {
            "segmentation": "<abs_path>/segmented_overlay.png",
            "mesh":         "<abs_path>/generated_mesh.glb",
            "labels":       "<abs_path>/generated_mesh_labels.json",
            "seg_debug":    "<abs_path>/seg_debug.png",
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Flush GPU before first CUDA operation ────────────────────────
    # (create_segmentation_overlay removed — seg_masks.png is now the display image)

    # Flush GPU work before the next CUDA operation.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ── Step 2: Ring component segmentation (band + gemstone) ────────────────
    # DINO (~340 MB) + SAM 2 (~150 MB) need free VRAM.  TripoSG occupies most
    # of the GPU at this point, so we offload it to CPU first, run segmentation,
    # then reload it for mesh generation.  Without this, DINO/SAM OOM silently
    # and the pipeline falls back to HSV — which is why gemstone was never found.
    _pipe_offloaded = False
    _pipe_dtype = None
    if torch.cuda.is_available():
        try:
            _pipe_params = [p for p in pipe.parameters()]
            _pipe_target_device = _pipe_params[0].device if _pipe_params else None
            _pipe_dtype         = _pipe_params[0].dtype  if _pipe_params else None
            if _pipe_target_device and _pipe_target_device.type == "cuda":
                log.info("Offloading TripoSG to CPU to free VRAM for DINO+SAM …")
                pipe.to("cpu")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                _pipe_offloaded = True
                log.info("TripoSG offloaded. Running DINO+SAM segmentation.")
        except Exception as e:
            log.warning("Could not offload TripoSG to CPU: %s — segmentation may OOM", e)

    log.info("[PIPELINE] Running DINO+SAM2 segmentation …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_pil = _PIL_Image.open(image_path).convert("RGB")
    if dino_model is not None and torch.cuda.is_available():
        dino_model.to(device)
    detections = detect_components(image_pil, dino_processor, dino_model, device)
    masks = segment_with_sam2(image_pil, detections, device)
    GEM_KEYWORDS = {"gemstone", "gem", "diamond", "stone", "jewel"}
    for m in masks:
        raw = m["label"].lower()
        if any(k in raw for k in GEM_KEYWORDS):
            m["label"] = "gemstone"
        else:
            m["label"] = "band"
    visualize_and_save(image_pil, detections, masks,
                       output_prefix=os.path.join(output_dir, "seg"))
    img_h, img_w = image_pil.height, image_pil.width
    band_mask = _np.zeros((img_h, img_w), dtype=_np.uint8)
    gem_mask  = _np.zeros((img_h, img_w), dtype=_np.uint8)
    for m in masks:
        lbl = m["label"].lower()
        mask_bool = m["mask"].astype(bool)
        if any(kw in lbl for kw in ("gem", "diamond", "stone", "crystal", "jewel")):
            gem_mask[mask_bool] = 255
        else:
            band_mask[mask_bool] = 255
    seg = {
        "band_mask":     band_mask,
        "gemstone_mask": gem_mask,
        "components":    [m["label"] for m in masks],
        "seg_boxes":     os.path.join(output_dir, "seg_boxes.png"),
        "seg_masks":     os.path.join(output_dir, "seg_masks.png"),
        "debug_vis":     "",
    }
    log.info("Segmentation components found: %s", seg["components"])

    # Move DINO + SAM2 back to CPU now that segmentation is done.
    # This frees ~490 MB of VRAM before TripoSG diffusion inference, which can
    # require several GB of activation memory on top of the 2 GB model weights.
    if torch.cuda.is_available():
        if dino_model is not None:
            try:
                dino_model.to("cpu")
            except Exception as e:
                log.warning("Could not offload DINO to CPU: %s", e)
        if sam2_model is not None:
            try:
                sam2_model.to("cpu")
            except Exception as e:
                log.warning("Could not offload SAM2 to CPU: %s", e)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log.info("DINO + SAM2 moved to CPU. VRAM freed for TripoSG.")

    # Reload TripoSG back onto GPU before mesh generation
    if _pipe_offloaded and torch.cuda.is_available():
        try:
            log.info("Reloading TripoSG back to CUDA …")
            pipe.to("cuda", _pipe_dtype)
            torch.cuda.synchronize()
            log.info("TripoSG reloaded.")
        except Exception as e:
            log.warning("Could not reload TripoSG to GPU: %s", e)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ── Step 3: TripoSG mesh generation ──────────────────────────────────────
    log.info("Running TripoSG mesh generation …")
    mesh = run_triposg(
        pipe=pipe,
        image_input=image_path,
        rmbg_net=rmbg_net,
        seed=seed,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )

    mesh_path = os.path.join(output_dir, "generated_mesh.glb")
    mesh.export(mesh_path)
    log.info("Mesh exported: %s", mesh_path)

    # ── Step 4: Transfer 2D segmentation → 3D face labels ────────────────────
    log.info("Transferring segmentation to 3D mesh …")
    labels_result = transfer_segmentation_to_3d(
        glb_path=mesh_path,
        band_mask=seg["band_mask"],
        gemstone_mask=seg["gemstone_mask"],
        output_dir=output_dir,
    )
    log.info("3D face labels: %s", labels_result["label_counts"])

    return {
        "segmentation": os.path.join(output_dir, "seg_masks.png"),
        "mesh":         mesh_path,
        "labels":       labels_result["labels_path"],
        "seg_debug":    seg.get("debug_vis") or "",
        "seg_boxes":    seg.get("seg_boxes") or "",
        "seg_masks":    seg.get("seg_masks") or "",
    }
