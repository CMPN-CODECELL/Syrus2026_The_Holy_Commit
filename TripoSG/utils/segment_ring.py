"""
Ring component segmentation: produces binary band and gemstone masks.
 
PRIMARY — Grounding DINO + SAM 2
  Mirrors the PROVEN working logic from test_segmentation.py exactly.
  Uses IDEA-Research/grounding-dino-tiny → facebook/sam2-hiera-small.
 
  Key differences from previous broken version:
  - ALL detections go to SAM (no pre-classification that silently drops detections)
  - Labels are classified AFTER masking, using fuzzy substring matching
  - Full tracebacks are always logged (no silent swallowing)
 
FALLBACK — HSV + RMBG-1.4
  Used when DINO/SAM imports fail or any exception occurs.
"""
 
import cv2
import logging
import os
import traceback

import numpy as np

from utils.visualize_masks import get_rmbg_mask
from constants import JEWELRY_LABELS, DINO_THRESHOLD

log = logging.getLogger(__name__)

# ── Grounding DINO / SAM 2 config ────────────────────────────────────────────

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL_ID = "facebook/sam2-hiera-small"
 
 
# ── Public model-loading helpers ─────────────────────────────────────────────

def load_dino(device: str | None = None):
    """
    Load Grounding DINO. Returns (processor, model) or (None, None) on failure.
    Called at server startup to cache models across requests.
    """
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError as e:
        log.warning("[SEG] transformers not available (%s) — DINO disabled", e)
        return None, None

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        log.info("[SEG] Loading Grounding DINO (%s) on %s", DINO_MODEL_ID, device)
        processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
        model.eval()
        log.info("[SEG] Grounding DINO loaded OK")
        return processor, model
    except Exception:
        log.error("[SEG] Failed to load Grounding DINO:\n%s", traceback.format_exc())
        return None, None


def load_sam2(device: str | None = None):
    """
    Load SAM2 base model. Returns sam2_model or None on failure.
    Called at server startup to cache models across requests.
    A lightweight SAM2ImagePredictor is created per-request from this model.
    """
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        log.info("[SEG] Loading SAM 2 (%s) on %s", SAM2_MODEL_ID, device)
        try:
            from sam2.build_sam import build_sam2_hf
            sam2_model = build_sam2_hf(SAM2_MODEL_ID, device=device)
        except Exception:
            from sam2.build_sam import build_sam2
            sam2_model = build_sam2("sam2_hiera_s.yaml", "sam2_hiera_small.pt", device=device)
        log.info("[SEG] SAM 2 loaded OK")
        return sam2_model
    except Exception:
        log.error("[SEG] Failed to load SAM 2:\n%s", traceback.format_exc())
        return None


def _classify_label(raw_label: str) -> str:
    """
    Classify a DINO detection label as 'gem' or 'band'.
    Uses substring matching to handle periods, whitespace, and partial matches
    that the transformers post-processor may return.
    """
    label = raw_label.strip().rstrip(".").lower()
    if any(kw in label for kw in ("gem", "diamond", "stone", "crystal", "jewel")):
        return "gem"
    if any(kw in label for kw in ("band", "metal", "ring", "prong", "setting", "shank")):
        return "band"
    return "unknown"

def _save_masks_vis(image_pil, masks, output_path) -> None:
    """
    Save SAM2 mask visualization.
    Masks half of test/test_segmentation.py::visualize_and_save(), copied directly.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image_np = np.array(image_pil.convert("RGB"))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np)

    overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
    mask_colors = plt.cm.tab10(np.linspace(0, 1, max(len(masks), 1)))

    for i, m in enumerate(masks):
        color = mask_colors[i % len(mask_colors)]
        mask_bool = m["mask"].astype(bool)
        mask_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
        mask_rgba[mask_bool] = [*color[:3], 0.45]
        overlay = np.maximum(overlay, mask_rgba)

        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            cx, cy = xs.mean(), ys.mean()
            ax.text(cx, cy, m["label"], color='white', fontsize=9,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color[:3], alpha=0.8))

    ax.imshow(overlay)
    ax.set_title(f"SAM 2 Masks: {len(masks)} segments")
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info("[SEG] Saved masks vis: %s", output_path)


def save_segmentation_debug_images(image_pil, detections, masks_out, output_dir) -> dict:
    """
    Save DINO bounding-box and SAM2 mask visualizations, mirroring the
    output of test/test_segmentation.py's visualize_and_save() function.

    Returns:
        {"seg_boxes": "<path>/seg_boxes.png", "seg_masks": "<path>/seg_masks.png"}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image_np = np.array(image_pil.convert("RGB"))

    boxes_path = os.path.join(output_dir, "seg_boxes.png")
    masks_path = os.path.join(output_dir, "seg_masks.png")

    # ── Bounding boxes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(detections), 1)))

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        color = colors[i % len(colors)]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{det['label']} ({det['score']:.2f})",
                color='white', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

    ax.set_title(f"Grounding DINO: {len(detections)} detections")
    ax.axis('off')
    fig.savefig(boxes_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info("[SEG] Saved %s", boxes_path)

    # ── SAM2 masks ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np)

    overlay    = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
    mask_colors = plt.cm.tab10(np.linspace(0, 1, max(len(masks_out), 1)))

    for i, m in enumerate(masks_out):
        color     = mask_colors[i % len(mask_colors)]
        mask_bool = m["mask"].astype(bool)
        mask_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
        mask_rgba[mask_bool] = [*color[:3], 0.45]
        overlay = np.maximum(overlay, mask_rgba)

        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            cx, cy = xs.mean(), ys.mean()
            ax.text(cx, cy, m["label"], color='white', fontsize=9,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color[:3], alpha=0.8))

    ax.imshow(overlay)
    ax.set_title(f"SAM2 masks: {len(masks_out)} segments")
    ax.axis('off')
    fig.savefig(masks_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info("[SEG] Saved %s", masks_path)

    return {"seg_boxes": boxes_path, "seg_masks": masks_path}
 
 
# ── Public entry point ────────────────────────────────────────────────────────

def segment_ring_components(
    image_path: str,
    rmbg_net,
    output_dir: str | None = None,
    dino_processor=None,
    dino_model=None,
    sam2_model=None,
) -> dict:
    """
    Segment a ring image into band and gemstone binary masks.
    Tries Grounding DINO + SAM 2 first; falls back to HSV if unavailable.

    Args:
        dino_processor: Pre-loaded DINO processor (from load_dino()). If None, loads locally.
        dino_model:     Pre-loaded DINO model (from load_dino()). If None, loads locally.
        sam2_model:     Pre-loaded SAM2 base model (from load_sam2()). If None, loads locally.
    """
    log.info("[SEG] === segment_ring_components: %s", image_path)

    result = _segment_dino_sam(
        image_path, output_dir,
        dino_processor=dino_processor,
        dino_model=dino_model,
        sam2_model=sam2_model,
    )
    if result is not None:
        return result

    log.info("[SEG] Primary (DINO+SAM) unavailable or insufficient — running HSV fallback")
    return _segment_hsv(image_path, rmbg_net, output_dir)
 
 
# ── Primary: Grounding DINO + SAM 2 ──────────────────────────────────────────

def _segment_dino_sam(
    image_path: str,
    output_dir,
    dino_processor=None,
    dino_model=None,
    sam2_model=None,
) -> dict | None:
    """
    Run Grounding DINO → SAM 2 segmentation.
    Returns result dict on success, None to trigger HSV fallback.

    If dino_processor/dino_model/sam2_model are provided (pre-loaded at server startup),
    they are used directly and NOT deleted after use. If None, models are loaded locally
    and freed when done (original per-request behaviour, used in standalone test scripts).
    """
    _models_provided = dino_processor is not None and dino_model is not None and sam2_model is not None

    try:
        import torch
        from PIL import Image
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        log.warning("[SEG] Required import not available (%s) — skipping DINO+SAM", e)
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load image ───────────────────────────────────────────────────────────
    try:
        image_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.warning("[SEG] Cannot open image: %s", e)
        return None

    img_w, img_h = image_pil.size
    log.info("[SEG] DINO input: %dx%d  path=%s", img_w, img_h, image_path)

    # ── Step 1: Grounding DINO ───────────────────────────────────────────────
    processor   = dino_processor
    gdino_model = dino_model
    if processor is None or gdino_model is None:
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            log.info("[SEG] Loading Grounding DINO (%s) on %s", DINO_MODEL_ID, device)
            processor   = AutoProcessor.from_pretrained(DINO_MODEL_ID)
            gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_MODEL_ID
            ).to(device)
            gdino_model.eval()
            log.info("[SEG] Grounding DINO loaded OK")
        except Exception:
            log.error("[SEG] Failed to load Grounding DINO:\n%s", traceback.format_exc())
            return None
    else:
        # Pre-loaded model may be on CPU (kept there to save VRAM). Move to CUDA for inference.
        if torch.cuda.is_available() and next(gdino_model.parameters()).device.type != "cuda":
            log.info("[SEG] Moving pre-loaded DINO to CUDA for inference")
            gdino_model = gdino_model.to(device)
        log.info("[SEG] Using pre-loaded Grounding DINO")

    text_prompt = " ".join(JEWELRY_LABELS)
    log.info("[SEG] DINO prompt: %r  threshold=%.2f", text_prompt, DINO_THRESHOLD)

    try:
        inputs = processor(
            images=image_pil, text=text_prompt, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = gdino_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=DINO_THRESHOLD,
            text_threshold=DINO_THRESHOLD,
            target_sizes=[(image_pil.height, image_pil.width)],
        )
    except Exception:
        log.error("[SEG] DINO inference failed:\n%s", traceback.format_exc())
        if not _models_provided:
            try:
                del gdino_model, processor
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return None

    # Parse detections — move all values to CPU immediately so CUDA memory can be freed
    detections = []
    result = results[0]
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        detections.append({
            "label": label,
            "box":   box.cpu().numpy().tolist(),
            "score": round(float(score), 3),
        })
    detections.sort(key=lambda d: d["score"], reverse=True)

    # Always free DINO inference intermediates — results are already parsed to CPU lists above.
    # This is critical when using pre-loaded models: without it, inputs/outputs hold CUDA memory
    # during the entire SAM2 + TripoSG inference that follows.
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    log.info("[SEG] DINO found %d detections:", len(detections))
    for d in detections:
        cls = _classify_label(d["label"])
        log.info("[SEG]   label=%r → classified=%s  score=%.3f  box=%s",
                 d["label"], cls, d["score"], [round(c, 1) for c in d["box"]])

    if len(detections) == 0:
        log.warning("[SEG] DINO found 0 detections — falling back to HSV")
        if not _models_provided:
            del gdino_model, processor
        return None

    # Free locally-loaded DINO model (pre-loaded models are managed by run_pipeline.py)
    if not _models_provided:
        del gdino_model, processor
        log.info("[SEG] Freed locally-loaded Grounding DINO")

    # ── Step 2: SAM 2 ───────────────────────────────────────────────────────
    # Pass ALL detections to SAM.
    _local_sam2 = sam2_model is None
    if _local_sam2:
        try:
            try:
                from sam2.build_sam import build_sam2_hf
                sam2_model = build_sam2_hf(SAM2_MODEL_ID, device=device)
            except Exception:
                from sam2.build_sam import build_sam2
                sam2_model = build_sam2("sam2_hiera_s.yaml", "sam2_hiera_small.pt", device=device)
            log.info("[SEG] SAM 2 loaded OK")
        except Exception:
            log.error("[SEG] Failed to load SAM 2:\n%s", traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    else:
        log.info("[SEG] Using pre-loaded SAM 2")

    # If the pre-loaded SAM2 model is on CPU, move to CUDA for inference
    if _models_provided and torch.cuda.is_available():
        _sam2_device = next(iter(sam2_model.parameters())).device.type if hasattr(sam2_model, "parameters") else "cpu"
        if _sam2_device != "cuda":
            log.info("[SEG] Moving pre-loaded SAM 2 to CUDA for inference")
            sam2_model.to(device)

    # Create a fresh predictor per-request (set_image is stateful)
    predictor = SAM2ImagePredictor(sam2_model)

    image_np = np.array(image_pil)
    predictor.set_image(image_np)

    # Get a mask for EVERY detection
    all_masks = []
    for det in detections:
        box = np.array(det["box"]).reshape(1, 4)
        try:
            masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0]
            all_masks.append({
                "mask": mask,
                "label": det["label"],
                "det_score": det["score"],
                "sam_score": round(float(scores[0]), 3),
                "classification": _classify_label(det["label"]),
            })
            log.info("[SEG]   SAM mask for %r (%s): %d px  sam_score=%.3f",
                     det["label"], _classify_label(det["label"]),
                     int(mask.sum()), float(scores[0]))
        except Exception as e:
            log.warning("[SEG]   SAM prediction failed for %r: %s", det["label"], e)

    # Free predictor (lightweight wrapper); free base model only if loaded locally
    del predictor
    if _local_sam2:
        del sam2_model
        log.info("[SEG] Freed locally-loaded SAM 2")
    # Always synchronize after SAM2 inference to ensure all GPU work is complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Save DINO+SAM2 visualizations using test_segmentation.py::visualize_and_save
    for m in all_masks:
        m["label"] = "gemstone" if m["classification"] == "gem" else "band"
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../test'))
    from test_segmentation import visualize_and_save
    visualize_and_save(image_pil, detections, all_masks, output_prefix=os.path.join(output_dir, "seg"))

    # ── Step 3: Combine masks by classification ──────────────────────────────
    # NOW classify the masks (after SAM, not before)
    gem_mask  = np.zeros((img_h, img_w), dtype=np.uint8)
    band_mask = np.zeros((img_h, img_w), dtype=np.uint8)
 
    for m in all_masks:
        mask_bool = m["mask"].astype(bool)
        if m["classification"] == "gem":
            gem_mask[mask_bool] = 255
        elif m["classification"] == "band":
            band_mask[mask_bool] = 255
        else:
            # Unknown labels → assign to band (safer default)
            band_mask[mask_bool] = 255
            log.info("[SEG]   Unknown label %r assigned to band", m["label"])
 
    # Gemstone takes priority in overlapping regions
    band_mask = cv2.bitwise_and(band_mask, cv2.bitwise_not(gem_mask))
 
    gem_px  = int(np.count_nonzero(gem_mask))
    band_px = int(np.count_nonzero(band_mask))
    log.info("[SEG] Final masks — band: %d px, gemstone: %d px", band_px, gem_px)
 
    # Check we got both components
    if gem_px == 0:
        log.warning("[SEG] DINO+SAM produced 0 gemstone pixels — falling back to HSV")
        return None
 
    if band_px == 0:
        log.warning("[SEG] DINO+SAM produced 0 band pixels — falling back to HSV")
        return None
 
    # ── Step 4: Build result ─────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
 
    # Resize masks if image dimensions differ
    h_bgr, w_bgr = img_bgr.shape[:2]
    if band_mask.shape != (h_bgr, w_bgr):
        band_mask = cv2.resize(band_mask, (w_bgr, h_bgr), interpolation=cv2.INTER_NEAREST)
        gem_mask  = cv2.resize(gem_mask,  (w_bgr, h_bgr), interpolation=cv2.INTER_NEAREST)
 
    components = ["band", "gemstone"]
    log.info("[SEG] DINO+SAM SUCCESS — components: %s", components)

    # Save DINO boxes + SAM2 masks visualizations (test_segmentation.py style)
    vis_paths = {}
    if output_dir:
        try:
            vis_paths = save_segmentation_debug_images(
                image_pil, detections, all_masks, output_dir
            )
        except Exception:
            log.warning("[SEG] Could not save debug images:\n%s", traceback.format_exc())

    result = _build_result(band_mask, gem_mask, components, img_bgr, output_dir)
    result["seg_boxes"] = vis_paths.get("seg_boxes")
    result["seg_masks"] = vis_paths.get("seg_masks")
    return result
 
 
# ── Fallback: HSV + RMBG-1.4 ─────────────────────────────────────────────────
 
def _segment_hsv(image_path: str, rmbg_net, output_dir) -> dict:
    """
    Original HSV-based segmentation (RMBG foreground + explicit metal ranges).
    Preserved as fallback when DINO+SAM is unavailable.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"[SEG] Could not read image: {image_path}")
    h, w = img_bgr.shape[:2]
 
    log.info("[SEG][HSV] Input image: shape=(%d, %d, %d), dtype=%s, path=%s",
             h, w, img_bgr.shape[2], img_bgr.dtype, image_path)
 
    # ── Step 1: RMBG foreground mask ─────────────────────────────────────────
    fg_mask_raw = get_rmbg_mask(image_path, rmbg_net)
    fg_mask = cv2.resize(fg_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
 
    fg_pixels = int(np.count_nonzero(fg_mask))
    log.info("[SEG][HSV] Foreground pixels (RMBG): %d  (%.1f%% of %dx%d image)",
             fg_pixels, 100.0 * fg_pixels / max(w * h, 1), w, h)
 
    if fg_pixels == 0:
        log.warning("[SEG][HSV] RMBG found no foreground — returning full image as band")
        band_mask = np.full((h, w), 255, dtype=np.uint8)
        gem_mask  = np.zeros((h, w), dtype=np.uint8)
        return _build_result(band_mask, gem_mask, ["band"], img_bgr, output_dir)
 
    # ── Step 2: Explicit HSV metal detection ─────────────────────────────────
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
 
    gold_mask   = cv2.inRange(img_hsv, (12,  60,  80), (42, 255, 255))
    silver_mask = cv2.inRange(img_hsv, ( 0,   0, 110), (180, 54, 255))
    rosegold_lo = cv2.inRange(img_hsv, ( 0,  40, 100), (12, 200, 255))
    rosegold_hi = cv2.inRange(img_hsv, (165, 40, 100), (180, 200, 255))
    rose_mask   = cv2.bitwise_or(rosegold_lo, rosegold_hi)
 
    metal_raw = cv2.bitwise_or(gold_mask, silver_mask)
    metal_raw = cv2.bitwise_or(metal_raw, rose_mask)
    metal_raw = cv2.bitwise_and(metal_raw, fg_mask)
 
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    metal_mask = cv2.morphologyEx(metal_raw, cv2.MORPH_CLOSE, k7)
    metal_mask = cv2.bitwise_and(metal_mask, fg_mask)
 
    gem_candidate = cv2.bitwise_and(fg_mask, cv2.bitwise_not(metal_mask))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gem_candidate = cv2.morphologyEx(gem_candidate, cv2.MORPH_OPEN,  k5)
    gem_candidate = cv2.morphologyEx(gem_candidate, cv2.MORPH_CLOSE, k5)
    gem_candidate = cv2.bitwise_and(gem_candidate, fg_mask)
 
    gold_px   = int(np.count_nonzero(gold_mask   & fg_mask))
    silver_px = int(np.count_nonzero(silver_mask & fg_mask))
    metal_px  = int(np.count_nonzero(metal_mask))
    gem_px    = int(np.count_nonzero(gem_candidate))
 
    log.info("[SEG][HSV] HSV metal detection:")
    log.info("[SEG][HSV]   gold_mask   : %6d px  (H=12-42, S>=60, V>=80)", gold_px)
    log.info("[SEG][HSV]   silver_mask : %6d px  (any H, S<55, V>=110)", silver_px)
    log.info("[SEG][HSV]   rose_mask   : %6d px  (H=0-12 or 165-180, S>=40)",
             int(np.count_nonzero(rose_mask & fg_mask)))
    log.info("[SEG][HSV]   metal_mask  : %6d px  (%.1f%% of fg)",
             metal_px, 100.0 * metal_px / max(fg_pixels, 1))
    log.info("[SEG][HSV]   gem (Stage1): %6d px  (%.1f%% of fg)",
             gem_px, 100.0 * gem_px / max(fg_pixels, 1))
 
    # ── Step 3: Spatial fallback if Stage 1 gem < 5% of fg ───────────────────
    if gem_px < fg_pixels * 0.05:
        log.info("[SEG][HSV] Stage-1 gemstone < 5%% of fg → spatial fallback")
 
        ys, xs = np.where(fg_mask > 0)
        top_y    = float(np.percentile(ys, 25))
        x_min_fg = int(xs.min())
        x_max_fg = int(xs.max())
        x_ctr    = (x_min_fg + x_max_fg) // 2
        x_margin = int(0.25 * (x_max_fg - x_min_fg))
 
        in_top    = ys < top_y
        in_centre = np.abs(xs.astype(int) - x_ctr) <= x_margin
        in_gem    = in_top & in_centre
 
        gem_spatial = np.zeros_like(fg_mask)
        if in_gem.any():
            gem_spatial[ys[in_gem], xs[in_gem]] = 255
            gem_spatial = cv2.bitwise_and(gem_spatial, fg_mask)
 
        sp_pixels = int(np.count_nonzero(gem_spatial))
        log.info("[SEG][HSV] Spatial candidate: %d px", sp_pixels)
 
        if sp_pixels > 100:
            gem_candidate = gem_spatial
        else:
            top_y20  = float(np.percentile(ys, 20))
            in_top20 = ys < top_y20
            gem_top  = np.zeros_like(fg_mask)
            if in_top20.any():
                gem_top[ys[in_top20], xs[in_top20]] = 255
            gem_candidate = cv2.bitwise_and(gem_top, fg_mask)
 
        gem_px = int(np.count_nonzero(gem_candidate))
        log.info("[SEG][HSV] After spatial fallback: gemstone=%d px", gem_px)
 
    # ── Step 4: Build final masks ─────────────────────────────────────────────
    gem_mask  = gem_candidate
    band_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(gem_mask))
 
    components = []
    if np.any(band_mask > 0): components.append("band")
    if np.any(gem_mask  > 0): components.append("gemstone")
    if not components:        components = ["band"]
 
    log.info("[SEG][HSV] Final: %s", components)
    return _build_result(band_mask, gem_mask, components, img_bgr, output_dir)
 
 
# ── Internal helpers ──────────────────────────────────────────────────────────
 
def _build_result(band_mask, gem_mask, components, img_bgr, output_dir, debug_path=None):
    if output_dir and debug_path is None:
        debug_path = _save_debug_vis(img_bgr, band_mask, gem_mask, output_dir)
    return {
        "band_mask":     band_mask,
        "gemstone_mask": gem_mask,
        "components":    components,
        "debug_vis":     debug_path,
    }
 
 
def _save_debug_vis(img_bgr, band_mask, gem_mask, output_dir) -> str:
    vis = img_bgr.copy()
 
    if np.any(band_mask > 0):
        blue = np.array([180, 80, 20], dtype=np.float32)
        px = band_mask > 0
        vis[px] = (vis[px].astype(np.float32) * 0.55 + blue * 0.45).clip(0, 255).astype(np.uint8)
 
    if np.any(gem_mask > 0):
        green = np.array([30, 210, 80], dtype=np.float32)
        px = gem_mask > 0
        vis[px] = (vis[px].astype(np.float32) * 0.55 + green * 0.45).clip(0, 255).astype(np.uint8)
 
    for arr, colour in [(band_mask, (180, 80, 20)), (gem_mask, (30, 210, 80))]:
        cnts, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, colour, 2)
 
    cv2.putText(vis, "Band",     (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 80,  20), 2)
    cv2.putText(vis, "Gemstone", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,  210, 80), 2)
 
    path = os.path.join(output_dir, "seg_debug.png")
    cv2.imwrite(path, vis)
    log.info("[SEG] Debug vis saved: %s", path)
    return path