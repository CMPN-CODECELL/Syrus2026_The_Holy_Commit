"""
test_segmentation.py — Test Grounding DINO + SAM 2 on a jewelry image.

Usage:
    python test_segmentation.py my_ring.png
    python test_segmentation.py my_ring.png --threshold 0.2

Output:
    - output_boxes.png   → image with bounding boxes from Grounding DINO
    - output_masks.png   → image with colored segmentation masks from SAM 2
    - output_labels.json → detected components with bounding boxes and scores
"""

import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────
# These are the text prompts Grounding DINO will search for.
# Tweak these based on what works for your jewelry images.
# IMPORTANT: each label must be lowercase and end with a period.
JEWELRY_LABELS = [
    "gemstone.",
    "diamond.",
    "metal band.",
    "ring band.",
    "prong.",
    "setting.",
    "stone.",
]


def load_grounding_dino(device):
    """Load Grounding DINO from HuggingFace. No CUDA compilation needed."""
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError as e:
        print(f"Failed to import transformers: {e}")
        print("Please install compatible versions of transformers and torchvision.")
        return None, None

    print("[1/4] Loading Grounding DINO...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    print(f"       Loaded on {device} | ~340MB VRAM")
    return processor, model


def detect_components(image, processor, model, device, threshold=0.25):
    """
    Run Grounding DINO to find jewelry components.
    Returns list of {label, box [x1,y1,x2,y2], score}.
    """
    print("[2/4] Detecting jewelry components...")

    # Grounding DINO (transformers 5.x) expects a single merged string, e.g. "gem. band."
    text_labels = " ".join(JEWELRY_LABELS)

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[(image.height, image.width)],
    )

    detections = []
    result = results[0]
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        box = box.cpu().numpy().tolist()
        detections.append({
            "label": label,
            "box": [round(c, 1) for c in box],  # [x1, y1, x2, y2]
            "score": round(score.item(), 3),
        })

    # Sort by score descending
    detections.sort(key=lambda d: d["score"], reverse=True)

    print(f"       Found {len(detections)} components:")
    for d in detections:
        print(f"         {d['label']:20s} score={d['score']:.3f}  box={d['box']}")

    return detections


def segment_with_sam2(image, detections, device):
    """
    Take Grounding DINO boxes → SAM 2 masks.
    Returns list of (mask, label, score) tuples.
    """
    print("[3/4] Running SAM 2 segmentation...")

    # Try HuggingFace-based loading first, fall back to config-based
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = build_sam2_hf("facebook/sam2-hiera-small", device=device)
    except Exception:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        # If the HF loading fails, try direct checkpoint
        sam2_model = build_sam2("sam2_hiera_s.yaml", "sam2_hiera_small.pt", device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    image_np = np.array(image.convert("RGB"))
    predictor.set_image(image_np)

    masks_out = []
    for det in detections:
        box = np.array(det["box"]).reshape(1, 4)  # SAM 2 expects (1, 4)

        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False,  # Single best mask per box
        )

        masks_out.append({
            "mask": masks[0],        # (H, W) boolean array
            "label": det["label"],
            "det_score": det["score"],
            "sam_score": round(scores[0].item(), 3),
        })
        print(f"         {det['label']:20s} → mask pixels: {masks[0].sum():,}")

    return masks_out


def visualize_and_save(image, detections, masks, output_prefix="output"):
    """Save visualization images showing boxes and masks."""
    print("[4/4] Saving visualizations...")
    image_np = np.array(image.convert("RGB"))
    h, w = image_np.shape[:2]

    # --- Bounding box visualization ---
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
    fig.savefig(f"{output_prefix}_boxes.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"       Saved {output_prefix}_boxes.png")

    # --- Mask visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np)

    # Create composite overlay
    overlay = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
    mask_colors = plt.cm.tab10(np.linspace(0, 1, max(len(masks), 1)))

    for i, m in enumerate(masks):
        color = mask_colors[i % len(mask_colors)]
        mask_bool = m["mask"].astype(bool)
        mask_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
        mask_rgba[mask_bool] = [*color[:3], 0.45]
        overlay = np.maximum(overlay, mask_rgba)  # Layer masks

        # Label at mask centroid
        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            cx, cy = xs.mean(), ys.mean()
            ax.text(cx, cy, m["label"], color='white', fontsize=9,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color[:3], alpha=0.8))

    ax.imshow(overlay)
    ax.set_title(f"SAM 2 Masks: {len(masks)} segments")
    ax.axis('off')
    fig.savefig(f"{output_prefix}_masks.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"       Saved {output_prefix}_masks.png")

    # --- Save JSON labels ---
    labels_out = []
    for i, m in enumerate(masks):
        mask_bool = m["mask"].astype(bool)
        labels_out.append({
            "label": m["label"],
            "det_score": m["det_score"],
            "sam_score": m["sam_score"],
            "mask_pixels": int(mask_bool.sum()),
            "mask_percent": round(float(100 * mask_bool.sum() / (h * w)), 2),
        })

    with open(f"{output_prefix}_labels.json", "w") as f:
        json.dump(labels_out, f, indent=2)
    print(f"       Saved {output_prefix}_labels.json")


def main():
    parser = argparse.ArgumentParser(description="Test Grounding DINO + SAM 2 on jewelry image")
    parser.add_argument("image", help="Path to jewelry image (PNG/JPG)")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Detection confidence threshold (lower = more detections, default 0.25)")
    parser.add_argument("--output", default="output",
                        help="Output file prefix (default: 'output')")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Custom detection labels (override defaults). Each must end with a period.")
    args = parser.parse_args()

    # Allow custom labels
    global JEWELRY_LABELS
    if args.labels:
        JEWELRY_LABELS = [l if l.endswith(".") else l + "." for l in args.labels]
        print(f"Using custom labels: {JEWELRY_LABELS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({image.width}x{image.height})")

    # --- Step 1: Grounding DINO detection ---
    processor, gdino_model = load_grounding_dino(device)
    detections = detect_components(image, processor, gdino_model, device, args.threshold)

    if len(detections) == 0:
        print("\n⚠️  No components detected! Try:")
        print("   1. Lower the threshold: --threshold 0.15")
        print("   2. Try different labels: --labels 'gem.' 'gold band.' 'jewel.'")
        print("   3. Check if the image has a clean background")

        # Still save the boxes image (empty) for reference
        visualize_and_save(image, detections, [], args.output)
        return

    # Free Grounding DINO VRAM before loading SAM 2
    del gdino_model, processor
    torch.cuda.empty_cache()
    print("       (Freed Grounding DINO VRAM)")

    # --- Step 2: SAM 2 segmentation ---
    masks = segment_with_sam2(image, detections, device)

    # Free SAM 2 VRAM
    torch.cuda.empty_cache()

    # --- Step 3: Visualize ---
    visualize_and_save(image, detections, masks, args.output)

    print("\n✅ Done! Check output files:")
    print(f"   {args.output}_boxes.png  — bounding box detections")
    print(f"   {args.output}_masks.png  — segmentation masks")
    print(f"   {args.output}_labels.json — component data")
    print(f"\nVRAM peak: ~500MB (Grounding DINO ~340MB + SAM 2 ~150MB, loaded sequentially)")


if __name__ == "__main__":
    main()