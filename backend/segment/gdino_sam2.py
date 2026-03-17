"""
Jewelry segmentation using Grounding DINO + SAM 2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from segment.jewelry_prompts import get_prompts_for_type

# Resolve paths relative to the backend package root
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_MODELS_DIR = _BACKEND_DIR / "models"
_WEIGHTS_DIR = _BACKEND_DIR / "weights"

GDINO_CONFIG = _MODELS_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = _WEIGHTS_DIR / "groundingdino_swint_ogc.pth"
SAM2_WEIGHTS = _WEIGHTS_DIR / "sam2.1_hiera_small.pt"
SAM2_CONFIG = "sam2.1_hiera_s.yaml"


class JewelrySegmenter:
    """
    Wraps Grounding DINO + SAM 2 for per-component jewelry segmentation.

    Usage:
        segmenter = JewelrySegmenter(device="cuda")
        segmenter.load_models()
        segments = segmenter.segment("ring.png", jewelry_type="ring")
        segmenter.unload_models()
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._gdino_model = None
        self._sam2_predictor = None

    # ── Model lifecycle ────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load Grounding DINO and SAM 2 into memory."""
        self._load_gdino()
        self._load_sam2()

    def unload_models(self) -> None:
        """Release GPU memory held by the segmentation models."""
        import gc
        self._gdino_model = None
        self._sam2_predictor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Public API ─────────────────────────────────────────────────────────

    def segment(
        self,
        image_path: str,
        jewelry_type: str = "ring",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> List[dict]:
        """
        Detect and segment jewelry components in an image.

        Args:
            image_path: Path to the preprocessed jewelry image (RGB/RGBA PNG).
            jewelry_type: "ring" | "necklace" | "earring" | "bracelet" | "auto".
            box_threshold: GDINO confidence threshold for bounding boxes.
            text_threshold: GDINO token confidence threshold.

        Returns:
            List of segment dicts, each containing:
                label          – component name (str)
                bbox           – [x0, y0, x1, y1] in pixel coords
                confidence     – float in [0, 1]
                mask           – np.ndarray of shape (H, W), dtype bool
                area_fraction  – fraction of image covered by this mask
                segment_id     – unique integer index
        """
        if self._gdino_model is None or self._sam2_predictor is None:
            raise RuntimeError("Call load_models() before segment().")

        from PIL import Image as PILImage

        image = PILImage.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        prompts = get_prompts_for_type(jewelry_type)
        text_prompt = " . ".join(prompts) + " ."

        # ── Grounding DINO detection ───────────────────────────────────────
        boxes, scores, labels = self._run_gdino(
            image, text_prompt, box_threshold, text_threshold
        )

        if len(boxes) == 0:
            return []

        # ── SAM 2 mask prediction for each box ────────────────────────────
        self._sam2_predictor.set_image(image_np)

        segments: List[dict] = []
        for seg_id, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x0, y0, x1, y1 = [int(v) for v in box]
            masks, _, _ = self._sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([[x0, y0, x1, y1]]),
                multimask_output=False,
            )
            mask: np.ndarray = masks[0].astype(bool)  # (H, W)
            area_fraction = float(mask.sum()) / (h * w)

            segments.append(
                {
                    "label": label,
                    "bbox": [x0, y0, x1, y1],
                    "confidence": float(score),
                    "mask": mask,
                    "area_fraction": area_fraction,
                    "segment_id": seg_id,
                }
            )

        return segments

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_gdino(self) -> None:
        try:
            import sys
            sys.path.insert(0, str(_MODELS_DIR / "GroundingDINO"))
            from groundingdino.util.inference import load_model
            self._gdino_model = load_model(
                str(GDINO_CONFIG), str(GDINO_WEIGHTS)
            )
            self._gdino_model.to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Grounding DINO: {exc}. "
                "Ensure models/GroundingDINO is cloned and weights exist."
            ) from exc

    def _load_sam2(self) -> None:
        try:
            import sys
            sys.path.insert(0, str(_MODELS_DIR / "sam2"))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                SAM2_CONFIG,
                str(SAM2_WEIGHTS),
                device=self.device,
            )
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SAM 2: {exc}. "
                "Ensure models/sam2 is cloned and weights exist."
            ) from exc

    def _run_gdino(
        self,
        image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ):
        """Run Grounding DINO and return (boxes, scores, labels) in pixel coords."""
        import torch
        from groundingdino.util.inference import predict
        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.Resize((800, 800)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            boxes_norm, logits, phrases = predict(
                model=self._gdino_model,
                image=img_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

        # Convert normalized [cx, cy, w, h] → pixel [x0, y0, x1, y1]
        img_w, img_h = image.size
        boxes_pixel = []
        for box in boxes_norm:
            cx, cy, bw, bh = box.tolist()
            x0 = (cx - bw / 2) * img_w
            y0 = (cy - bh / 2) * img_h
            x1 = (cx + bw / 2) * img_w
            y1 = (cy + bh / 2) * img_h
            boxes_pixel.append([x0, y0, x1, y1])

        scores = logits.tolist()
        return boxes_pixel, scores, phrases
