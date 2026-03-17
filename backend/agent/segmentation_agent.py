# JewelForge v2
import numpy as np
from segment.jewelry_prompts import JEWELRY_PROMPTS


class SegmentationAgent:
    """Autonomous segmentation with retry tiers + color fallback."""

    MIN_SEGMENTS = {"ring": 2, "necklace": 2, "earring": 1, "bracelet": 1}

    def __init__(self, segmenter):
        self.segmenter = segmenter

    def segment_with_retries(self, image_path, jewelry_type="ring"):
        tiers = JEWELRY_PROMPTS.get(jewelry_type, JEWELRY_PROMPTS["ring"])
        min_segs = self.MIN_SEGMENTS.get(jewelry_type, 1)
        best_segments, warnings = [], []

        for tier_idx, prompts in enumerate(tiers):
            text_prompt = " . ".join(prompts) + " ."
            # Lower thresholds on retries
            bt = 0.25 if tier_idx == 0 else 0.18
            tt = 0.20 if tier_idx == 0 else 0.15

            segments = self.segmenter.segment(
                image_path,
                text_prompt_override=text_prompt,
                box_threshold=bt,
                text_threshold=tt,
            )

            if len(segments) >= min_segs:
                best_segments = segments
                break

            if len(segments) > len(best_segments):
                best_segments = segments
            warnings.append(f"Tier {tier_idx+1}: {len(segments)} segments (need {min_segs})")

        # Deduplicate overlapping segments (IoU > 0.7)
        best_segments = self._deduplicate(best_segments)

        # Color-based fallback if still insufficient
        if len(best_segments) < min_segs:
            color_segs = self._color_fallback(image_path)
            if len(color_segs) > len(best_segments):
                best_segments = color_segs
                warnings.append("Using color-based segmentation fallback")

        confidence = "high" if len(best_segments) >= min_segs else "low"
        return {
            "segments": best_segments,
            "confidence": confidence,
            "warnings": warnings,
            "attempts": len(tiers),
        }

    def _deduplicate(self, segments):
        if len(segments) <= 1:
            return segments
        kept = []
        for seg in sorted(segments, key=lambda s: s["confidence"], reverse=True):
            if not any(self._iou(seg["mask"], k["mask"]) > 0.7 for k in kept):
                kept.append(seg)
        return kept

    def _iou(self, a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / union if union > 0 else 0

    def _color_fallback(self, image_path):
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        segments = []

        # Gold metal
        mask = cv2.inRange(hsv, (15, 80, 100), (35, 255, 255))
        if mask.sum() > 0.01 * h * w:
            segments.append({
                "label": "metal",
                "mask": mask.astype(bool),
                "confidence": 0.5,
                "area_fraction": mask.sum() / (h * w),
                "segment_id": len(segments),
                "bbox": self._mask_bbox(mask),
            })

        # Gemstones (high saturation color ranges)
        color_ranges = [
            ("red_stone",   (0,   120, 80), (10,  255, 255)),
            ("blue_stone",  (100, 120, 80), (130, 255, 255)),
            ("green_stone", (35,  120, 80), (85,  255, 255)),
        ]
        for name, lo, hi in color_ranges:
            mask = cv2.inRange(hsv, lo, hi)
            if mask.sum() > 0.005 * h * w:
                segments.append({
                    "label": name,
                    "mask": mask.astype(bool),
                    "confidence": 0.3,
                    "area_fraction": mask.sum() / (h * w),
                    "segment_id": len(segments),
                    "bbox": self._mask_bbox(mask),
                })

        return segments

    def _mask_bbox(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return [0, 0, 0, 0]
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return [int(x1), int(y1), int(x2), int(y2)]
