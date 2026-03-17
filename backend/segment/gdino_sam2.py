# JewelForge v2
import torch
import numpy as np
from PIL import Image
from .jewelry_prompts import JEWELRY_PROMPTS


class JewelrySegmenter:
    def __init__(self, device="cuda"):
        self.device = device
        self.gdino_model = None
        self.sam2_predictor = None

    def load_models(self):
        import sys
        sys.path.insert(0, "models/GroundingDINO")
        from groundingdino.util.inference import load_model
        self.gdino_model = load_model(
            "weights/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth",
            device=self.device,
        )
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2 = build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            "weights/sam2.1_hiera_small.pt",
            device=self.device,
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2)

    def unload_models(self):
        del self.gdino_model, self.sam2_predictor
        self.gdino_model = self.sam2_predictor = None
        torch.cuda.empty_cache()

    def segment(
        self,
        image_path,
        jewelry_type="ring",
        text_prompt_override=None,
        box_threshold=0.25,
        text_threshold=0.2,
    ):
        import sys
        sys.path.insert(0, "models/GroundingDINO")
        from groundingdino.util.inference import predict, load_image as gdino_load_image

        prompts = text_prompt_override or (
            " . ".join(JEWELRY_PROMPTS.get(jewelry_type, JEWELRY_PROMPTS["ring"])[0]) + " ."
        )

        image_source, image_tensor = gdino_load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.gdino_model,
            image=image_tensor,
            caption=prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if len(boxes) == 0:
            return []

        h, w = image_source.shape[:2]
        boxes_px = boxes.cpu().numpy()
        boxes_px[:, [0, 2]] *= w
        boxes_px[:, [1, 3]] *= h

        self.sam2_predictor.set_image(image_source)
        segments = []
        for i, (box, logit, phrase) in enumerate(zip(boxes_px, logits, phrases)):
            masks, scores, _ = self.sam2_predictor.predict(
                box=box.reshape(1, 4),
                multimask_output=False,
            )
            segments.append({
                "label": phrase.strip(),
                "bbox": box.tolist(),
                "confidence": float(logit.item()),
                "mask": masks[0],
                "area_fraction": float(masks[0].sum() / (h * w)),
                "segment_id": i,
            })
        return segments
