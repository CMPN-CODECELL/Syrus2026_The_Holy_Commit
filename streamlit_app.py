import base64
import io
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image


DEFAULT_LABELS = [
    "gemstone.",
    "diamond.",
    "metal band.",
    "ring band.",
    "prong.",
    "setting.",
    "stone.",
]


@st.cache_resource
def load_grounding_dino(device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


@st.cache_resource
def load_sam2_predictor(device: str):
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2_hf("facebook/sam2-hiera-small", device=device)
        return SAM2ImagePredictor(sam2_model), "sam2"
    except Exception:
        # Keep app usable on low-storage setups by falling back to box masks.
        return None, "box"


def detect_components(image: Image.Image, processor, model, device: str, labels: List[str], threshold: float):
    text_labels = " ".join(labels)
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
        detections.append(
            {
                "label": label,
                "box": [round(float(c), 1) for c in box.cpu().numpy().tolist()],
                "score": round(float(score.item()), 3),
            }
        )

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def segment_with_sam2(image: Image.Image, detections: List[Dict[str, Any]], predictor):
    image_np = np.array(image.convert("RGB"))
    predictor.set_image(image_np)

    masks_out = []
    for det in detections:
        box = np.array(det["box"]).reshape(1, 4)
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        masks_out.append(
            {
                "mask": masks[0],
                "label": det["label"],
                "det_score": det["score"],
                "sam_score": round(float(scores[0].item()), 3),
            }
        )

    return masks_out


def segment_with_box_masks(image: Image.Image, detections: List[Dict[str, Any]]):
    h, w = image.height, image.width
    masks_out = []

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        masks_out.append(
            {
                "mask": mask,
                "label": det["label"],
                "det_score": det["score"],
                "sam_score": 0.0,
            }
        )

    return masks_out


def render_boxes_image(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_np)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(detections), 1)))

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        color = colors[i % len(colors)]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 5, 5),
            f"{det['label']} ({det['score']:.2f})",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    ax.set_title(f"Detections: {len(detections)}")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_masks_image(image: Image.Image, masks: List[Dict[str, Any]]) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
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
            ax.text(
                cx,
                cy,
                m["label"],
                color="white",
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color[:3], alpha=0.8),
            )

    ax.imshow(overlay)
    ax.set_title(f"Segments: {len(masks)}")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def labels_to_dataframe(masks: List[Dict[str, Any]], image_size: Tuple[int, int]) -> pd.DataFrame:
    w, h = image_size
    rows = []
    total_pixels = w * h

    for m in masks:
        mask_pixels = int(m["mask"].astype(bool).sum())
        rows.append(
            {
                "label": m["label"],
                "det_score": m["det_score"],
                "sam_score": m["sam_score"],
                "mask_pixels": mask_pixels,
                "mask_percent": round(100.0 * mask_pixels / total_pixels, 2),
            }
        )

    return pd.DataFrame(rows)


def run_triposg_command(command_template: str, input_path: Path, output_glb_path: Path):
    if not command_template.strip():
        return False, "TripoSG command is empty."

    command = command_template.format(input=str(input_path), output=str(output_glb_path))
    parts = shlex.split(command, posix=False)

    result = subprocess.run(parts, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown error"
        return False, f"TripoSG command failed: {stderr}"

    if not output_glb_path.exists():
        return False, "TripoSG command succeeded but output GLB was not found."

    return True, result.stdout.strip() or "TripoSG generation completed."


def render_glb_preview(glb_path: Path):
    data = glb_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    html = f"""
    <script type=\"module\" src=\"https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js\"></script>
    <model-viewer
      src=\"data:model/gltf-binary;base64,{b64}\"
      alt=\"Generated 3D model\"
      auto-rotate
      camera-controls
      style=\"width: 100%; height: 480px; background: #f6f7f9; border-radius: 12px;\">
    </model-viewer>
    """

    st.components.v1.html(html, height=500)


def main():
    st.set_page_config(page_title="TripoSG Jewelry Studio", layout="wide")
    st.title("TripoSG Jewelry Studio")
    st.caption("Upload a 2D jewelry image, segment components, and generate a GLB model.")

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Detection threshold", 0.05, 0.9, 0.25, 0.01)
        labels_raw = st.text_area(
            "Component labels (one per line)",
            value="\n".join(DEFAULT_LABELS),
            help="Each label should end with a period. Example: gemstone.",
        )
        triposg_command = st.text_input(
            "TripoSG command",
            value="",
            help="Use placeholders {input} and {output}, for example: python TripoSG/main.py --image {input} --output {output}",
        )

    upload = st.file_uploader("Upload 2D jewelry image", type=["png", "jpg", "jpeg", "webp"])

    if upload is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(upload).convert("RGB")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    labels = []
    for line in labels_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(line if line.endswith(".") else f"{line}.")

    if not labels:
        st.error("Please provide at least one label.")
        return

    if st.button("Run Segmentation + 3D Generation", type="primary"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Running on device: {device}")

        with st.spinner("Loading Grounding DINO..."):
            processor, gdino_model = load_grounding_dino(device)

        with st.spinner("Detecting jewelry components..."):
            detections = detect_components(image, processor, gdino_model, device, labels, threshold)

        if len(detections) == 0:
            st.warning("No components detected. Try a lower threshold or different labels.")
            return

        # Free detection model memory before SAM2
        del gdino_model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

        with st.spinner("Preparing segmentation masks..."):
            predictor, mode = load_sam2_predictor(device)
            if mode == "sam2":
                masks = segment_with_sam2(image, detections, predictor)
                st.info("Mask mode: SAM2 (pixel-level segmentation)")
            else:
                masks = segment_with_box_masks(image, detections)
                st.warning("Mask mode: Box fallback (SAM2 not installed). Install sam2 for finer masks.")

        boxes_img = render_boxes_image(image, detections)
        masks_img = render_masks_image(image, masks)

        b1, b2 = st.columns(2)
        with b1:
            st.subheader("Segmented Components (Boxes)")
            st.image(boxes_img, use_container_width=True)
        with b2:
            st.subheader("Segmented Components (Masks)")
            st.image(masks_img, use_container_width=True)

        labels_df = labels_to_dataframe(masks, image.size)
        st.subheader("Detected Components")
        st.dataframe(labels_df, use_container_width=True)

        labels_json_bytes = labels_df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            "Download labels JSON",
            data=labels_json_bytes,
            file_name="output_labels.json",
            mime="application/json",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input_image.png"
            output_glb_path = tmpdir_path / "output.glb"
            image.save(input_path)

            with st.spinner("Running TripoSG 2D -> 3D..."):
                success, msg = run_triposg_command(triposg_command, input_path, output_glb_path)

                # Fallback for demo repos that already contain a sample output.glb
                if not success:
                    fallback_glb = Path("output.glb")
                    if fallback_glb.exists():
                        output_glb_path.write_bytes(fallback_glb.read_bytes())
                        success = True
                        msg = (
                            "TripoSG command was not executed successfully, so a fallback sample GLB from the repo root "
                            "was used for preview/download."
                        )

            if success:
                st.success(msg)
                st.subheader("3D Rendered Output (GLB Preview)")
                render_glb_preview(output_glb_path)

                glb_bytes = output_glb_path.read_bytes()
                st.download_button(
                    "Download Generated GLB",
                    data=glb_bytes,
                    file_name="output.glb",
                    mime="model/gltf-binary",
                )
            else:
                st.error(msg)
                st.info(
                    "Set a valid TripoSG command in the sidebar with placeholders {input} and {output}. "
                    "Example: python TripoSG/main.py --image {input} --output {output}"
                )


if __name__ == "__main__":
    main()
