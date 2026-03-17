"""
Image preprocessing for JewelForge v2.
Removes background, centers, and resizes jewelry images.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def preprocess_image(
    input_path: str,
    output_dir: str,
    target_size: int = 512,
) -> dict:
    """
    Remove background, center-crop, and resize a jewelry image.

    Args:
        input_path: Path to the source image (PNG or JPG).
        output_dir: Directory where output images will be saved.
        target_size: Square output dimension in pixels (default 512).

    Returns:
        dict with keys:
            "rgb"  – path to RGB image on gray background (for TripoSG)
            "rgba" – path to RGBA image with transparent background (for segmentation)
    """
    try:
        from rembg import remove
    except ImportError:
        raise ImportError(
            "rembg is not installed. Run: pip install 'rembg[gpu]'"
        )

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_path).stem

    # ── Load and remove background ─────────────────────────────────────────
    with open(input_path, "rb") as fh:
        raw_bytes = fh.read()

    rgba_bytes = remove(raw_bytes)
    rgba_img: Image.Image = Image.open(__import__("io").BytesIO(rgba_bytes)).convert("RGBA")

    # ── Center-crop to the bounding box of the foreground ─────────────────
    rgba_img = _crop_to_subject(rgba_img)

    # ── Resize to target_size × target_size ───────────────────────────────
    rgba_resized: Image.Image = rgba_img.resize(
        (target_size, target_size), Image.LANCZOS
    )

    # ── Save RGBA version ─────────────────────────────────────────────────
    rgba_path = os.path.join(output_dir, f"{stem}_rgba.png")
    rgba_resized.save(rgba_path, format="PNG")

    # ── Composite onto mid-gray background for TripoSG ────────────────────
    gray_bg = Image.new("RGBA", (target_size, target_size), (127, 127, 127, 255))
    gray_bg.paste(rgba_resized, mask=rgba_resized.split()[3])
    rgb_img = gray_bg.convert("RGB")

    rgb_path = os.path.join(output_dir, f"{stem}_preprocessed.png")
    rgb_img.save(rgb_path, format="PNG")

    return {"rgb": rgb_path, "rgba": rgba_path}


def _crop_to_subject(rgba_img: Image.Image) -> Image.Image:
    """
    Crop tightly around the non-transparent pixels and add a small margin,
    then return a square crop.
    """
    alpha = np.array(rgba_img.split()[3])
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any():
        return rgba_img  # fully transparent – return as-is

    row_min, row_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    col_min, col_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

    # Expand to square
    h = row_max - row_min
    w = col_max - col_min
    side = max(h, w)
    margin = int(side * 0.05)  # 5 % padding
    side += margin * 2

    cx = (col_min + col_max) // 2
    cy = (row_min + row_max) // 2

    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(rgba_img.width, x0 + side)
    y1 = min(rgba_img.height, y0 + side)

    return rgba_img.crop((x0, y0, x1, y1))
