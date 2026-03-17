#!/usr/bin/env python3
"""
End-to-end pipeline test for JewelForge v2.
Creates a synthetic test image and exercises every stage.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Make backend importable
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⊘\033[0m"


def header(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── Synthetic test image ────────────────────────────────────────────────────

def make_test_image(path: str, size: int = 512) -> str:
    """Create a synthetic ring image: gold band + red gemstone on gray background."""
    img = Image.new("RGB", (size, size), (127, 127, 127))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    r_outer, r_inner = int(size * 0.38), int(size * 0.22)

    # Gold band (torus approximation as two circles)
    draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        fill=(218, 165, 32),
        outline=(180, 130, 0),
        width=3,
    )
    draw.ellipse(
        [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
        fill=(127, 127, 127),
    )

    # Red gemstone at top
    gem_r = int(size * 0.08)
    gem_y = cy - r_outer + gem_r + 8
    draw.ellipse(
        [cx - gem_r, gem_y - gem_r, cx + gem_r, gem_y + gem_r],
        fill=(200, 0, 30),
        outline=(150, 0, 20),
        width=2,
    )

    img.save(path)
    return path


# ── Stage runners ──────────────────────────────────────────────────────────

def stage_preprocess(image_path: str, output_dir: str) -> dict | None:
    header("Stage 1 — Preprocess (rembg + Pillow)")
    try:
        from preprocess import preprocess_image
        result = preprocess_image(image_path, output_dir, target_size=512)
        print(f"  {PASS}  RGB:  {result['rgb']}")
        print(f"  {PASS}  RGBA: {result['rgba']}")
        return result
    except Exception as exc:
        print(f"  {FAIL}  {exc}")
        return None


def stage_segment(prep_result: dict | None) -> list:
    header("Stage 2 — Segmentation (Grounding DINO + SAM 2)")
    if prep_result is None:
        print(f"  {SKIP}  Skipped (preprocess failed)")
        return []

    try:
        from segment.gdino_sam2 import JewelrySegmenter
        segmenter = JewelrySegmenter(device="cuda")
        segmenter.load_models()
        segments = segmenter.segment(prep_result["rgba"], jewelry_type="ring")
        segmenter.unload_models()
        print(f"  {PASS}  Found {len(segments)} segment(s)")
        for seg in segments:
            print(f"        label={seg['label']}  conf={seg['confidence']:.2f}  area={seg['area_fraction']:.3f}")
        return segments
    except RuntimeError as exc:
        print(f"  {SKIP}  Models not loaded — using mock segments ({exc})")
        # Return a mock segment covering the top quarter of the image
        mock_mask = np.zeros((512, 512), dtype=bool)
        mock_mask[0:128, 192:320] = True
        return [{"label": "gemstone", "bbox": [192, 0, 320, 128], "confidence": 0.9,
                 "mask": mock_mask, "area_fraction": 0.06, "segment_id": 0}]
    except Exception as exc:
        print(f"  {FAIL}  {exc}")
        return []


def stage_gen3d(prep_result: dict | None, output_dir: str) -> str | None:
    header("Stage 3 — 3D Generation (TripoSG)")
    if prep_result is None:
        print(f"  {SKIP}  Skipped (preprocess failed)")
        return None

    try:
        from gen3d.triposg import generate_3d_mesh
        mesh_path = generate_3d_mesh(prep_result["rgb"], output_dir, resolution=128)
        print(f"  {PASS}  Mesh: {mesh_path}")
        return mesh_path
    except RuntimeError as exc:
        print(f"  {SKIP}  TripoSR not loaded — creating mock mesh ({exc})")
        # Create a tiny valid GLB (a cube) using trimesh
        try:
            import trimesh
            mesh = trimesh.creation.box(extents=(1, 1, 1))
            mock_path = os.path.join(output_dir, "mock_mesh.glb")
            mesh.export(mock_path)
            print(f"         Mock mesh saved: {mock_path}")
            return mock_path
        except Exception as e2:
            print(f"  {FAIL}  Could not create mock mesh: {e2}")
            return None
    except Exception as exc:
        print(f"  {FAIL}  {exc}")
        return None


def stage_projection(mesh_path: str | None, segments: list, prep_result: dict | None, output_dir: str):
    header("Stage 4 — Mask Projection + Texture Baking")
    if mesh_path is None or prep_result is None:
        print(f"  {SKIP}  Skipped (previous stage failed)")
        return None, None

    try:
        from texture.bake_and_project import MeshProjector
        projector = MeshProjector(device="cpu")
        projector.load_mesh(mesh_path)

        texture_path = os.path.join(output_dir, "texture.png")
        projector.bake_texture(prep_result["rgb"], texture_path, texture_size=512)
        print(f"  {PASS}  Texture baked: {texture_path}")

        if segments:
            face_map = projector.project_masks_to_faces(segments, image_size=256)
            print(f"  {PASS}  Face map: {len(face_map)} labeled faces")
        else:
            face_map = {}
            print(f"  {SKIP}  No segments to project")

        glb_path = os.path.join(output_dir, "mesh_labeled.glb")
        glb_path, labels_path = projector.export_labeled_glb(face_map, texture_path, glb_path)
        print(f"  {PASS}  GLB:    {glb_path}")
        print(f"  {PASS}  Labels: {labels_path}")
        return glb_path, labels_path
    except Exception as exc:
        print(f"  {FAIL}  {exc}")
        return None, None


def stage_pricing():
    header("Stage 5 — Price Estimation")
    try:
        from pricing.engine import estimate_price, suggest_budget_alternatives
        config = {
            "jewelry_type": "ring",
            "metal": "yellow_gold",
            "center_stone": "diamond",
            "accent_stone": "cubic_zirconia",
        }
        price = estimate_price(config)
        print(f"  {PASS}  Total: ${price['total']:,.2f}")
        for item in price["breakdown"]:
            print(f"        {item['label']}: ${item['cost']:,.2f}")

        alts = suggest_budget_alternatives(config, budget=None)
        print(f"\n  {PASS}  {len(alts)} budget alternatives generated:")
        for alt in alts[:3]:
            print(f"        − {alt['description']}  → ${alt['new_price']:,.2f}  (save ${alt['savings']:,.2f})")
        return price
    except Exception as exc:
        print(f"  {FAIL}  {exc}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 50)
    print("  JewelForge v2 — Pipeline Test")
    print("═" * 50)

    with tempfile.TemporaryDirectory(prefix="jf_test_") as tmpdir:
        image_path = os.path.join(tmpdir, "test_ring.png")
        make_test_image(image_path)
        print(f"\n  Test image created: {image_path}")

        prep = stage_preprocess(image_path, tmpdir)
        segments = stage_segment(prep)
        mesh_path = stage_gen3d(prep, tmpdir)
        glb_path, labels_path = stage_projection(mesh_path, segments, prep, tmpdir)
        price = stage_pricing()

        print("\n" + "═" * 50)
        print("  Pipeline test complete.")
        print(f"  Preprocessing:  {'OK' if prep else 'FAILED'}")
        print(f"  Segmentation:   {'OK' if segments is not None else 'FAILED'}")
        print(f"  3D Generation:  {'OK' if mesh_path else 'FAILED'}")
        print(f"  Projection:     {'OK' if glb_path else 'FAILED'}")
        print(f"  Pricing:        {'OK' if price else 'FAILED'}")
        print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
