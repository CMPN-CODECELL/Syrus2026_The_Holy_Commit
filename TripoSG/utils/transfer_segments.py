"""
Transfer 2D ring segmentation masks to 3D mesh face labels.

Since TripoSG GLBs contain only geometry (no vertex colours), we classify
faces by projecting each face centroid onto the XY plane (the canonical
"front" view that matches the input image) and looking up which 2D mask
region it lands in.

Output: <prefix>_labels.json with per-face component indices.
Schema:
    {
        "components":  ["band", "gemstone"],
        "face_labels": [0, 0, 1, 0, 1, ...],   // one int per face
        "face_count":  N,
        "label_counts": {"band": N0, "gemstone": N1}
    }
"""

import json
import logging
import os

import numpy as np
import trimesh

log = logging.getLogger(__name__)

COMPONENTS = ["band", "gemstone"]


# ── Public entry point ────────────────────────────────────────────────────────

def transfer_segmentation_to_3d(
    glb_path: str,
    band_mask: np.ndarray,
    gemstone_mask: np.ndarray,
    output_dir: str,
    prefix: str = "generated_mesh",
) -> dict:
    """
    Assign per-face labels to a TripoSG GLB and write <prefix>_labels.json.

    Args:
        glb_path:      Path to the exported GLB.
        band_mask:     uint8 (H, W) mask — 255 = band pixel.
        gemstone_mask: uint8 (H, W) mask — 255 = gemstone pixel.
        output_dir:    Where to write <prefix>_labels.json.
        prefix:        Stem used for the labels filename.

    Returns:
        {
            "labels_path":  str,
            "face_count":   int,
            "label_counts": {"band": N0, "gemstone": N1},
        }
    """
    mesh = _load_mesh(glb_path)
    n_faces = len(mesh.faces)
    log.info("Mesh loaded: %d vertices, %d faces", len(mesh.vertices), n_faces)

    # ── XY-plane projection ───────────────────────────────────────────────────
    face_labels = _project_and_classify(mesh, band_mask, gemstone_mask)

    # Sanity-check: label array must have exactly one entry per face.
    if len(face_labels) != n_faces:
        raise RuntimeError(
            f"Face count mismatch: GLB has {n_faces} faces but labels "
            f"has {len(face_labels)}. Aborting label transfer to prevent corrupt output."
        )

    unique, counts = np.unique(face_labels, return_counts=True)
    label_counts = {COMPONENTS[i]: 0 for i in range(len(COMPONENTS))}
    for u, c in zip(unique, counts):
        if 0 <= u < len(COMPONENTS):
            label_counts[COMPONENTS[u]] = int(c)
    log.info("Face label distribution: %s", label_counts)

    # ── Validate: if only 1 unique label, log a clear warning ─────────────────
    if len(unique) == 1:
        log.warning(
            "transfer_segmentation_to_3d: only 1 unique label found (%s). "
            "All faces assigned to '%s'. Check that the 2D gemstone_mask "
            "has non-zero area.",
            unique[0], COMPONENTS[unique[0]],
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    labels_path = os.path.join(output_dir, f"{prefix}_labels.json")
    payload = {
        "components":   COMPONENTS,
        "face_labels":  face_labels.tolist(),
        "face_count":   n_faces,
        "label_counts": label_counts,
    }
    with open(labels_path, "w") as fh:
        json.dump(payload, fh)
    log.info("Labels written: %s", labels_path)

    return {
        "labels_path":  labels_path,
        "face_count":   n_faces,
        "label_counts": label_counts,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_mesh(glb_path: str) -> trimesh.Trimesh:
    """Load a GLB and return a single Trimesh (concatenating multi-mesh scenes)."""
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
    elif isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        raise ValueError(f"Unexpected trimesh type: {type(loaded)}")

    if not meshes:
        raise ValueError(f"No triangle meshes found in {glb_path}")

    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def _project_and_classify(
    mesh: trimesh.Trimesh,
    band_mask: np.ndarray,
    gemstone_mask: np.ndarray,
) -> np.ndarray:
    """
    Project face centroids onto the XY plane (front view), normalise to
    image coordinates, look up each mask.

    The ring is assumed to be centred at the origin (TripoSG normalises its
    output mesh). X increases right, Y increases up in mesh space.
    In image space: u = (x - x_min) / x_range, v = 1 - (y - y_min) / y_range.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)   # (V, 3)
    faces = np.asarray(mesh.faces,    dtype=np.int64)     # (F, 3)

    # Face centroids
    centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0

    xs = centroids[:, 0]
    ys = centroids[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_range = max(float(x_max - x_min), 1e-6)
    y_range = max(float(y_max - y_min), 1e-6)

    img_h, img_w = gemstone_mask.shape

    # Normalise to pixel indices
    px = np.clip(((xs - x_min) / x_range * (img_w - 1)).astype(np.int32), 0, img_w - 1)
    py = np.clip(((1.0 - (ys - y_min) / y_range) * (img_h - 1)).astype(np.int32), 0, img_h - 1)

    # Look up gemstone mask at projected position
    is_gem = gemstone_mask[py, px] > 128

    face_labels = np.where(is_gem, 1, 0).astype(np.int32)

    log.info(
        "Projection: %d / %d faces classified as gemstone (%.1f%%)",
        int(is_gem.sum()), len(face_labels), 100 * is_gem.mean(),
    )
    return face_labels
