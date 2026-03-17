"""
Texture baking and mask-to-3D-face projection for JewelForge v2.
Supports nvdiffrast (primary), pytorch3d, and trimesh raycasting (fallback).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _BACKEND_DIR / "projection_config.json"


def _load_projector_backend() -> str:
    try:
        with open(_CONFIG_PATH) as fh:
            return json.load(fh).get("projector", "nvdiffrast")
    except Exception:
        return "nvdiffrast"


class MeshProjector:
    """
    Projects a 2D jewelry image onto a 3D mesh to create a baked texture,
    and maps segmentation masks to mesh face labels.

    Usage:
        proj = MeshProjector(device="cuda")
        proj.load_mesh("ring.glb")
        proj.bake_texture("ring_preprocessed.png", "ring_texture.png")
        face_map = proj.project_masks_to_faces(segments)
        glb_path, labels_path = proj.export_labeled_glb(face_map, "ring_texture.png", "ring_labeled.glb")
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._backend = _load_projector_backend()
        self._mesh: Optional[trimesh.Trimesh] = None
        self._vertices: Optional[np.ndarray] = None  # (V, 3) float32
        self._faces: Optional[np.ndarray] = None      # (F, 3) int32
        self._uvs: Optional[np.ndarray] = None        # (V, 2) float32
        self._face_id_map: Optional[np.ndarray] = None  # (H, W) int32

        # Try to initialise the preferred backend
        self._rast_ctx = None
        if self._backend == "nvdiffrast":
            self._rast_ctx = self._init_nvdiffrast()
            if self._rast_ctx is None:
                self._backend = "trimesh_raycast"

    # ── Public API ─────────────────────────────────────────────────────────

    def load_mesh(self, mesh_path: str) -> None:
        """Load mesh from a .glb or .obj file; generate UVs if missing."""
        scene_or_mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(scene_or_mesh, trimesh.Scene):
            meshes = list(scene_or_mesh.dump())
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene_or_mesh

        self._mesh = mesh
        self._vertices = np.array(mesh.vertices, dtype=np.float32)
        self._faces = np.array(mesh.faces, dtype=np.int32)

        # Generate UVs if the mesh doesn't have them
        if mesh.visual is not None and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            self._uvs = np.array(mesh.visual.uv, dtype=np.float32)
        else:
            self._uvs = self._generate_uvs()

    def rasterize(self, image_size: int = 512) -> Tuple[object, np.ndarray]:
        """
        Render the mesh from the front view and produce a face-id map.

        Returns:
            (rast_output, face_id_map) where face_id_map is (H, W) int32,
            with -1 for background pixels and face index elsewhere.
        """
        if self._backend == "nvdiffrast" and self._rast_ctx is not None:
            return self._rasterize_nvdiffrast(image_size)
        return self._rasterize_trimesh(image_size)

    def bake_texture(
        self,
        input_image_path: str,
        output_texture_path: str,
        texture_size: int = 1024,
    ) -> str:
        """
        Project input image colors onto the UV map via face_id_map.

        Args:
            input_image_path: Path to the preprocessed RGB image.
            output_texture_path: Where to save the baked texture PNG.
            texture_size: Output texture resolution (square).

        Returns:
            Path to the saved texture PNG.
        """
        import cv2
        from PIL import Image

        if self._face_id_map is None:
            _, self._face_id_map = self.rasterize(image_size=512)

        src_img = np.array(Image.open(input_image_path).convert("RGB").resize((512, 512)))
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

        h_src, w_src = src_img.shape[:2]
        face_map = self._face_id_map  # (H, W)

        # For each pixel in the render, look up its face and UV, write to texture
        ys, xs = np.where(face_map >= 0)
        for y, x in zip(ys, xs):
            face_idx = face_map[y, x]
            src_color = src_img[y * h_src // face_map.shape[0], x * w_src // face_map.shape[1]]
            if self._uvs is not None and self._faces is not None:
                tri = self._faces[face_idx]
                uv_centroid = self._uvs[tri].mean(axis=0)
                tx = int(np.clip(uv_centroid[0] * (texture_size - 1), 0, texture_size - 1))
                ty = int(np.clip((1 - uv_centroid[1]) * (texture_size - 1), 0, texture_size - 1))
                texture[ty, tx] = src_color

        # Fill holes with iterative dilation
        mask = (texture.sum(axis=2) == 0).astype(np.uint8)
        for _ in range(8):
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(texture, kernel, iterations=1)
            texture = np.where(mask[:, :, None] == 1, dilated, texture)
            mask = (texture.sum(axis=2) == 0).astype(np.uint8)

        Image.fromarray(texture).save(output_texture_path)
        return output_texture_path

    def project_masks_to_faces(
        self, segments: List[dict], image_size: int = 512
    ) -> Dict[int, str]:
        """
        For each segment mask, find the overlapping face IDs and vote per face.

        Args:
            segments: Output of JewelrySegmenter.segment().
            image_size: Resolution at which the face_id_map was rendered.

        Returns:
            Dict mapping face_index (int) → component_name (str).
        """
        if self._face_id_map is None:
            _, self._face_id_map = self.rasterize(image_size=image_size)

        face_id_map = self._face_id_map
        face_votes: Dict[int, Dict[str, int]] = {}

        for seg in segments:
            mask: np.ndarray = seg["mask"]  # (H_orig, W_orig) bool
            label: str = seg["label"]

            # Resize mask to match face_id_map size if needed
            if mask.shape != face_id_map.shape:
                import cv2
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (face_id_map.shape[1], face_id_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            overlapping_faces = face_id_map[mask & (face_id_map >= 0)]
            for fid in overlapping_faces:
                fid = int(fid)
                if fid not in face_votes:
                    face_votes[fid] = {}
                face_votes[fid][label] = face_votes[fid].get(label, 0) + 1

        # Winner-takes-all vote per face
        face_to_component: Dict[int, str] = {}
        for fid, votes in face_votes.items():
            face_to_component[fid] = max(votes, key=votes.get)  # type: ignore[arg-type]

        return face_to_component

    def export_labeled_glb(
        self,
        face_to_component: Dict[int, str],
        texture_path: str,
        output_path: str,
    ) -> Tuple[str, str]:
        """
        Export GLB with baked texture + save _labels.json with face→component mapping.

        Returns:
            (glb_path, labels_json_path)
        """
        if self._mesh is None:
            raise RuntimeError("Call load_mesh() first.")

        from PIL import Image

        texture_img = Image.open(texture_path)
        mesh_copy = self._mesh.copy()

        # Attach texture
        mesh_copy.visual = trimesh.visual.TextureVisuals(
            uv=self._uvs,
            image=texture_img,
        )

        mesh_copy.export(output_path)

        labels_path = output_path.replace(".glb", "_labels.json")
        with open(labels_path, "w") as fh:
            json.dump({str(k): v for k, v in face_to_component.items()}, fh, indent=2)

        return output_path, labels_path

    # ── Private helpers ────────────────────────────────────────────────────

    def _generate_uvs(self) -> np.ndarray:
        """Generate UV coordinates via xatlas."""
        try:
            import xatlas
            vmapping, indices, uvs = xatlas.parametrize(self._vertices, self._faces)
            self._faces = indices
            return uvs.astype(np.float32)
        except Exception:
            # Fallback: spherical projection
            v = self._vertices
            lon = np.arctan2(v[:, 0], v[:, 2])
            lat = np.arcsin(np.clip(v[:, 1] / (np.linalg.norm(v, axis=1) + 1e-8), -1, 1))
            u = (lon / (2 * np.pi) + 0.5).astype(np.float32)
            t = (lat / np.pi + 0.5).astype(np.float32)
            return np.stack([u, t], axis=1)

    def _init_nvdiffrast(self):
        try:
            import nvdiffrast.torch as dr
            import torch
            ctx = dr.RasterizeCudaContext() if self.device == "cuda" else dr.RasterizeGLContext()
            return ctx
        except Exception:
            return None

    def _rasterize_nvdiffrast(self, image_size: int) -> Tuple[object, np.ndarray]:
        import nvdiffrast.torch as dr
        import torch

        verts = torch.from_numpy(self._vertices).float().to(self.device)
        faces = torch.from_numpy(self._faces).int().to(self.device)

        # Normalize vertices to [-1, 1] NDC (front-view orthographic)
        verts_ndc = verts.clone()
        for i in range(3):
            v_min, v_max = verts_ndc[:, i].min(), verts_ndc[:, i].max()
            verts_ndc[:, i] = 2 * (verts_ndc[:, i] - v_min) / (v_max - v_min + 1e-8) - 1

        # Append w=1 for homogeneous coords
        ones = torch.ones(verts_ndc.shape[0], 1, device=self.device)
        verts_h = torch.cat([verts_ndc, ones], dim=1).unsqueeze(0)

        rast_out, _ = dr.rasterize(self._rast_ctx, verts_h, faces, resolution=[image_size, image_size])
        face_id_map = rast_out[0, :, :, 3].long().cpu().numpy() - 1  # 0-indexed, -1=bg

        return rast_out, face_id_map.astype(np.int32)

    def _rasterize_trimesh(self, image_size: int) -> Tuple[None, np.ndarray]:
        """Fallback: trimesh raycasting from front orthographic view."""
        if self._mesh is None:
            raise RuntimeError("No mesh loaded.")

        verts = self._vertices
        bounds_min = verts.min(axis=0)
        bounds_max = verts.max(axis=0)
        center = (bounds_min + bounds_max) / 2
        extent = (bounds_max - bounds_min).max()

        # Cast rays from z+ towards z-
        xs = np.linspace(center[0] - extent / 2, center[0] + extent / 2, image_size)
        ys = np.linspace(center[1] - extent / 2, center[1] + extent / 2, image_size)
        xx, yy = np.meshgrid(xs, ys[::-1])

        origins = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, center[2] + extent)], axis=1)
        directions = np.tile([0.0, 0.0, -1.0], (origins.shape[0], 1))

        ray_caster = trimesh.ray.ray_pyembree.RayMeshIntersector(self._mesh)
        locations, ray_indices, tri_indices = ray_caster.intersects_location(
            origins, directions, multiple_hits=False
        )

        face_id_map = np.full(image_size * image_size, -1, dtype=np.int32)
        face_id_map[ray_indices] = tri_indices
        face_id_map = face_id_map.reshape(image_size, image_size)

        return None, face_id_map
