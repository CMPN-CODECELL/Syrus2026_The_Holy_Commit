"""
Texture baking + mask projection using trimesh raycasting.
No GPU compilation needed. Works on CPU. ~5-10s per operation.
"""
import numpy as np
import trimesh
import cv2
from PIL import Image


class MeshProjector:
    def __init__(self, device="cpu"):
        self.device = device
        self.mesh = None
        self.uvs = None

    def load_mesh(self, mesh_path: str):
        mesh = trimesh.load(mesh_path, force='mesh')
        trimesh.repair.fix_normals(mesh)
        self.mesh = mesh

        # Get or generate UV coordinates
        from trimesh.visual import TextureVisuals
        if isinstance(mesh.visual, TextureVisuals) and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            self.uvs = mesh.visual.uv
        else:
            self._generate_uvs()

    def _generate_uvs(self):
        """Generate UV coordinates using xatlas."""
        import xatlas
        vmapping, indices, uvs = xatlas.parametrize(
            self.mesh.vertices.astype(np.float32),
            self.mesh.faces.astype(np.uint32),
        )
        # Rebuild mesh with new topology
        self.mesh = trimesh.Trimesh(
            vertices=self.mesh.vertices[vmapping],
            faces=indices,
        )
        self.uvs = uvs

    def _get_face_id_map(self, image_size: int = 512) -> np.ndarray:
        """
        Cast rays from a grid to get face IDs at each pixel.
        Returns (H, W) int array where -1 = background.
        """
        # Center and scale mesh to fit in [-1, 1]
        bounds = self.mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        scale = (bounds[1] - bounds[0]).max()
        if scale == 0:
            scale = 1.0

        # Create ray grid (orthographic, front view)
        margin = 1.2  # slight margin around mesh
        x = np.linspace(-margin, margin, image_size)
        y = np.linspace(margin, -margin, image_size)  # flip Y for image coords
        xx, yy = np.meshgrid(x, y)

        # Rays from z=+5 pointing toward -z
        z_offset = 5.0
        ray_origins = np.stack([
            xx.ravel() * scale / 2 + center[0],
            yy.ravel() * scale / 2 + center[1],
            np.full(image_size ** 2, center[2] + z_offset),
        ], axis=-1)
        ray_directions = np.tile([0.0, 0.0, -1.0], (image_size ** 2, 1))

        # Cast rays
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False,
        )

        face_id_map = np.full((image_size, image_size), -1, dtype=np.int32)
        for ray_idx, face_idx in zip(index_ray, index_tri):
            row = ray_idx // image_size
            col = ray_idx % image_size
            face_id_map[row, col] = face_idx

        return face_id_map

    def bake_texture(self, input_image_path: str, output_texture_path: str,
                     texture_size: int = 1024) -> str:
        """
        Project input image colors onto UV texture atlas via raycasting.
        """
        input_img = np.array(
            Image.open(input_image_path).convert("RGB").resize((512, 512))
        ).astype(np.float32) / 255.0

        face_id_map = self._get_face_id_map(512)

        # For each visible pixel, find the UV coordinates of the hit face
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
        weight = np.zeros((texture_size, texture_size), dtype=np.float32)

        h, w = face_id_map.shape
        for row in range(h):
            for col in range(w):
                face_idx = face_id_map[row, col]
                if face_idx < 0:
                    continue

                # Get the face's UV coordinates (average of 3 vertices)
                face_verts = self.mesh.faces[face_idx]
                if self.uvs is not None and len(self.uvs) > 0:
                    face_uvs = self.uvs[face_verts]  # (3, 2)
                    avg_uv = face_uvs.mean(axis=0)
                else:
                    continue

                # Map UV to texture pixel
                tex_x = int(np.clip(avg_uv[0] * (texture_size - 1), 0, texture_size - 1))
                tex_y = int(np.clip((1.0 - avg_uv[1]) * (texture_size - 1), 0, texture_size - 1))

                # Write input image color at this texture position
                color = input_img[row, col]
                texture[tex_y, tex_x] += color
                weight[tex_y, tex_x] += 1.0

        # Average where multiple pixels mapped to same texel
        valid = weight > 0
        texture[valid] /= weight[valid, np.newaxis]

        # Convert to uint8
        texture_uint8 = (texture * 255).astype(np.uint8)

        # Fill holes via iterative dilation
        mask = (~valid).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(30):
            dilated = cv2.dilate(texture_uint8, kernel, iterations=1)
            texture_uint8[mask.astype(bool)] = dilated[mask.astype(bool)]
            mask = cv2.erode(mask, kernel, iterations=1)
            if mask.sum() == 0:
                break

        Image.fromarray(texture_uint8).save(output_texture_path)
        return output_texture_path

    def project_masks_to_faces(self, segments: list, image_size: int = 512) -> dict:
        """
        Project 2D segmentation masks onto 3D faces via face_id_map.
        """
        face_id_map = self._get_face_id_map(image_size)
        n_faces = len(self.mesh.faces)

        # Vote array: face_votes[face_idx][segment_idx] = pixel count
        face_votes = np.zeros((n_faces, len(segments)), dtype=np.int32)

        for seg_idx, seg in enumerate(segments):
            mask = seg["mask"]

            # Resize mask if needed
            if mask.shape != (image_size, image_size):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (image_size, image_size), Image.NEAREST
                    )
                ) > 127

            # Ensure boolean mask before bitwise AND
            mask = mask.astype(bool)
            # Where mask overlaps with visible faces
            overlap = mask & (face_id_map >= 0)
            visible_face_ids = face_id_map[overlap]

            for fid in visible_face_ids:
                if 0 <= fid < n_faces:
                    face_votes[fid][seg_idx] += 1

        # Assign each face to the segment with the most votes
        face_to_component = {}
        for face_idx in range(n_faces):
            if face_votes[face_idx].sum() > 0:
                best_seg = face_votes[face_idx].argmax()
                face_to_component[face_idx] = segments[best_seg]["label"]
            else:
                face_to_component[face_idx] = "metal_body"

        return face_to_component

    def export_labeled_glb(self, face_to_component: dict,
                           texture_path: str, output_path: str) -> tuple:
        """Export GLB with baked texture + labels JSON."""
        import json
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual import TextureVisuals

        mesh = self.mesh.copy()
        texture_img = Image.open(texture_path)
        material = PBRMaterial(baseColorTexture=texture_img)

        if self.uvs is not None:
            mesh.visual = TextureVisuals(uv=self.uvs, material=material)

        mesh.export(output_path)

        # Save component labels
        labels_path = output_path.replace(".glb", "_labels.json")
        component_faces = {}
        for fi, comp in face_to_component.items():
            component_faces.setdefault(comp, []).append(fi)

        with open(labels_path, "w") as f:
            json.dump({
                "face_to_component": {str(k): v for k, v in face_to_component.items()},
                "component_faces": component_faces,
                "components": list(component_faces.keys()),
            }, f)

        return output_path, labels_path
