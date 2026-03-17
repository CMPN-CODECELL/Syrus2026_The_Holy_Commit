# JewelForge v2
import torch
import numpy as np
import cv2
import trimesh
import xatlas
from PIL import Image

try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False


class MeshProjector:
    def __init__(self, device="cuda"):
        self.device = device
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.uvs = None

        if NVDIFFRAST_AVAILABLE:
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = None

    def load_mesh(self, mesh_path):
        mesh = trimesh.load(mesh_path, force='mesh')
        trimesh.repair.fix_normals(mesh)
        self.mesh = mesh
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device)

        if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            self.uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=self.device)
        else:
            vmapping, indices, uvs = xatlas.parametrize(
                mesh.vertices.astype(np.float32),
                mesh.faces.astype(np.uint32),
            )
            self.faces = torch.tensor(indices, dtype=torch.int32, device=self.device)
            self.vertices = torch.tensor(
                mesh.vertices[vmapping], dtype=torch.float32, device=self.device
            )
            self.uvs = torch.tensor(uvs, dtype=torch.float32, device=self.device)

    def rasterize(self, image_size=512):
        if not NVDIFFRAST_AVAILABLE:
            raise RuntimeError("nvdiffrast not available. Use raycast_fallback instead.")

        # Camera: perspective, front-facing
        fov_rad = np.radians(50.0)
        f = 1.0 / np.tan(fov_rad / 2.0)
        near, far = 0.01, 10.0
        proj = torch.tensor([
            [f, 0, 0,                         0],
            [0, f, 0,                         0],
            [0, 0, (far + near) / (near - far), -1],
            [0, 0, 2 * far * near / (near - far), 0],
        ], dtype=torch.float32, device=self.device)

        view = torch.eye(4, dtype=torch.float32, device=self.device)
        view[2, 3] = -2.0  # camera distance
        mvp = proj @ view

        verts_homo = torch.cat(
            [self.vertices, torch.ones(self.vertices.shape[0], 1, device=self.device)], dim=1
        )
        verts_clip = (mvp @ verts_homo.T).T.unsqueeze(0)

        rast, _ = dr.rasterize(
            self.glctx, verts_clip, self.faces, resolution=[image_size, image_size]
        )
        face_id_map = rast[0, :, :, 3].long() - 1  # -1 = background
        return rast, face_id_map

    def bake_texture(self, input_image_path, output_texture_path, texture_size=1024):
        if not NVDIFFRAST_AVAILABLE:
            # Fallback: copy input image as texture
            img = Image.open(input_image_path).convert("RGB").resize(
                (texture_size, texture_size), Image.LANCZOS
            )
            img.save(output_texture_path)
            return output_texture_path

        input_img = np.array(
            Image.open(input_image_path).convert("RGB").resize((512, 512))
        )
        input_tensor = torch.tensor(input_img, dtype=torch.float32, device=self.device) / 255.0

        rast, face_id_map = self.rasterize(512)
        uv_map, _ = dr.interpolate(self.uvs.unsqueeze(0).contiguous(), rast, self.faces)
        uv_map = uv_map[0]

        texture = torch.zeros(
            texture_size, texture_size, 3, dtype=torch.float32, device=self.device
        )
        weight = torch.zeros(
            texture_size, texture_size, 1, dtype=torch.float32, device=self.device
        )

        visible = face_id_map >= 0
        vis_uv = uv_map[visible]
        vis_color = input_tensor[visible]

        tex_x = (vis_uv[:, 0] * (texture_size - 1)).long().clamp(0, texture_size - 1)
        tex_y = ((1.0 - vis_uv[:, 1]) * (texture_size - 1)).long().clamp(0, texture_size - 1)

        texture[tex_y, tex_x] += vis_color
        weight[tex_y, tex_x] += 1.0

        valid = weight.squeeze(-1) > 0
        texture[valid] /= weight[valid]

        texture_np = (texture.cpu().numpy() * 255).astype(np.uint8)
        # Fill holes via iterative dilation
        mask = (~valid.cpu().numpy()).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(20):
            dilated = cv2.dilate(texture_np, kernel, iterations=1)
            texture_np[mask.astype(bool)] = dilated[mask.astype(bool)]
            mask = cv2.erode(mask, kernel, iterations=1)
            if mask.sum() == 0:
                break

        Image.fromarray(texture_np).save(output_texture_path)
        return output_texture_path

    def project_masks_to_faces(self, segments, image_size=512):
        if not NVDIFFRAST_AVAILABLE:
            # Fallback: assign all faces to the first segment label
            n_faces = len(self.faces)
            if segments:
                default_label = segments[0]["label"]
            else:
                default_label = "metal_body"
            return {i: default_label for i in range(n_faces)}

        _, face_id_map = self.rasterize(image_size)
        face_id_np = face_id_map.cpu().numpy()
        n_faces = len(self.faces)
        face_votes = np.zeros((n_faces, len(segments)), dtype=int)

        for seg_idx, seg in enumerate(segments):
            mask = seg["mask"]
            if mask.shape != (image_size, image_size):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (image_size, image_size), Image.NEAREST
                    )
                ) > 127
            overlap = mask & (face_id_np >= 0)
            for fid in face_id_np[overlap]:
                if 0 <= fid < n_faces:
                    face_votes[fid][seg_idx] += 1

        face_to_component = {}
        for fi in range(n_faces):
            if face_votes[fi].sum() > 0:
                face_to_component[fi] = segments[face_votes[fi].argmax()]["label"]
            else:
                face_to_component[fi] = "metal_body"
        return face_to_component

    def export_labeled_glb(self, face_to_component, texture_path, output_path):
        import json
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual import TextureVisuals

        mesh = self.mesh.copy()
        texture_img = Image.open(texture_path)
        material = PBRMaterial(baseColorTexture=texture_img)
        mesh.visual = TextureVisuals(uv=self.uvs.cpu().numpy(), material=material)
        mesh.export(output_path)

        labels_path = output_path.replace(".glb", "_labels.json")
        component_faces = {}
        for fi, comp in face_to_component.items():
            component_faces.setdefault(comp, []).append(fi)

        with open(labels_path, "w") as f:
            json.dump(
                {
                    "face_to_component": {str(k): v for k, v in face_to_component.items()},
                    "component_faces": component_faces,
                    "components": list(component_faces.keys()),
                },
                f,
            )
        return output_path, labels_path
