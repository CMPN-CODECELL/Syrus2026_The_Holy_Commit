# JewelForge v2
import trimesh
import numpy as np


def get_face_id_map_raycast(mesh_path, image_size=256):
    """
    CPU-based ray casting fallback for generating a face ID map.
    Returns a (image_size x image_size) int array where each pixel holds
    the index of the intersected face, or -1 for background.

    Slower than nvdiffrast (~5-10s for 256px) but requires no GPU compilation.
    """
    mesh = trimesh.load(mesh_path, force='mesh')

    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    xx, yy = np.meshgrid(x, y)

    ray_origins = np.stack(
        [xx.ravel(), yy.ravel(), np.full(image_size ** 2, 3.0)], axis=-1
    )
    ray_directions = np.tile([0, 0, -1], (image_size ** 2, 1))

    _, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins.astype(np.float64),
        ray_directions=ray_directions.astype(np.float64),
        multiple_hits=False,
    )

    face_id_map = np.full((image_size, image_size), -1, dtype=int)
    for ray_idx, face_idx in zip(index_ray, index_tri):
        row = ray_idx // image_size
        col = ray_idx % image_size
        face_id_map[row, col] = face_idx

    return face_id_map


def project_masks_raycast(mesh_path, segments, image_size=256):
    """
    Project 2D segmentation masks onto 3D face groups using raycasting.
    Returns face_to_component dict mapping face index → component label.
    """
    face_id_map = get_face_id_map_raycast(mesh_path, image_size)

    mesh = trimesh.load(mesh_path, force='mesh')
    n_faces = len(mesh.faces)
    face_votes = np.zeros((n_faces, len(segments)), dtype=int)

    for seg_idx, seg in enumerate(segments):
        from PIL import Image
        mask = seg["mask"]
        if mask.shape != (image_size, image_size):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (image_size, image_size), Image.NEAREST
                )
            ) > 127

        overlap = mask & (face_id_map >= 0)
        for fid in face_id_map[overlap]:
            if 0 <= fid < n_faces:
                face_votes[fid][seg_idx] += 1

    face_to_component = {}
    for fi in range(n_faces):
        if face_votes[fi].sum() > 0:
            face_to_component[fi] = segments[face_votes[fi].argmax()]["label"]
        else:
            face_to_component[fi] = "metal_body"

    return face_to_component
