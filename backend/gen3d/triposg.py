# JewelForge v2
import os
import sys
import torch
import trimesh

TARGET_FACE_COUNT = 100000


def _mesh_to_pymesh(vertices, faces):
    import pymeshlab

    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(mesh)
    return mesh_set


def _pymesh_to_trimesh(mesh):
    return trimesh.Trimesh(
        vertices=mesh.vertex_matrix(),
        faces=mesh.face_matrix(),
        process=False,
    )


def _set_face_count(mesh, target_faces=TARGET_FACE_COUNT):
    current_faces = int(mesh.faces.shape[0])
    if current_faces == target_faces:
        return mesh

    adjusted = mesh.copy()

    if current_faces < target_faces:
        # Subdivide until we have enough triangles to decimate down cleanly.
        while int(adjusted.faces.shape[0]) < target_faces:
            adjusted = adjusted.subdivide()

    if int(adjusted.faces.shape[0]) > target_faces:
        mesh_set = _mesh_to_pymesh(adjusted.vertices, adjusted.faces)
        mesh_set.meshing_merge_close_vertices()
        mesh_set.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
        adjusted = _pymesh_to_trimesh(mesh_set.current_mesh())

    return adjusted


def generate_3d_mesh(
    image_path,
    output_dir,
    num_inference_steps=50,
    guidance_scale=7.0,
    target_face_count=TARGET_FACE_COUNT,
):
    """
    Generate a 3D mesh from a 512x512 RGB PNG using TripoSG.
    Returns path to the exported .glb file.
    """
    sys.path.insert(0, os.path.abspath("models/TripoSG"))

    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from PIL import Image

    weights_dir = "weights/TripoSG"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = TripoSGPipeline.from_pretrained(weights_dir).to(device, dtype)

    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        outputs = pipe(
            image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(42),
            flash_octree_depth=7,   # 128³ grid — fast enough for hackathon demo
            use_flash_decoder=True,
        )

    mesh = outputs.meshes[0]
    original_faces = int(mesh.faces.shape[0])
    mesh = _set_face_count(mesh, target_faces=target_face_count)
    final_faces = int(mesh.faces.shape[0])
    print(f"TripoSG face count: {original_faces} -> {final_faces}")

    output_path = f"{output_dir}/mesh.glb"
    mesh.export(output_path)

    del pipe
    torch.cuda.empty_cache()
    return output_path
