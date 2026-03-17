# JewelForge v2
import torch
import sys


def generate_3d_mesh(image_path, output_dir, resolution=256, chunk_size=4096):
    """
    Generate a 3D mesh from a 512x512 RGB PNG using TripoSG.
    Returns path to the exported .obj file.
    """
    torch.cuda.empty_cache()
    sys.path.insert(0, "models/TripoSG")
    from tsr.system import TSR
    from PIL import Image

    model = TSR.from_pretrained(
        "weights/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to("cuda")

    image = Image.open(image_path)
    with torch.no_grad():
        scene_codes = model([image], device="cuda")
        meshes = model.extract_mesh(scene_codes, resolution=resolution)

    output_path = f"{output_dir}/mesh.obj"
    meshes[0].export(output_path)

    del model
    torch.cuda.empty_cache()
    return output_path
