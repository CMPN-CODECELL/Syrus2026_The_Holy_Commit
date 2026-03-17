import argparse
import gc
import sys
from pathlib import Path

import torch
from PIL import Image


def generate_3d_mesh(image_path: Path, output_path: Path, repo_path: Path, chunk_size: int, resolution: int, device: str):
    if not repo_path.exists():
        raise FileNotFoundError(
            f"TripoSR repo not found at {repo_path}. Clone it first: "
            "git clone https://github.com/VAST-AI-Research/TripoSR TripoSR"
        )

    sys.path.insert(0, str(repo_path.resolve()))

    from tsr.system import TSR

    try:
        from tsr.utils import to_gradio_3d_orientation
    except Exception:
        to_gradio_3d_orientation = None

    model = None
    try:
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)

        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            scene_codes = model([image], device=device)

        meshes = model.extract_mesh(scene_codes, resolution=resolution)
        mesh = meshes[0]
        if to_gradio_3d_orientation is not None:
            mesh = to_gradio_3d_orientation(mesh)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run local TripoSR 2D -> 3D mesh generation")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--repo", default="TripoSR", help="Path to local TripoSR repository")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Renderer chunk size (2048 or lower if OOM)")
    parser.add_argument("--resolution", type=int, default=256, help="Mesh extraction resolution")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    repo_path = Path(args.repo)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    generate_3d_mesh(
        image_path=input_path,
        output_path=output_path,
        repo_path=repo_path,
        chunk_size=args.chunk_size,
        resolution=args.resolution,
        device=device,
    )

    print(f"GLB written to: {output_path}")


if __name__ == "__main__":
    main()
