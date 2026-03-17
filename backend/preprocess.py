# JewelForge v2
from rembg import remove
from PIL import Image


def preprocess_image(input_path: str, output_dir: str, target_size: int = 512) -> dict:
    """
    Preprocess a jewelry image:
    - Remove background (rembg)
    - Center-crop to square
    - Resize to target_size x target_size
    - Export two variants:
        rgb:  PNG with gray background (for TripoSG)
        rgba: PNG with transparency (for segmentation)
    """
    img = Image.open(input_path)
    img_rgba = remove(img)

    # Center-crop to square
    w, h = img_rgba.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img_cropped = img_rgba.crop((left, top, left + side, top + side))
    img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)

    # RGB with gray bg (for TripoSG)
    bg = Image.new("RGB", (target_size, target_size), (200, 200, 200))
    bg.paste(img_resized, mask=img_resized.split()[3])
    rgb_path = f"{output_dir}/preprocessed.png"
    bg.save(rgb_path, "PNG")

    # RGBA (for segmentation)
    rgba_path = f"{output_dir}/preprocessed_rgba.png"
    img_resized.save(rgba_path, "PNG")

    return {"rgb": rgb_path, "rgba": rgba_path}
