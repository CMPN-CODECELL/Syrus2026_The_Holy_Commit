# JewelForge v2
"""
10-point verification script for JewelForge v2 setup.
Run from the backend/ directory: python verify_setup.py
"""
import sys
import os

PASS = "✓"
FAIL = "✗"
checks_passed = 0
checks_total = 0


def check(name, fn):
    global checks_passed, checks_total
    checks_total += 1
    try:
        fn()
        print(f"  {PASS} {name}")
        checks_passed += 1
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")


print("\nJewelForge v2 — Setup Verification\n" + "=" * 40)

# 1. CUDA
def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available!"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"\n     GPU: {name} ({vram:.1f} GB VRAM)", end="")

check("PyTorch + CUDA", check_cuda)

# 2. rembg
def check_rembg():
    from rembg import remove
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8) * 128)
    remove(img)

check("rembg (background removal)", check_rembg)

# 3. Grounding DINO importable
def check_gdino():
    sys.path.insert(0, "models/GroundingDINO")
    from groundingdino.util.inference import load_model  # noqa: F401

check("Grounding DINO (import)", check_gdino)

# 4. SAM 2 importable
def check_sam2():
    from sam2.build_sam import build_sam2  # noqa: F401

check("SAM 2 (import)", check_sam2)

# 5. TripoSG importable
def check_triposg():
    sys.path.insert(0, "models/TripoSG")
    from tsr.system import TSR  # noqa: F401

check("TripoSG (import)", check_triposg)

# 6. Weights present
def check_weights():
    required = [
        "weights/groundingdino_swint_ogc.pth",
        "weights/sam2.1_hiera_small.pt",
        "weights/TripoSR/config.yaml",
        "weights/TripoSR/model.ckpt",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    assert not missing, f"Missing: {missing}"

check("Model weights present", check_weights)

# 7. nvdiffrast (optional)
def check_nvdiffrast():
    import nvdiffrast.torch as dr  # noqa: F401
    print(" (GPU rasterization available)", end="")

try:
    check("nvdiffrast (GPU rasterization)", check_nvdiffrast)
except Exception:
    checks_total += 1  # already counted but didn't pass
    print(f"  ~ nvdiffrast not available — trimesh raycasting fallback will be used")

# 8. trimesh
def check_trimesh():
    import trimesh
    m = trimesh.creation.box()
    assert len(m.faces) > 0

check("trimesh", check_trimesh)

# 9. Anthropic API key
def check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    assert key.startswith("sk-ant"), (
        "ANTHROPIC_API_KEY not set or invalid. "
        "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
    )

check("Anthropic API key", check_api_key)

# 10. FastAPI importable
def check_fastapi():
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401

check("FastAPI + uvicorn", check_fastapi)

print(f"\n{'=' * 40}")
print(f"Passed {checks_passed}/{checks_total} checks")
if checks_passed == checks_total:
    print("All checks passed! Ready to run.")
else:
    print(f"{checks_total - checks_passed} issue(s) to fix before running.")
print()
