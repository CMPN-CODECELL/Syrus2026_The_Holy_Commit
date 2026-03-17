#!/usr/bin/env python3
"""
JewelForge v2 — Setup verification script.
Run this to confirm all 10 components are ready before starting the server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _BACKEND_DIR / "weights"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    icon = PASS if ok else FAIL
    msg = f"  {detail}" if detail else ""
    print(f"  {icon}  {name}{msg}")


# ── 1. Python version ──────────────────────────────────────────────────────
print("\n[1] Python version")
major, minor, *_ = sys.version_info
ok = major == 3 and minor >= 10
check(f"Python {major}.{minor}.x", ok, "" if ok else f"  → got {sys.version.split()[0]}, need 3.10+")

# ── 2. PyTorch + CUDA ─────────────────────────────────────────────────────
print("\n[2] PyTorch + CUDA")
try:
    import torch  # type: ignore

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    vram = (
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        if cuda_ok
        else "N/A"
    )
    check("PyTorch importable", True, f"  v{torch.__version__}")
    check("CUDA available", cuda_ok, f"  GPU: {gpu_name}  VRAM: {vram}")
except ImportError as e:
    check("PyTorch importable", False, f"  → {e}")
    check("CUDA available", False, "  → PyTorch not installed")

# ── 3. rembg ──────────────────────────────────────────────────────────────
print("\n[3] rembg")
try:
    import rembg  # type: ignore
    check("rembg importable", True, f"  v{rembg.__version__}")
except ImportError as e:
    check("rembg importable", False, f"  → {e}")

# ── 4. Grounding DINO ─────────────────────────────────────────────────────
print("\n[4] Grounding DINO")
gdino_clone = _BACKEND_DIR / "models" / "GroundingDINO"
gdino_weights = _WEIGHTS_DIR / "groundingdino_swint_ogc.pth"
try:
    sys.path.insert(0, str(gdino_clone))
    import groundingdino  # type: ignore
    check("Grounding DINO importable", True)
except ImportError as e:
    check("Grounding DINO importable", False, f"  → {e}  (clone to models/GroundingDINO)")
check(
    "GDINO weights exist",
    gdino_weights.exists(),
    f"  {gdino_weights}" if gdino_weights.exists() else f"  → missing: {gdino_weights}",
)

# ── 5. SAM 2 ──────────────────────────────────────────────────────────────
print("\n[5] SAM 2")
sam2_clone = _BACKEND_DIR / "models" / "sam2"
sam2_weights = _WEIGHTS_DIR / "sam2.1_hiera_small.pt"
try:
    sys.path.insert(0, str(sam2_clone))
    import sam2  # type: ignore
    check("SAM 2 importable", True)
except ImportError as e:
    check("SAM 2 importable", False, f"  → {e}  (clone to models/sam2)")
check(
    "SAM 2 weights exist",
    sam2_weights.exists(),
    f"  {sam2_weights}" if sam2_weights.exists() else f"  → missing: {sam2_weights}",
)

# ── 6. TripoSG ────────────────────────────────────────────────────────────
print("\n[6] TripoSG / TripoSR")
triposr_local = _BACKEND_DIR / "models" / "TripoSR"
try:
    sys.path.insert(0, str(triposr_local))
    import tsr  # type: ignore
    check("TripoSG importable", True)
except ImportError as e:
    check("TripoSG importable", False, f"  → {e}  (clone to models/TripoSR or pip install tsr)")

# ── 7. Mask projector ─────────────────────────────────────────────────────
print("\n[7] Mask projector")
projector_found = None
try:
    import nvdiffrast.torch  # type: ignore
    projector_found = "nvdiffrast"
except ImportError:
    pass
if projector_found is None:
    try:
        import pytorch3d  # type: ignore
        projector_found = "pytorch3d"
    except ImportError:
        pass
if projector_found is None:
    try:
        import trimesh.ray.ray_pyembree  # type: ignore
        projector_found = "trimesh_raycast"
    except Exception:
        projector_found = "trimesh (pure Python fallback)"

check("Projector available", projector_found is not None, f"  using: {projector_found}")

# ── 8. xatlas ─────────────────────────────────────────────────────────────
print("\n[8] xatlas (UV unwrapping)")
try:
    import xatlas  # type: ignore
    check("xatlas importable", True)
except ImportError as e:
    check("xatlas importable", False, f"  → {e}")

# ── 9. API key ────────────────────────────────────────────────────────────
print("\n[9] API key")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
openai_key = os.environ.get("OPENAI_API_KEY", "")
api_ok = bool(anthropic_key and anthropic_key != "sk-ant-your-key-here") or bool(openai_key)
check(
    "API key set",
    api_ok,
    "  ANTHROPIC_API_KEY" if anthropic_key else "  → set ANTHROPIC_API_KEY in .env",
)

# ── 10. FastAPI ───────────────────────────────────────────────────────────
print("\n[10] FastAPI")
try:
    import fastapi  # type: ignore
    check("FastAPI importable", True, f"  v{fastapi.__version__}")
except ImportError as e:
    check("FastAPI importable", False, f"  → {e}")

# ── Summary ───────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"\n{'─' * 50}")
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  All checks passed! Run: uvicorn main:app --reload")
else:
    print("  Some checks failed. Review the ✗ items above.")
print()

sys.exit(0 if passed == total else 1)
