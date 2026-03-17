#!/usr/bin/env python3
"""
Test all 3 agentic behaviors of JewelForge v2.
Run from the backend/ directory: python ../scripts/test_agentic.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

PASS = "✓"
FAIL = "✗"
SKIP = "~"


def test_section(name):
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print('─' * 50)


def run_test(name, fn):
    try:
        result = fn()
        print(f"  {PASS} {name}")
        if result:
            print(f"      → {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        return False


# ── Agentic Behavior 1: Self-correcting segmentation ──────────────────────────
test_section("Agentic Behavior #1: Self-Correcting Segmentation")

from segment.jewelry_prompts import JEWELRY_PROMPTS

def test_prompt_tiers_exist():
    for jtype in ("ring", "necklace", "earring", "bracelet"):
        assert jtype in JEWELRY_PROMPTS, f"Missing: {jtype}"
        assert len(JEWELRY_PROMPTS[jtype]) >= 2, f"Need ≥2 tiers for {jtype}"
    return f"{len(JEWELRY_PROMPTS)} jewelry types, multi-tier prompts verified"

run_test("Prompt tiers defined for all jewelry types", test_prompt_tiers_exist)

from agent.segmentation_agent import SegmentationAgent

def test_color_fallback_interface():
    # Test without real models — just validate the interface
    class MockSegmenter:
        def segment(self, *args, **kwargs):
            return []  # always returns empty (simulates failed detection)
    agent = SegmentationAgent(MockSegmenter())
    # Color fallback should be triggered when no segments found
    assert hasattr(agent, '_color_fallback')
    assert hasattr(agent, '_deduplicate')
    assert hasattr(agent, 'segment_with_retries')
    return "SegmentationAgent interface verified"

run_test("SegmentationAgent retry/fallback interface", test_color_fallback_interface)

def test_deduplication():
    import numpy as np
    class MockSegmenter:
        def segment(self, *a, **kw): return []
    agent = SegmentationAgent(MockSegmenter())

    # Create two nearly-identical masks (high IoU)
    mask_a = np.zeros((10, 10), dtype=bool); mask_a[2:8, 2:8] = True
    mask_b = np.zeros((10, 10), dtype=bool); mask_b[2:8, 2:8] = True  # identical
    segs = [
        {"label": "metal", "mask": mask_a, "confidence": 0.9, "area_fraction": 0.36, "segment_id": 0, "bbox": [2,2,8,8]},
        {"label": "metal", "mask": mask_b, "confidence": 0.7, "area_fraction": 0.36, "segment_id": 1, "bbox": [2,2,8,8]},
    ]
    deduped = agent._deduplicate(segs)
    assert len(deduped) == 1, f"Expected 1 after dedup, got {len(deduped)}"
    return f"Deduplicated 2 → {len(deduped)} (identical masks removed)"

run_test("Deduplication of overlapping masks (IoU > 0.7)", test_deduplication)


# ── Agentic Behavior 2: Mesh quality validation ────────────────────────────────
test_section("Agentic Behavior #2: Mesh Quality Validation")

from agent.mesh_validator import MeshValidator
import trimesh
import tempfile
import numpy as np

def test_valid_mesh():
    mesh = trimesh.creation.sphere(radius=1.0)
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        mesh.export(f.name)
        result = MeshValidator.validate(f.name)
    os.unlink(f.name)
    assert result["recommendation"] == "accept"
    return f"Sphere: aspect={result['metrics']['aspect_ratio']}, rec=accept"

run_test("Valid sphere mesh → accept", test_valid_mesh)

def test_flat_mesh():
    verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=float)
    faces = np.array([[0,1,2],[0,2,3]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        mesh.export(f.name)
        result = MeshValidator.validate(f.name)
    os.unlink(f.name)
    assert result["recommendation"] in ("retry_with_rotation", "use_fallback")
    return f"Flat quad: aspect={result['metrics']['aspect_ratio']}, rec={result['recommendation']}"

run_test("Flat mesh → retry_with_rotation", test_flat_mesh)

def test_missing_file():
    result = MeshValidator.validate("/nonexistent/path/mesh.obj")
    assert result["valid"] == False
    assert result["recommendation"] == "use_fallback"
    return "Missing file → use_fallback"

run_test("Missing mesh file → use_fallback", test_missing_file)


# ── Agentic Behavior 3: Multi-step goal planning (pricing logic) ───────────────
test_section("Agentic Behavior #3: Multi-Step Goal Planning (Pricing Engine)")

from pricing.engine import estimate_price, suggest_budget_alternatives

def test_basic_price():
    config = {"jewelry_type": "ring", "metal": "yellow_gold", "gemstones": {"gemstone_center": "diamond"}}
    p = estimate_price(config)
    assert p["total"] > 0
    assert "metal" in p and "gemstones" in p and "labor_estimate" in p
    return f"Ring (18K gold + diamond) = ${p['total']:,.2f}"

run_test("Basic price estimation", test_basic_price)

def test_budget_alternatives():
    config = {"jewelry_type": "ring", "metal": "platinum", "gemstones": {"gemstone_center": "diamond"}}
    suggestions = suggest_budget_alternatives(config)
    assert len(suggestions) > 0
    assert all(s["savings"] > 0 for s in suggestions)
    top = suggestions[0]
    return f"Top saving: {top['description']} saves ${top['savings']:,.2f}"

run_test("Budget alternatives generated", test_budget_alternatives)

def test_budget_constraint():
    """Simulate LLM goal: find combo under $500"""
    budget = 500.0
    configs_to_try = [
        {"jewelry_type": "ring", "metal": "white_gold", "gemstones": {"gemstone_center": "cubic_zirconia"}},
        {"jewelry_type": "ring", "metal": "silver", "gemstones": {"gemstone_center": "amethyst"}},
        {"jewelry_type": "ring", "metal": "silver", "gemstones": {}},
    ]
    under_budget = [c for c in configs_to_try if estimate_price(c)["total"] <= budget]
    assert len(under_budget) >= 1
    best = min(under_budget, key=lambda c: abs(estimate_price(c)["total"] - budget))
    p = estimate_price(best)
    return f"Found combo under ${budget}: {best['metal']} = ${p['total']:.2f}"

run_test("Budget constraint satisfaction (≤$500)", test_budget_constraint)

def test_orchestrator_direct_swap():
    from agent.orchestrator import JewelForgeAgent
    agent = JewelForgeAgent()
    # Manually inject state (as if pipeline ran)
    agent.state["components"] = {
        "metal_band": {"type": "metal", "segment_id": 0, "detection_label": "band", "area_fraction": 0.6},
        "gemstone_center": {"type": "gemstone", "segment_id": 1, "detection_label": "stone", "area_fraction": 0.2},
    }
    agent.state["materials_applied"] = {"metal_band": "yellow_gold", "gemstone_center": "diamond"}
    agent.state["jewelry_type"] = "ring"

    result = agent.direct_swap("metal_band", "rose_gold")
    assert result["success"] == True
    assert result["material"] == "rose_gold"
    assert result["price"]["total"] > 0
    return f"Direct swap: metal_band→rose_gold, price=${result['price']['total']:.2f}"

run_test("Orchestrator direct_swap (Mode 3)", test_orchestrator_direct_swap)

def test_comparator():
    from pricing.comparator import compare_configurations
    configs = [
        {"label": "Budget", "jewelry_type": "ring", "metal": "silver", "gemstones": {}},
        {"label": "Mid-range", "jewelry_type": "ring", "metal": "white_gold", "gemstones": {"gemstone_center": "sapphire"}},
        {"label": "Luxury", "jewelry_type": "ring", "metal": "platinum", "gemstones": {"gemstone_center": "diamond"}},
    ]
    result = compare_configurations(configs)
    assert result["cheapest_label"] == "Budget"
    assert result["most_expensive_label"] == "Luxury"
    spread = result["price_range"]["spread"]
    return f"Price spread: ${spread:,.2f} ({result['cheapest_label']} → {result['most_expensive_label']})"

run_test("Price comparator (3 configs)", test_comparator)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 50}")
print("  All agentic behavior tests complete.")
print(f"{'=' * 50}\n")
