# JewelForge v2
from .engine import estimate_price
from ..materials.presets import METAL_PRESETS, GEMSTONE_PRESETS


def compare_configurations(configs: list[dict]) -> dict:
    """
    Compare multiple jewelry configurations side by side.
    Each config dict should have keys: label, jewelry_type, metal, gemstones, metal_weight_grams (optional).
    Returns a comparison table with prices and differences from cheapest.
    """
    if not configs:
        return {"error": "No configurations provided"}

    results = []
    for cfg in configs:
        label = cfg.pop("label", f"Config {len(results)+1}")
        price = estimate_price(cfg)
        results.append({
            "label": label,
            "config": cfg,
            "price_breakdown": price,
            "total": price["total"],
        })

    results.sort(key=lambda r: r["total"])
    cheapest = results[0]["total"]

    for r in results:
        r["vs_cheapest"] = round(r["total"] - cheapest, 2)
        r["vs_cheapest_pct"] = round((r["vs_cheapest"] / cheapest * 100) if cheapest > 0 else 0, 1)

    return {
        "configurations": results,
        "cheapest_label": results[0]["label"],
        "most_expensive_label": results[-1]["label"],
        "price_range": {
            "min": results[0]["total"],
            "max": results[-1]["total"],
            "spread": round(results[-1]["total"] - results[0]["total"], 2),
        },
    }


def find_closest_to_budget(budget: float, jewelry_type: str = "ring") -> dict:
    """
    Find the best material combination closest to (but under) a given budget.
    Returns the best config and its price.
    """
    best = None
    best_price = 0.0

    for metal_key in METAL_PRESETS:
        # Try with no gemstones
        cfg = {"jewelry_type": jewelry_type, "metal": metal_key, "gemstones": {}}
        p = estimate_price(cfg)["total"]
        if p <= budget and p > best_price:
            best_price = p
            best = {**cfg, "estimated_price": p}

        # Try with one center gemstone
        for gem_key in GEMSTONE_PRESETS:
            cfg = {
                "jewelry_type": jewelry_type,
                "metal": metal_key,
                "gemstones": {"gemstone_center": gem_key},
            }
            p = estimate_price(cfg)["total"]
            if p <= budget and p > best_price:
                best_price = p
                best = {**cfg, "estimated_price": p}

    if best is None:
        # All combos exceed budget — return cheapest
        cfg = {"jewelry_type": jewelry_type, "metal": "silver", "gemstones": {}}
        p = estimate_price(cfg)["total"]
        best = {**cfg, "estimated_price": p, "note": "Exceeds budget — cheapest option returned"}

    return best
