"""
Price estimation engine for JewelForge v2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Data tables ────────────────────────────────────────────────────────────

METAL_PRICES: Dict[str, Dict[str, Any]] = {
    "yellow_gold": {"price_per_gram": 62.50, "display_name": "Yellow Gold (18K)"},
    "white_gold": {"price_per_gram": 65.00, "display_name": "White Gold (18K)"},
    "rose_gold": {"price_per_gram": 60.00, "display_name": "Rose Gold (18K)"},
    "platinum": {"price_per_gram": 32.00, "display_name": "Platinum (950)"},
    "silver": {"price_per_gram": 0.85, "display_name": "Sterling Silver (925)"},
    "copper": {"price_per_gram": 0.01, "display_name": "Copper"},
}

GEMSTONE_PRICES: Dict[str, Dict[str, Any]] = {
    "diamond": {"price_per_carat": 5000.00, "display_name": "Diamond"},
    "ruby": {"price_per_carat": 3000.00, "display_name": "Ruby"},
    "sapphire": {"price_per_carat": 2500.00, "display_name": "Sapphire"},
    "emerald": {"price_per_carat": 4000.00, "display_name": "Emerald"},
    "amethyst": {"price_per_carat": 20.00, "display_name": "Amethyst"},
    "topaz": {"price_per_carat": 25.00, "display_name": "Topaz"},
    "cubic_zirconia": {"price_per_carat": 2.00, "display_name": "Cubic Zirconia"},
}

# Default weight/carat estimates per jewelry type
DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "ring": {"metal_grams": 4.5, "center_carats": 0.5, "accent_carats": 0.2},
    "necklace": {"metal_grams": 8.0, "center_carats": 0.3, "accent_carats": 0.0},
    "earring": {"metal_grams": 3.0, "center_carats": 0.2, "accent_carats": 0.1},
    "bracelet": {"metal_grams": 12.0, "center_carats": 0.0, "accent_carats": 0.3},
}

LABOR_MARKUP = 0.30  # 30 % labor + overhead markup


# ── Public functions ────────────────────────────────────────────────────────

def estimate_price(config: dict) -> dict:
    """
    Calculate a price estimate from a jewelry configuration.

    Expected config keys (all optional, defaults applied):
        jewelry_type    – "ring" | "necklace" | "earring" | "bracelet" (default "ring")
        metal           – key from METAL_PRICES (default "yellow_gold")
        center_stone    – key from GEMSTONE_PRICES (default "diamond")
        accent_stone    – key from GEMSTONE_PRICES (default "cubic_zirconia")
        metal_grams     – float (default from DEFAULT_WEIGHTS)
        center_carats   – float
        accent_carats   – float

    Returns:
        dict with keys: metal_cost, center_stone_cost, accent_stone_cost,
                        labor_cost, total, breakdown (list of line items).
    """
    jewelry_type = config.get("jewelry_type", "ring")
    defaults = DEFAULT_WEIGHTS.get(jewelry_type, DEFAULT_WEIGHTS["ring"])

    metal_key = config.get("metal", "yellow_gold")
    center_key = config.get("center_stone", "diamond")
    accent_key = config.get("accent_stone", "cubic_zirconia")

    metal_grams = float(config.get("metal_grams", defaults["metal_grams"]))
    center_carats = float(config.get("center_carats", defaults["center_carats"]))
    accent_carats = float(config.get("accent_carats", defaults["accent_carats"]))

    metal_data = METAL_PRICES.get(metal_key, METAL_PRICES["yellow_gold"])
    center_data = GEMSTONE_PRICES.get(center_key, GEMSTONE_PRICES["diamond"])
    accent_data = GEMSTONE_PRICES.get(accent_key, GEMSTONE_PRICES["cubic_zirconia"])

    metal_cost = metal_grams * metal_data["price_per_gram"]
    center_cost = center_carats * center_data["price_per_carat"]
    accent_cost = accent_carats * accent_data["price_per_carat"]
    materials_subtotal = metal_cost + center_cost + accent_cost
    labor_cost = materials_subtotal * LABOR_MARKUP
    total = materials_subtotal + labor_cost

    return {
        "metal_cost": round(metal_cost, 2),
        "center_stone_cost": round(center_cost, 2),
        "accent_stone_cost": round(accent_cost, 2),
        "labor_cost": round(labor_cost, 2),
        "total": round(total, 2),
        "breakdown": [
            {
                "label": f"Metal — {metal_data['display_name']} ({metal_grams:.1f} g)",
                "cost": round(metal_cost, 2),
            },
            {
                "label": f"Center stone — {center_data['display_name']} ({center_carats:.2f} ct)",
                "cost": round(center_cost, 2),
            },
            {
                "label": f"Accent stones — {accent_data['display_name']} ({accent_carats:.2f} ct)",
                "cost": round(accent_cost, 2),
            },
            {"label": "Labor & overhead (30 %)", "cost": round(labor_cost, 2)},
        ],
        "config": {
            "jewelry_type": jewelry_type,
            "metal": metal_key,
            "center_stone": center_key,
            "accent_stone": accent_key,
            "metal_grams": metal_grams,
            "center_carats": center_carats,
            "accent_carats": accent_carats,
        },
    }


def suggest_budget_alternatives(
    config: dict, budget: Optional[float] = None
) -> List[dict]:
    """
    Generate ranked cheaper alternatives by swapping expensive materials.

    Args:
        config: Same shape as the input to estimate_price().
        budget: Optional max price target; alternatives will be filtered to fit.

    Returns:
        List of dicts (sorted cheapest first), each with:
            description  – human-readable summary of the swap
            savings      – dollar amount saved vs original
            new_price    – estimated total after swap
            config_changes – dict of keys changed from original config
    """
    baseline = estimate_price(config)
    baseline_total = baseline["total"]

    alternatives = []

    # 1. Swap center stone to cheaper options
    for gem_key, gem_data in GEMSTONE_PRICES.items():
        if gem_key == config.get("center_stone", "diamond"):
            continue
        alt_config = {**config, "center_stone": gem_key}
        alt_price = estimate_price(alt_config)
        if alt_price["total"] < baseline_total:
            alternatives.append(
                {
                    "description": f"Replace center stone with {gem_data['display_name']}",
                    "savings": round(baseline_total - alt_price["total"], 2),
                    "new_price": alt_price["total"],
                    "config_changes": {"center_stone": gem_key},
                }
            )

    # 2. Swap metal to silver
    if config.get("metal", "yellow_gold") != "silver":
        alt_config = {**config, "metal": "silver"}
        alt_price = estimate_price(alt_config)
        alternatives.append(
            {
                "description": "Use Sterling Silver instead of precious metal",
                "savings": round(baseline_total - alt_price["total"], 2),
                "new_price": alt_price["total"],
                "config_changes": {"metal": "silver"},
            }
        )

    # 3. Reduce carat weight by 20 %
    orig_ct = float(config.get("center_carats", DEFAULT_WEIGHTS.get(config.get("jewelry_type", "ring"), {}).get("center_carats", 0.5)))
    if orig_ct > 0.1:
        alt_config = {**config, "center_carats": round(orig_ct * 0.8, 3)}
        alt_price = estimate_price(alt_config)
        alternatives.append(
            {
                "description": f"Reduce center stone to {alt_config['center_carats']:.2f} ct (−20 %)",
                "savings": round(baseline_total - alt_price["total"], 2),
                "new_price": alt_price["total"],
                "config_changes": {"center_carats": alt_config["center_carats"]},
            }
        )

    # 4. Full budget alternative: silver + cubic zirconia
    alt_config = {**config, "metal": "silver", "center_stone": "cubic_zirconia", "accent_stone": "cubic_zirconia"}
    alt_price = estimate_price(alt_config)
    alternatives.append(
        {
            "description": "Budget alternative: Silver + Cubic Zirconia",
            "savings": round(baseline_total - alt_price["total"], 2),
            "new_price": alt_price["total"],
            "config_changes": {"metal": "silver", "center_stone": "cubic_zirconia", "accent_stone": "cubic_zirconia"},
        }
    )

    # Sort by savings descending, then optionally filter by budget
    alternatives.sort(key=lambda x: x["savings"], reverse=True)
    if budget is not None:
        alternatives = [a for a in alternatives if a["new_price"] <= budget]

    return alternatives
