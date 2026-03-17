# JewelForge v2
from materials.presets import METAL_PRESETS, GEMSTONE_PRESETS, DEFAULT_WEIGHTS


def estimate_price(config: dict) -> dict:
    """
    config keys:
      jewelry_type: str
      metal: str (metal key)
      metal_weight_grams: float (optional, uses default)
      gemstones: dict[component_name -> gemstone_key]
    """
    jtype = config.get("jewelry_type", "ring")
    defaults = DEFAULT_WEIGHTS.get(jtype, DEFAULT_WEIGHTS["ring"])

    metal_key = config.get("metal", "yellow_gold")
    metal_grams = config.get("metal_weight_grams", defaults["metal_grams"])
    metal_cost = METAL_PRESETS[metal_key]["price_per_gram"] * metal_grams

    gem_costs = {}
    total_gem = 0.0
    for comp, gem_key in config.get("gemstones", {}).items():
        carats = (
            defaults["center_stone_carats"]
            if "center" in comp
            else defaults["accent_stone_carats"]
        )
        cost = GEMSTONE_PRESETS[gem_key]["price_per_carat"] * carats
        gem_costs[comp] = {
            "type": gem_key,
            "carats": carats,
            "cost": round(cost, 2),
        }
        total_gem += cost

    labor = (metal_cost + total_gem) * 0.30
    return {
        "metal": {
            "type": metal_key,
            "grams": metal_grams,
            "cost": round(metal_cost, 2),
        },
        "gemstones": gem_costs,
        "labor_estimate": round(labor, 2),
        "total": round(metal_cost + total_gem + labor, 2),
    }


def suggest_budget_alternatives(config: dict) -> list:
    current_total = estimate_price(config)["total"]
    suggestions = []

    metal = config.get("metal", "yellow_gold")

    # Swap metal to silver
    if metal not in ("silver", "copper"):
        alt = {**config, "metal": "silver"}
        s = estimate_price(alt)["total"]
        suggestions.append({
            "description": f"{METAL_PRESETS[metal]['name']} → Silver",
            "new_total": s,
            "savings": round(current_total - s, 2),
            "changes": {"metal": "silver"},
        })

    # Swap each expensive gemstone to CZ
    for comp, gem in config.get("gemstones", {}).items():
        if gem in ("diamond", "ruby", "sapphire", "emerald"):
            alt_gems = {**config["gemstones"], comp: "cubic_zirconia"}
            alt = {**config, "gemstones": alt_gems}
            s = estimate_price(alt)["total"]
            suggestions.append({
                "description": f"{comp}: {gem} → CZ",
                "new_total": s,
                "savings": round(current_total - s, 2),
                "changes": {"gemstones": {comp: "cubic_zirconia"}},
            })

    # Full budget combo: silver + all CZ
    expensive_gems = [
        g for g in config.get("gemstones", {}).values()
        if g in ("diamond", "ruby", "sapphire", "emerald")
    ]
    if metal not in ("silver", "copper") and expensive_gems:
        alt_gems = {
            c: "cubic_zirconia" if g in ("diamond", "ruby", "sapphire", "emerald") else g
            for c, g in config.get("gemstones", {}).items()
        }
        alt = {**config, "metal": "silver", "gemstones": alt_gems}
        s = estimate_price(alt)["total"]
        suggestions.append({
            "description": "Full budget: Silver + CZ",
            "new_total": s,
            "savings": round(current_total - s, 2),
            "changes": {"metal": "silver", "gemstones": alt_gems},
        })

    return sorted(suggestions, key=lambda x: x["savings"], reverse=True)
