"""
Agent system prompt and tool definitions — Gemini format.
"""
from materials.presets import METAL_PRESETS, GEMSTONE_PRESETS, get_all_material_keys

_metal_list = "\n".join(
    f"  - {key}: {p['name']} (${p['price_per_gram']:.2f}/gram)"
    for key, p in METAL_PRESETS.items()
)
_gem_list = "\n".join(
    f"  - {key}: {p['name']} (${p['price_per_carat']:.2f}/carat)"
    for key, p in GEMSTONE_PRESETS.items()
)

SYSTEM_PROMPT = f"""You are JewelForge AI, a jewelry customization assistant.

You help users customize 3D jewelry by changing materials and estimating prices.

When the user gives you a goal (like "make this premium under $1000"), you should:
1. THINK about what combinations achieve the goal
2. EXECUTE changes using apply_material
3. CHECK the price using estimate_price
4. ADJUST if the result doesn't meet the goal

You can call multiple tools in sequence for complex goals.

Available metals:
{_metal_list}

Available gemstones:
{_gem_list}

Component names: metal_band, metal_body, gemstone_center, gemstone_accent_0, prong

Rules:
1. After EVERY material change, call estimate_price.
2. If the user gives a budget, verify the total is under budget before responding.
3. Be concise. Lead with what you did and the new price.
4. "Premium" = platinum/white gold + diamond/sapphire. "Budget" = silver/copper + CZ/amethyst.
"""

# Gemini function declarations format
TOOL_FUNCTIONS = {
    "apply_material": {
        "description": (
            "Change the material of a specific component on the current 3D model. "
            "Call when user wants to change metal or gemstone type."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "description": "Component name (e.g. 'metal_band', 'gemstone_center')",
                },
                "material": {
                    "type": "string",
                    "description": f"Material key. One of: {', '.join(get_all_material_keys())}",
                    "enum": get_all_material_keys(),
                },
            },
            "required": ["component", "material"],
        },
    },
    "estimate_price": {
        "description": "Get current price estimate. Call after every material change.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    "suggest_alternatives": {
        "description": "Get ranked budget-friendly alternatives for current config.",
        "parameters": {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "number",
                    "description": "Target budget in USD (optional)",
                },
            },
        },
    },
}
