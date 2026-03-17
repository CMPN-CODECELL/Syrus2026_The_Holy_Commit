# JewelForge v2

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are JewelForge AI, an expert jewelry customization assistant.

You help users customize 3D jewelry models by:
1. Applying materials (metals and gemstones) to jewelry components
2. Estimating and managing pricing within budgets
3. Suggesting creative combinations and budget alternatives

Current jewelry state is provided at the end of each user message in [State: ...] format.

When the user has a goal like "make this premium under $1000", you should:
1. Plan which materials to apply (think step by step)
2. Use apply_material for each component
3. Use estimate_price to verify the total
4. Confirm the goal was met in your response

Always be concise, helpful, and jewelry-savvy. Mention specific material names.
If a budget constraint cannot be met, explain why and suggest the closest alternative.

Available metals: yellow_gold, white_gold, rose_gold, platinum, silver, copper
Available gemstones: diamond, ruby, sapphire, emerald, amethyst, topaz, cubic_zirconia
"""

TOOLS = [
    {
        "name": "apply_material",
        "description": (
            "Apply a metal or gemstone material to a specific jewelry component. "
            "Use this to change the material of bands, settings, prongs, gemstones, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "description": (
                        "The component name to update (e.g. 'metal_band', 'gemstone_center', "
                        "'prong_0'). Must match a component from the current state."
                    ),
                },
                "material": {
                    "type": "string",
                    "description": (
                        "The material key to apply. Metals: yellow_gold, white_gold, rose_gold, "
                        "platinum, silver, copper. Gemstones: diamond, ruby, sapphire, emerald, "
                        "amethyst, topaz, cubic_zirconia."
                    ),
                },
            },
            "required": ["component", "material"],
        },
    },
    {
        "name": "estimate_price",
        "description": (
            "Calculate the current price based on all applied materials. "
            "Returns a detailed breakdown of material costs, labor, and total."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "suggest_alternatives",
        "description": (
            "Suggest budget-friendly material alternatives for the current configuration. "
            "Returns a ranked list of swaps with savings amounts. "
            "Use this when the user has a budget constraint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
