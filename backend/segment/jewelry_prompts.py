"""
Grounding DINO text prompts for jewelry component detection.
"""

from typing import List

JEWELRY_PROMPTS: dict = {
    "ring": [
        "metal band",
        "gemstone",
        "center stone",
        "accent stone",
        "prong",
        "setting",
        "halo",
        "shank",
        "side stone",
        "bezel",
    ],
    "necklace": [
        "chain",
        "pendant",
        "gemstone",
        "clasp",
        "bail",
        "center stone",
        "metal link",
        "charm",
    ],
    "earring": [
        "stud",
        "gemstone",
        "post",
        "hoop",
        "drop",
        "dangle",
        "metal body",
        "center stone",
        "back",
    ],
    "bracelet": [
        "metal band",
        "chain",
        "gemstone",
        "clasp",
        "link",
        "charm",
        "center stone",
        "accent stone",
    ],
}


def get_prompts_for_type(jewelry_type: str) -> List[str]:
    """
    Return the detection prompt list for the given jewelry type.

    Args:
        jewelry_type: One of "ring", "necklace", "earring", "bracelet", or "auto".

    Returns:
        List of component name strings suitable for Grounding DINO text queries.
        If "auto" is requested, returns the deduplicated union of all types.
    """
    if jewelry_type == "auto":
        combined: List[str] = []
        seen: set = set()
        for prompts in JEWELRY_PROMPTS.values():
            for p in prompts:
                if p not in seen:
                    combined.append(p)
                    seen.add(p)
        return combined

    return JEWELRY_PROMPTS.get(jewelry_type, JEWELRY_PROMPTS["ring"])
