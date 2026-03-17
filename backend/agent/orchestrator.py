"""
JewelForge v2 — Agentic AI orchestrator powered by Claude.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import anthropic

# Imported lazily during tool execution to avoid heavy imports at startup
from materials.presets import METAL_PRESETS, GEMSTONE_PRESETS
from pricing.engine import estimate_price, suggest_budget_alternatives


class JewelForgeAgent:
    """
    Conversational AI agent for jewelry customization.

    Maintains multi-turn conversation history and dispatches tool calls to
    the JewelForge pipeline (segmentation, 3D generation, pricing, etc.).
    """

    SYSTEM_PROMPT = """You are JewelForge, an expert AI jewelry customization assistant.

You help users visualize, customize, and price jewelry in real time. You understand:
- Jewelry components: metal bands, shanks, prongs, bezels, halos, gemstone settings,
  center stones, accent stones, pavé, channel-set stones, side stones.
- Materials: yellow gold, white gold, rose gold, platinum, sterling silver;
  diamonds, rubies, sapphires, emeralds, amethyst, topaz, cubic zirconia.
- 3D pipeline: segmentation, mesh generation, texture baking, material assignment.
- Pricing: per-gram metal costs, per-carat gem costs, labor markup.

Workflow:
1. When the user uploads an image, call run_pipeline to segment it and generate a 3D mesh.
2. After the pipeline runs, explain which components were found.
3. When the user asks to change a material, call apply_material with the component and preset.
4. Keep price estimates updated; call estimate_price after any material change.
5. If the user asks about budget, call suggest_alternatives.
6. When the user wants to download, call export_model.

Tone: friendly, knowledgeable, concise. Use jewelry industry terminology but explain it naturally."""

    TOOLS: List[Dict[str, Any]] = [
        {
            "name": "run_pipeline",
            "description": "Run the full 2D→3D pipeline: preprocess image, segment components, generate 3D mesh, bake texture. Call this immediately after an image is uploaded.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the uploaded image file."},
                    "jewelry_type": {
                        "type": "string",
                        "enum": ["ring", "necklace", "earring", "bracelet", "auto"],
                        "description": "Type of jewelry. Use 'auto' to detect automatically.",
                    },
                },
                "required": ["image_path"],
            },
        },
        {
            "name": "apply_material",
            "description": "Apply a PBR material preset to a specific jewelry component in the 3D viewer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "component": {"type": "string", "description": "Component name (e.g. 'metal band', 'center stone')."},
                    "material_type": {"type": "string", "enum": ["metal", "gemstone"]},
                    "material_key": {"type": "string", "description": "Key from METAL_PRESETS or GEMSTONE_PRESETS."},
                },
                "required": ["component", "material_type", "material_key"],
            },
        },
        {
            "name": "estimate_price",
            "description": "Calculate the current price estimate for the jewelry configuration.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "jewelry_type": {"type": "string"},
                    "metal": {"type": "string"},
                    "center_stone": {"type": "string"},
                    "accent_stone": {"type": "string"},
                    "metal_grams": {"type": "number"},
                    "center_carats": {"type": "number"},
                    "accent_carats": {"type": "number"},
                },
            },
        },
        {
            "name": "suggest_alternatives",
            "description": "Generate cheaper material alternatives that reduce the price while maintaining quality.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "config": {"type": "object", "description": "Current jewelry config dict."},
                    "budget": {"type": "number", "description": "Optional max price target in USD."},
                },
                "required": ["config"],
            },
        },
        {
            "name": "identify_components",
            "description": "Return the list of detected jewelry components and their current material assignments.",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "export_model",
            "description": "Prepare the current 3D model for download.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["glb", "stl"], "description": "Export format."},
                },
                "required": ["format"],
            },
        },
    ]

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_state: Dict[str, Any] = {
            "job_id": None,
            "mesh_path": None,
            "labels_path": None,
            "components": [],
            "materials_applied": {},
            "jewelry_type": "auto",
            "price_estimate": None,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def process_message(
        self,
        user_message: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message, run any tool calls, and return a response.

        Args:
            user_message: Text from the user.
            image_path: Optional path to a newly uploaded image.

        Returns:
            dict with:
                response_text  – assistant's final text reply
                actions        – list of action dicts executed (tool name + result)
        """
        content: List[Any] = [{"type": "text", "text": user_message}]

        if image_path:
            import base64
            with open(image_path, "rb") as fh:
                b64 = base64.standard_b64encode(fh.read()).decode()
            ext = image_path.rsplit(".", 1)[-1].lower()
            media_type = "image/png" if ext == "png" else "image/jpeg"
            content.insert(0, {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })

        self.conversation_history.append({"role": "user", "content": content})

        actions: List[Dict[str, Any]] = []
        response_text = ""

        # Agentic loop: keep running while Claude returns tool_use blocks
        while True:
            response = self._client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=self._build_system_prompt(),
                tools=self.TOOLS,
                messages=self.conversation_history,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if text_blocks:
                response_text = " ".join(b.text for b in text_blocks)

            # Append assistant turn to history
            self.conversation_history.append(
                {"role": "assistant", "content": response.content}
            )

            if not tool_uses:
                break

            # Execute all tool calls and collect results
            tool_results = []
            for tool_use in tool_uses:
                result = self._execute_tool(tool_use.name, tool_use.input)
                actions.append({"tool": tool_use.name, "input": tool_use.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result),
                })

            self.conversation_history.append({"role": "user", "content": tool_results})

        return {"response_text": response_text, "actions": actions}

    # ── Tool dispatcher ────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        dispatch = {
            "run_pipeline": self._run_full_pipeline,
            "apply_material": self._apply_material,
            "estimate_price": self._estimate_price,
            "suggest_alternatives": self._suggest_alternatives,
            "identify_components": self._identify_components,
            "export_model": self._export_model,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(tool_input)
        except Exception as exc:
            return {"error": str(exc)}

    # ── Tool implementations ───────────────────────────────────────────────

    def _run_full_pipeline(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        from pipeline import run_full_pipeline, generate_job_id

        job_id = generate_job_id()
        output_dir = f"outputs/{job_id}"
        os.makedirs(output_dir, exist_ok=True)

        result = run_full_pipeline(
            image_path=inp["image_path"],
            output_dir=output_dir,
            jewelry_type=inp.get("jewelry_type", "auto"),
        )

        self.current_state.update({
            "job_id": job_id,
            "mesh_path": result.get("mesh_path"),
            "labels_path": result.get("labels_path"),
            "components": result.get("components", []),
            "jewelry_type": inp.get("jewelry_type", "auto"),
            "price_estimate": result.get("price"),
        })

        return result

    def _apply_material(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        component = inp["component"]
        material_key = inp["material_key"]
        material_type = inp["material_type"]

        presets = METAL_PRESETS if material_type == "metal" else GEMSTONE_PRESETS
        if material_key not in presets:
            return {"error": f"Unknown material key: {material_key}"}

        preset = presets[material_key]
        self.current_state["materials_applied"][component] = {
            "material_type": material_type,
            "material_key": material_key,
            "preset": preset,
        }

        return {
            "success": True,
            "component": component,
            "material": preset["name"],
            "preset": preset,
        }

    def _estimate_price(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        config = {**inp}
        if not config.get("jewelry_type"):
            config["jewelry_type"] = self.current_state.get("jewelry_type", "ring")

        # Infer metal + stone from applied materials
        for comp, mat in self.current_state.get("materials_applied", {}).items():
            if mat["material_type"] == "metal" and "metal" not in config:
                config["metal"] = mat["material_key"]
            elif mat["material_type"] == "gemstone" and "center_stone" not in config:
                config["center_stone"] = mat["material_key"]

        result = estimate_price(config)
        self.current_state["price_estimate"] = result
        return result

    def _suggest_alternatives(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        config = inp.get("config", {})
        budget = inp.get("budget")
        alts = suggest_budget_alternatives(config, budget)
        return {"alternatives": alts}

    def _identify_components(self, _inp: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "components": self.current_state.get("components", []),
            "materials_applied": self.current_state.get("materials_applied", {}),
            "jewelry_type": self.current_state.get("jewelry_type", "unknown"),
        }

    def _export_model(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        mesh_path = self.current_state.get("mesh_path")
        if not mesh_path or not os.path.exists(mesh_path):
            return {"error": "No mesh available. Run the pipeline first."}

        export_format = inp.get("format", "glb")
        if export_format == "glb":
            return {"download_url": f"/api/files/{self.current_state['job_id']}/mesh.glb"}

        # STL conversion
        import trimesh
        mesh = trimesh.load(mesh_path)
        stl_path = mesh_path.replace(".glb", ".stl")
        mesh.export(stl_path)
        return {"download_url": f"/api/files/{self.current_state['job_id']}/mesh.stl"}

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        state_summary = json.dumps(
            {
                "job_id": self.current_state["job_id"],
                "components_found": self.current_state["components"],
                "materials_applied": {
                    k: v["material_key"]
                    for k, v in self.current_state.get("materials_applied", {}).items()
                },
                "price_estimate": (
                    self.current_state["price_estimate"]["total"]
                    if self.current_state.get("price_estimate")
                    else None
                ),
            },
            indent=2,
        )
        return f"{self.SYSTEM_PROMPT}\n\n<current_state>\n{state_summary}\n</current_state>"
