# JewelForge v2
import os
import uuid
import json
import anthropic

from .prompts import MODEL, SYSTEM_PROMPT, TOOLS
from .segmentation_agent import SegmentationAgent
from .mesh_validator import MeshValidator
from ..segment.gdino_sam2 import JewelrySegmenter
from ..gen3d.triposg import generate_3d_mesh
from ..texture.bake_and_project import MeshProjector
from ..pricing.engine import estimate_price, suggest_budget_alternatives
from ..materials.presets import is_metal, is_gemstone
from ..preprocess import preprocess_image

client = anthropic.Anthropic()


class JewelForgeAgent:
    """
    Three modes:
      1. Pipeline (autonomous self-correction, no LLM)
      2. Chat (LLM reasoning loop with tools)
      3. Direct swap (pure code, no LLM)
    """

    def __init__(self):
        self.state = {
            "job_id": None,
            "jewelry_type": None,
            "components": {},
            "materials_applied": {},
            "price": None,
            "mesh_path": None,
            "pipeline_log": [],
        }
        self.conversation_history = []

    # --- MODE 1: Pipeline (Agentic — self-correcting, no LLM) ---

    def run_pipeline(self, image_path, jewelry_type="auto"):
        job_id = str(uuid.uuid4())[:8]
        output_dir = f"outputs/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        log = []

        # Step 1: Preprocess
        prep = preprocess_image(image_path, output_dir)

        # Step 2: Segment (AGENTIC — self-correcting retry loop)
        segmenter = JewelrySegmenter()
        segmenter.load_models()
        seg_agent = SegmentationAgent(segmenter)
        seg_result = seg_agent.segment_with_retries(prep["rgba"], jewelry_type)
        segmenter.unload_models()
        log.append({
            "step": "segment",
            "attempts": seg_result.get("attempts"),
            "confidence": seg_result["confidence"],
            "warnings": seg_result["warnings"],
        })

        # Step 3: Identify components (deterministic heuristics)
        jtype = jewelry_type if jewelry_type != "auto" else "ring"
        components = self._identify_components(seg_result["segments"], jtype)

        # Step 4: Generate 3D (AGENTIC — validates and retries)
        mesh_path = None
        candidate = None
        for attempt in range(2):
            input_img = prep["rgb"]
            if attempt > 0:
                input_img = self._rotate_image(prep["rgb"], 15, output_dir)
                log.append({"step": "gen3d_retry", "reason": "flat mesh, rotating 15°"})

            candidate = generate_3d_mesh(input_img, output_dir)
            validation = MeshValidator.validate(candidate)
            log.append({"step": "validate", "attempt": attempt + 1, **validation})

            if validation["recommendation"] == "accept":
                mesh_path = candidate
                break
            elif validation["recommendation"] != "retry_with_rotation":
                mesh_path = candidate  # best we have
                break

        if mesh_path is None:
            mesh_path = candidate

        # Step 5: Texture bake + mask projection
        projector = MeshProjector()
        projector.load_mesh(mesh_path)
        texture_path = f"{output_dir}/texture.png"
        projector.bake_texture(prep["rgb"], texture_path)
        face_to_comp = projector.project_masks_to_faces(seg_result["segments"])
        glb_path, labels_path = projector.export_labeled_glb(
            face_to_comp, texture_path, f"{output_dir}/labeled_mesh.glb"
        )

        # Step 6: Set defaults + price
        for comp, info in components.items():
            self.state["materials_applied"][comp] = (
                "yellow_gold" if info["type"] == "metal" else "diamond"
            )
        self.state.update({
            "job_id": job_id,
            "jewelry_type": jtype,
            "components": components,
            "mesh_path": glb_path,
            "pipeline_log": log,
        })
        self.state["price"] = estimate_price(self._pricing_config())

        return {
            "job_id": job_id,
            "mesh_url": f"/api/files/{job_id}/labeled_mesh.glb",
            "labels_url": f"/api/files/{job_id}/labeled_mesh_labels.json",
            "components": components,
            "price": self.state["price"],
            "pipeline_log": log,
        }

    # --- MODE 2: Chat (Agentic — LLM reasoning loop) ---

    def chat(self, user_message):
        self.conversation_history.append({
            "role": "user",
            "content": user_message + f"\n\n[State: {self._state_summary()}]",
        })

        actions, response_text = [], ""

        for iteration in range(8):  # max iterations safety
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.conversation_history,
            )

            self.conversation_history.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            for tb in response.content:
                if tb.type == "text":
                    response_text += tb.text

            if not tool_uses:
                break  # LLM is done

            # Execute tools and feed results back
            tool_results = []
            for tu in tool_uses:
                result = self._execute_tool(tu.name, tu.input)
                actions.append({"type": tu.name, **tu.input})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, default=str),
                })

            self.conversation_history.append({"role": "user", "content": tool_results})
            # Loop continues — LLM sees results and decides: more tools or final answer

        return {
            "response_text": response_text,
            "actions": actions,
            "price": self.state.get("price"),
        }

    # --- MODE 3: Direct swap (No LLM) ---

    def direct_swap(self, component, material):
        if component not in self.state["components"]:
            return {"error": f"Unknown component: {component}"}
        self.state["materials_applied"][component] = material
        self.state["price"] = estimate_price(self._pricing_config())
        return {
            "success": True,
            "component": component,
            "material": material,
            "price": self.state["price"],
        }

    # --- Helpers ---

    def _execute_tool(self, name, inputs):
        if name == "apply_material":
            return self.direct_swap(inputs["component"], inputs["material"])
        elif name == "estimate_price":
            self.state["price"] = estimate_price(self._pricing_config())
            return {"price": self.state["price"]}
        elif name == "suggest_alternatives":
            return {"suggestions": suggest_budget_alternatives(self._pricing_config())}
        return {"error": f"Unknown tool: {name}"}

    def _identify_components(self, segments, jewelry_type):
        components = {}
        metal_idx = gem_idx = 0
        for seg in sorted(segments, key=lambda s: s.get("area_fraction", 0), reverse=True):
            label = seg.get("label", "").lower()
            if any(kw in label for kw in ["gem", "stone", "diamond", "crystal", "ruby", "sapphire", "emerald"]):
                name = "gemstone_center" if gem_idx == 0 else f"gemstone_accent_{gem_idx-1}"
                ctype = "gemstone"
                gem_idx += 1
            elif any(kw in label for kw in ["prong", "setting", "claw"]):
                name = f"prong_{metal_idx}"
                ctype = "metal"
                metal_idx += 1
            else:
                if metal_idx == 0:
                    name = "metal_band" if jewelry_type == "ring" else "metal_body"
                else:
                    name = f"metal_part_{metal_idx}"
                ctype = "metal"
                metal_idx += 1
            components[name] = {
                "segment_id": seg.get("segment_id"),
                "type": ctype,
                "detection_label": seg.get("label", ""),
                "area_fraction": seg.get("area_fraction", 0),
            }
        return components

    def _pricing_config(self):
        metal = "yellow_gold"
        gems = {}
        for comp, mat in self.state["materials_applied"].items():
            if is_metal(mat):
                metal = mat
            elif is_gemstone(mat):
                gems[comp] = mat
        return {
            "jewelry_type": self.state.get("jewelry_type", "ring"),
            "metal": metal,
            "gemstones": gems,
        }

    def _state_summary(self):
        if not self.state["components"]:
            return "No jewelry loaded."
        parts = [f"Type: {self.state.get('jewelry_type', '?')}"]
        for c in self.state["components"]:
            parts.append(f"{c}={self.state['materials_applied'].get(c, '?')}")
        if self.state["price"]:
            parts.append(f"Total: ${self.state['price']['total']:,.2f}")
        return " | ".join(parts)

    def _rotate_image(self, path, angle, output_dir):
        from PIL import Image
        img = Image.open(path).rotate(angle, expand=False, fillcolor=(200, 200, 200))
        out = f"{output_dir}/preprocessed_rotated.png"
        img.save(out)
        return out
