"""
JewelForge v2 — Agentic Orchestrator (Gemini version)

Three modes:
  1. Pipeline (autonomous self-correction, no LLM)
  2. Chat (Gemini reasoning loop with function calling)
  3. Direct swap (pure code, no LLM)
"""
import json
import os
import uuid
import shutil
import numpy as np

from google import genai
from google.genai import types

from agent.prompts import SYSTEM_PROMPT, TOOL_FUNCTIONS
from pricing.engine import estimate_price, suggest_budget_alternatives
from materials.presets import (
    METAL_PRESETS, GEMSTONE_PRESETS, is_metal, is_gemstone,
    get_preset, DEFAULT_WEIGHTS, get_all_material_keys
)

MODEL = "gemini-2.5-flash"
_client = None


def _load_local_env():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = [
        os.path.join(repo_root, ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]

    for env_path in candidates:
        if not os.path.exists(env_path):
            continue

        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_client():
    global _client
    _load_local_env()
    if _client is None:
        _client = genai.Client()
    return _client


class JewelForgeAgent:
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
        self.chat_session = None

    # -----------------------------------------------------------------
    # MODE 1: Pipeline (Agentic — self-correcting, no LLM)
    # -----------------------------------------------------------------

    def _demo_output_paths(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        outputs_dir = os.path.join(base_dir, "outputs")
        if not os.path.isdir(outputs_dir):
            return None, None

        for entry in sorted(os.listdir(outputs_dir)):
            candidate_dir = os.path.join(outputs_dir, entry)
            glb_path = os.path.join(candidate_dir, "labeled_mesh.glb")
            labels_path = os.path.join(candidate_dir, "labeled_mesh_labels.json")
            if os.path.exists(glb_path) and os.path.exists(labels_path):
                return glb_path, labels_path

        return None, None

    def _mock_pipeline_result(
        self,
        image_path: str,
        jewelry_type: str = "auto",
        job_id: str | None = None,
        output_dir: str | None = None,
        reason: str = "ML models not available",
    ) -> dict:
        job_id = job_id or str(uuid.uuid4())[:8]
        jtype = jewelry_type if jewelry_type != "auto" else "ring"
        components = {
            "metal_band": {"segment_id": 0, "type": "metal", "detection_label": "metal band", "area_fraction": 0.3},
            "gemstone_center": {"segment_id": 1, "type": "gemstone", "detection_label": "gemstone", "area_fraction": 0.1},
        }
        mesh_url = None
        labels_url = None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            demo_glb, demo_labels = self._demo_output_paths()
            if demo_glb and demo_labels:
                shutil.copyfile(demo_glb, os.path.join(output_dir, "labeled_mesh.glb"))
                shutil.copyfile(demo_labels, os.path.join(output_dir, "labeled_mesh_labels.json"))
                mesh_url = f"/api/files/{job_id}/labeled_mesh.glb"
                labels_url = f"/api/files/{job_id}/labeled_mesh_labels.json"

        for comp, info in components.items():
            self.state["materials_applied"][comp] = "yellow_gold" if info["type"] == "metal" else "diamond"
        self.state.update({
            "job_id": job_id, "jewelry_type": jtype,
            "components": components,
            "mesh_path": os.path.join(output_dir, "labeled_mesh.glb") if output_dir and mesh_url else None,
            "pipeline_log": [{"step": "mock", "reason": reason}],
        })
        self.state["price"] = estimate_price(self._pricing_config())
        return {
            "job_id": job_id,
            "mesh_url": mesh_url,
            "labels_url": labels_url,
            "components": components,
            "price": self.state["price"],
            "pipeline_log": self.state["pipeline_log"],
            "mock": True,
            "message": "Using mock data — ML pipeline not available",
        }

    def run_pipeline(
        self,
        image_path: str,
        jewelry_type: str = "auto",
        job_id: str | None = None,
    ) -> dict:
        job_id = job_id or str(uuid.uuid4())[:8]
        output_dir = f"outputs/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        log = []

        try:
            # Step 1: Preprocess
            try:
                from preprocess import preprocess_image
            except Exception as exc:
                return self._mock_pipeline_result(
                    image_path,
                    jewelry_type,
                    job_id=job_id,
                    output_dir=output_dir,
                    reason=f"Preprocess import failed: {exc}",
                )

            prep = preprocess_image(image_path, output_dir)

            # Step 2: Segment (AGENTIC — self-correcting retry loop)
            try:
                from segment.gdino_sam2 import JewelrySegmenter
                from agent.segmentation_agent import SegmentationAgent
            except Exception as exc:
                return self._mock_pipeline_result(
                    image_path,
                    jewelry_type,
                    job_id=job_id,
                    output_dir=output_dir,
                    reason=f"Segmentation import failed: {exc}",
                )

            segmenter = JewelrySegmenter()
            segmenter.load_models()
            seg_agent = SegmentationAgent(segmenter)
            seg_result = seg_agent.segment_with_retries(prep["rgba"], jewelry_type)
            segmenter.unload_models()
            log.append({
                "step": "segment",
                "confidence": seg_result["confidence"],
                "warnings": seg_result["warnings"],
                "segment_count": len(seg_result["segments"]),
            })

            # Step 3: Identify components
            jtype = jewelry_type if jewelry_type != "auto" else "ring"
            components = self._identify_components(seg_result["segments"], jtype)

            # Step 4: Generate 3D (AGENTIC — validates and retries)
            try:
                from gen3d.triposg import generate_3d_mesh
                from agent.mesh_validator import MeshValidator
            except Exception as exc:
                return self._mock_pipeline_result(
                    image_path,
                    jewelry_type,
                    job_id=job_id,
                    output_dir=output_dir,
                    reason=f"3D generation import failed: {exc}",
                )

            mesh_path = None
            for attempt in range(2):
                input_img = prep["rgb"]
                if attempt > 0:
                    input_img = self._rotate_image(prep["rgb"], 15, output_dir)
                    log.append({"step": "gen3d_retry", "reason": "flat mesh"})

                candidate = generate_3d_mesh(input_img, output_dir)
                validation = MeshValidator.validate(candidate)
                log.append({"step": "validate", "attempt": attempt + 1, **validation})

                if validation["recommendation"] == "accept":
                    mesh_path = candidate
                    break
                if validation["recommendation"] != "retry_with_rotation":
                    mesh_path = candidate
                    break
            if mesh_path is None:
                mesh_path = candidate

            # Step 5: Texture + mask projection
            try:
                from texture.bake_and_project import MeshProjector
            except Exception as exc:
                return self._mock_pipeline_result(
                    image_path,
                    jewelry_type,
                    job_id=job_id,
                    output_dir=output_dir,
                    reason=f"Texture projection import failed: {exc}",
                )

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
                "job_id": job_id, "jewelry_type": jtype,
                "components": components, "mesh_path": glb_path,
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
        except Exception as exc:
            return self._mock_pipeline_result(
                image_path,
                jewelry_type,
                job_id=job_id,
                output_dir=output_dir,
                reason=f"Pipeline execution failed: {exc}",
            )

    # -----------------------------------------------------------------
    # MODE 2: Chat (Agentic — Gemini function-calling loop)
    # -----------------------------------------------------------------

    def chat(self, user_message: str) -> dict:
        """
        Gemini function-calling loop. Gemini calls tools, we execute them,
        feed results back, Gemini decides whether to call more or respond.
        """
        # Build tools list for Gemini
        function_declarations = []
        for name, spec in TOOL_FUNCTIONS.items():
            function_declarations.append(types.FunctionDeclaration(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
            ))

        tools = [types.Tool(function_declarations=function_declarations)]

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
        )

        # Build conversation with state context
        state_ctx = self._state_summary()
        enriched_message = user_message + f"\n\n[Current state: {state_ctx}]"

        # Initialize or continue chat session
        if self.chat_session is None:
            self.chat_session = _get_client().chats.create(model=MODEL, config=config)

        actions = []
        response_text = ""

        # Send message — Gemini may return function calls
        response = self.chat_session.send_message(enriched_message)

        # Agentic loop: keep going while Gemini wants to call functions
        max_iterations = 8
        for iteration in range(max_iterations):
            # Check if response contains function calls
            has_function_calls = False
            function_responses = []

            for part in response.candidates[0].content.parts:
                if part.function_call:
                    has_function_calls = True
                    fc = part.function_call
                    fc_name = fc.name
                    fc_args = dict(fc.args) if fc.args else {}

                    # Execute the tool
                    result = self._execute_tool(fc_name, fc_args)
                    actions.append({"type": fc_name, **fc_args})

                    # Build function response to send back
                    function_responses.append(types.Part.from_function_response(
                        name=fc_name,
                        response={"result": result},
                    ))

                elif part.text:
                    response_text += part.text

            if not has_function_calls:
                break  # Gemini is done — final text response

            # Send function results back to Gemini
            response = self.chat_session.send_message(function_responses)

        return {
            "response_text": response_text,
            "actions": actions,
            "price": self.state.get("price"),
        }

    # -----------------------------------------------------------------
    # MODE 3: Direct swap (No LLM — pure code)
    # -----------------------------------------------------------------

    def direct_swap(self, component: str, material: str) -> dict:
        if component not in self.state["components"]:
            return {"error": f"Unknown component: {component}"}
        self.state["materials_applied"][component] = material
        self.state["price"] = estimate_price(self._pricing_config())
        return {
            "success": True,
            "component": component,
            "material": material,
            "material_name": get_preset(material)["name"],
            "price": self.state["price"],
        }

    # -----------------------------------------------------------------
    # Tool implementations
    # -----------------------------------------------------------------

    def _execute_tool(self, name: str, inputs: dict) -> dict:
        try:
            if name == "apply_material":
                return self.direct_swap(inputs["component"], inputs["material"])
            elif name == "estimate_price":
                self.state["price"] = estimate_price(self._pricing_config())
                return {"price": self.state["price"]}
            elif name == "suggest_alternatives":
                alts = suggest_budget_alternatives(self._pricing_config())
                budget = inputs.get("budget")
                if budget:
                    alts = [a for a in alts if a["new_total"] <= budget]
                return {"suggestions": alts[:5]}
            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _identify_components(self, segments, jewelry_type):
        components = {}
        metal_idx = gem_idx = 0
        for seg in sorted(segments, key=lambda s: s.get("area_fraction", 0), reverse=True):
            label = seg.get("label", "").lower()
            if any(kw in label for kw in ["gem", "stone", "diamond", "crystal",
                                           "ruby", "sapphire", "emerald"]):
                name = "gemstone_center" if gem_idx == 0 else f"gemstone_accent_{gem_idx - 1}"
                ctype = "gemstone"
                gem_idx += 1
            elif any(kw in label for kw in ["prong", "setting", "claw"]):
                name = f"prong_{metal_idx}"
                ctype = "metal"
                metal_idx += 1
            else:
                name = ("metal_band" if jewelry_type == "ring" else "metal_body") \
                    if metal_idx == 0 else f"metal_part_{metal_idx}"
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
