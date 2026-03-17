# JewelForge v2
import trimesh
import numpy as np


class MeshValidator:
    @staticmethod
    def validate(mesh_path):
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except Exception as e:
            return {
                "valid": False,
                "recommendation": "use_fallback",
                "issues": [str(e)],
                "metrics": {},
            }

        extents = sorted(mesh.bounding_box.extents)
        aspect = extents[0] / extents[2] if extents[2] > 0 else 0
        issues = []

        if aspect < 0.08:
            issues.append("flat_mesh")
        if len(mesh.faces) < 100:
            issues.append("too_few_faces")
        if not mesh.is_watertight:
            issues.append("not_watertight")

        if "flat_mesh" in issues:
            rec = "retry_with_rotation"
        elif "too_few_faces" in issues:
            rec = "use_fallback"
        else:
            rec = "accept"

        return {
            "valid": rec == "accept",
            "recommendation": rec,
            "issues": issues,
            "metrics": {
                "aspect_ratio": round(aspect, 3),
                "face_count": len(mesh.faces),
                "vertex_count": len(mesh.vertices),
            },
        }
