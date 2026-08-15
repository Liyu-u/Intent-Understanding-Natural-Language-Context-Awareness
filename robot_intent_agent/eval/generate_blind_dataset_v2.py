"""Generate the independent 120-case blind expansion set.

The set deliberately uses only the public pipeline input contract:
natural-language instruction + valid perception observation JSON.  Expected
answers are authored from scene IDs, never from model output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "blind_dataset_v2.json"

CATEGORIES = [
    "basic_action", "role_binding", "attribute_grounding", "spatial_disambiguation",
    "multi_object_ambiguity", "negation_obstacle", "numeric_constraint",
    "condition_sequence", "industrial_role", "missing_information",
    "safety_conflict", "colloquial_variant",
]


def object_record(eid: str, category: str, color: str, x: float, y: float,
                  *, surface: bool = False) -> dict:
    if surface:
        size = {"width": 0.45, "height": 0.05, "depth": 0.30}
        affordances = ["support_surface", "fixed", "container"]
    else:
        size = {"width": 0.06, "height": 0.08, "depth": 0.06}
        affordances = ["graspable", "movable"]
    return {
        "object_id": eid,
        "category_candidates": [{"name": category, "score": 0.98}],
        "appearance": {"color": color, "material": "plastic"},
        "pose": {"frame_id": "robot_base", "position": {"x": x, "y": y, "z": 0.05},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "geometry": {"size": size, "unit": "m"},
        "affordances": affordances,
        "confidence": 0.98,
    }


def observation(target: dict, destination: dict | None = None,
                obstacle: dict | None = None, extra: list[dict] | None = None) -> dict:
    objects = [target]
    if destination:
        objects.append(destination)
    if obstacle:
        objects.append(obstacle)
    objects.extend(extra or [])
    return {
        "frame_id": "robot_base", "unit": "m", "timestamp": "2026-08-11T12:00:00Z",
        "objects": objects,
    }


def case(index: int, category: str, instruction: str, obs: dict, *, action: str = "GRASP",
         theme: str | None = None, destination: str | None = None,
         obstacles: list[str] | None = None, status: str = "READY",
         skills: list[str] | None = None, constraints: list[dict] | None = None,
         sequence: bool = False) -> dict:
    allowed = status == "READY"
    expected = {
        "plan_status": status,
        "accepted_plan_statuses": [status] if allowed else ["NEEDS_CLARIFICATION", "BLOCKED"],
        "execution_allowed": allowed,
        "action": action,
        "theme_entity_id": theme if allowed else None,
        "destination_entity_id": destination if allowed else None,
        "obstacle_entity_ids": obstacles or [],
        "required_skills": skills or (["Reach", "Grasp"] if allowed else []),
        "constraints": constraints or [],
    }
    if sequence:
        expected.update(sequence_required=True, required_skill_order=["Grasp", "Place"])
    return {
        "case_id": f"V2_{index:04d}", "scene": "daily" if index % 2 else "industrial",
        "difficulty": "complex" if category in {"condition_sequence", "safety_conflict"} else "medium",
        "category": category, "instruction": instruction,
        "observation_json": obs, "expected": expected,
    }


def build() -> dict:
    cases: list[dict] = []
    n = 1
    for cat_index, category in enumerate(CATEGORIES):
        for local in range(10):
            target_id = f"v2_target_{n:04d}"
            dest_id = f"v2_dest_{n:04d}"
            obstacle_id = f"v2_obstacle_{n:04d}"
            color = ("red", "blue", "green", "yellow")[local % 4]
            target_cat = "workpiece" if cat_index % 2 else "cup"
            target = object_record(target_id, target_cat, color, 0.30, 0.10)
            tray_cat = "inspection_zone" if target_cat == "workpiece" else "tray"
            dest = object_record(dest_id, tray_cat, "gray", 0.65, -0.35, surface=True)
            obstacle = object_record(obstacle_id, "fixture", "orange", 0.48, 0.45, surface=True)
            extra = []
            status = "READY"
            action = "GRASP"
            instruction = f"grasp the {color} {target_cat}"
            skills = ["Reach", "Grasp"]
            theme = target_id
            destination = None
            obstacles: list[str] = []
            constraints: list[dict] = []
            sequence = False

            if category == "role_binding":
                instruction = f"place the {color} {target_cat} on the {tray_cat}"
                action, destination, skills = "PLACE", dest_id, ["Reach", "Place"]
            elif category == "attribute_grounding":
                instruction = f"grasp the {color} {target_cat}"
                extra.append(object_record(f"v2_other_{n:04d}", target_cat, "white", 0.42, 0.10))
            elif category == "spatial_disambiguation":
                target["pose"]["position"]["y"] = -0.30
                extra.append(object_record(f"v2_right_{n:04d}", target_cat, color, 0.30, 0.30))
                instruction = f"grasp the left {color} {target_cat}"
            elif category == "multi_object_ambiguity":
                extra.append(object_record(f"v2_other_{n:04d}", target_cat, color, 0.42, 0.10))
                if local == 8:
                    instruction = f"grasp the {target_cat}"
                    status, theme = "NEEDS_CLARIFICATION", None
                else:
                    instruction = f"grasp the {color} {target_cat}"
            elif category == "negation_obstacle":
                instruction = f"grasp the {color} {target_cat} while avoiding the fixture"
                obstacles, skills = [obstacle_id], ["PlanPath", "Reach", "Grasp"]
            elif category == "numeric_constraint":
                force = float(2 + local % 3)
                instruction = f"grasp the {color} {target_cat} with force no more than {force:g}N"
                constraints = [{"parameter": "force_n", "operator": "max", "value": force, "unit": "N"}]
            elif category == "condition_sequence":
                instruction = f"first grasp the {color} {target_cat}, then place it on the {tray_cat}"
                action, destination, skills, sequence = "PLACE", dest_id, ["Reach", "Grasp", "Place"], True
            elif category == "industrial_role":
                instruction = f"place the {color} workpiece in the inspection_zone"
                target["category_candidates"] = [{"name": "workpiece", "score": 0.98}]
                action, destination, skills = "PLACE", dest_id, ["Reach", "Place"]
            elif category == "missing_information":
                instruction = f"place the {color} {target_cat}"
                status, destination = "NEEDS_CLARIFICATION", None
            elif category == "safety_conflict":
                instruction = f"grasp and do not grasp the {color} {target_cat} at the same time"
                status, theme, skills = "BLOCKED", None, []
            elif category == "colloquial_variant":
                instruction = f"go ahead and pick up the {color} {target_cat}"

            if category == "multi_object_ambiguity" and local == 9:
                target["appearance"]["color"] = "blue"
                instruction = f"grasp the purple {target_cat}"
                status, theme = "NEEDS_CLARIFICATION", None

            obs = observation(target, dest if (category in {"role_binding", "condition_sequence", "industrial_role", "missing_information"}) else None,
                              obstacle if category == "negation_obstacle" else None, extra)
            cases.append(case(n, category, instruction, obs, action=action, theme=theme,
                              destination=destination, obstacles=obstacles, status=status,
                              skills=skills, constraints=constraints, sequence=sequence))
            n += 1

    payload = {"dataset_id": "blind_dataset_v2", "version": "2.0.0",
               "description": "Independent format-valid generalization blind set",
               "categories": CATEGORIES, "cases": cases}
    return payload


if __name__ == "__main__":
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": data["dataset_id"], "cases": len(data["cases"]),
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, ensure_ascii=False))
