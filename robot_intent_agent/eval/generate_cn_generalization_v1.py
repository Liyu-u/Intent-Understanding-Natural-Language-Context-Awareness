"""Generate a Chinese-only daily/industrial generalization set.

This dataset isolates natural-language understanding.  Every observation is
format-valid, internally consistent, and contains the referenced entities.
No case relies on perception errors or malformed JSON.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "cn_generalization_v1.json"
CATEGORIES = [
    "同义动作", "角色与目的地", "属性组合接地", "空间关系消歧",
    "多对象指代", "否定与障碍物", "数值约束", "条件与顺序",
    "工业角色映射", "省略与上下文", "安全矛盾", "口语自然表达",
]


def obj(eid: str, category: str, color: str, x: float, y: float,
        *, surface: bool = False, material: str = "plastic") -> dict:
    return {
        "object_id": eid,
        "category_candidates": [{"name": category, "score": 0.98}],
        "appearance": {"color": color, "material": material},
        "pose": {"frame_id": "robot_base", "position": {"x": x, "y": y, "z": 0.05},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "geometry": {"size": {"width": 0.06 if not surface else 0.45,
                                "height": 0.08 if not surface else 0.05,
                                "depth": 0.06 if not surface else 0.30}, "unit": "m"},
        "affordances": ["graspable", "movable"] if not surface else
                       ["support_surface", "fixed", "container"],
        "confidence": 0.98,
    }


def obs(objects: list[dict]) -> dict:
    return {"frame_id": "robot_base", "unit": "m",
            "timestamp": "2026-08-12T12:00:00Z", "objects": objects}


def make_case(i: int, category: str, instruction: str, scene: dict,
              *, action: str = "GRASP", theme: str | None = None,
              destination: str | None = None, obstacles: list[str] | None = None,
              status: str = "READY", skills: list[str] | None = None,
              constraints: list[dict] | None = None, sequence: bool = False) -> dict:
    ready = status == "READY"
    expected = {
        "plan_status": status,
        "accepted_plan_statuses": [status] if ready else ["NEEDS_CLARIFICATION", "BLOCKED"],
        "execution_allowed": ready,
        "action": action,
        "theme_entity_id": theme if ready else None,
        "destination_entity_id": destination if ready else None,
        "obstacle_entity_ids": obstacles or [],
        "required_skills": skills or (["Reach", "Grasp"] if ready else []),
        "constraints": constraints or [],
    }
    if sequence:
        expected.update(sequence_required=True, required_skill_order=["Grasp", "Place"])
    return {"case_id": f"CN_{i:04d}", "scene": "daily" if i % 2 else "industrial",
            "difficulty": "complex" if category in {"条件与顺序", "安全矛盾"} else "medium",
            "category": category, "instruction": instruction,
            "observation_json": scene, "expected": expected}


def build() -> dict:
    cases = []
    i = 1
    colors = [("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("黄色", "yellow")]
    for cat_index, category in enumerate(CATEGORIES):
        for local in range(10):
            tid = f"cn_target_{i:04d}"
            did = f"cn_destination_{i:04d}"
            oid = f"cn_obstacle_{i:04d}"
            color_cn, color = colors[local % len(colors)]
            target_cat = "workpiece" if category == "工业角色映射" else "cup"
            target_cn = "工件" if target_cat == "workpiece" else "杯子"
            target = obj(tid, target_cat, color, 0.30, 0.10,
                         material="metal" if target_cat == "workpiece" else "plastic")
            tray_cat = "inspection_zone" if target_cat == "workpiece" else "tray"
            tray_cn = "检测区" if tray_cat == "inspection_zone" else "托盘"
            dest = obj(did, tray_cat, "gray", 0.65, -0.35, surface=True)
            obstacle = obj(oid, "fixture", "orange", 0.48, 0.45, surface=True)
            objects = [target]
            instruction = f"把{color_cn}{target_cn}拿起来"
            action, theme, destination = "GRASP", tid, None
            obstacles, skills, constraints = [], ["Reach", "Grasp"], []
            status, sequence = "READY", False

            if category == "同义动作":
                instruction = [f"把{color_cn}{target_cn}拿起来", f"请抓取{color_cn}{target_cn}",
                               f"拿起那个{color_cn}{target_cn}", f"帮我取起{color_cn}{target_cn}",
                               f"将{color_cn}{target_cn}抓稳"][local % 5]
            elif category == "角色与目的地":
                instruction = [f"把{color_cn}{target_cn}放到{tray_cn}上", f"将{color_cn}{target_cn}放入{tray_cn}",
                               f"拿起{color_cn}{target_cn}再放到{tray_cn}里"][local % 3]
                objects.append(dest)
                action, destination, skills = "PLACE", did, ["Reach", "Place"]
            elif category == "属性组合接地":
                target["appearance"]["material"] = "glass"
                target["category_candidates"] = [{"name": "glass_cup", "score": 0.98}]
                target_cat, target_cn = "glass_cup", "玻璃杯"
                instruction = f"拿起{color_cn}{target_cn}"
                objects.append(obj(f"cn_distractor_{i:04d}", "cup", "white", 0.42, 0.10))
            elif category == "空间关系消歧":
                target["pose"]["position"]["y"] = -0.30
                objects.append(obj(f"cn_right_{i:04d}", "cup", color, 0.30, 0.30))
                instruction = f"拿起左边的{color_cn}杯子"
            elif category == "多对象指代":
                objects.append(obj(f"cn_other_{i:04d}", "cup", "white", 0.42, 0.10))
                if local == 8:
                    instruction, status, theme = "把杯子拿起来", "NEEDS_CLARIFICATION", None
                else:
                    instruction = f"别拿白色的，拿起{color_cn}杯子"
                    obstacles, skills = [f"cn_other_{i:04d}"], ["PlanPath", "Reach", "Grasp"]
            elif category == "否定与障碍物":
                objects.append(obstacle)
                instruction = f"拿起{color_cn}杯子，避开夹具，不要碰到它"
                obstacles, skills = [oid], ["PlanPath", "Reach", "Grasp"]
            elif category == "数值约束":
                force = float(2 + local % 3)
                instruction = f"用不超过{force:g}N的力量拿起{color_cn}杯子"
                constraints = [{"parameter": "force_n", "operator": "max", "value": force, "unit": "N"}]
            elif category == "条件与顺序":
                objects.append(dest)
                instruction = f"先拿起{color_cn}杯子，完成后再把它放到托盘上"
                action, destination, skills, sequence = "PLACE", did, ["Reach", "Grasp", "Place"], True
            elif category == "工业角色映射":
                objects.append(dest)
                instruction = f"把{color_cn}工件上料到检测区"
                action, destination, skills = "PLACE", did, ["Reach", "Place"]
            elif category == "省略与上下文":
                objects.append(dest)
                instruction = f"先抓{color_cn}杯子，再把它放到托盘里"
                action, destination, skills, sequence = "PLACE", did, ["Reach", "Grasp", "Place"], True
            elif category == "安全矛盾":
                instruction = f"拿起{color_cn}杯子，同时保持它没有被拿起"
                status, theme, skills = "BLOCKED", None, []
            elif category == "口语自然表达":
                instruction = [f"麻烦把{color_cn}杯子递到我这边", f"帮我把{color_cn}杯子拿过来",
                               f"把{color_cn}杯子给我拿来", f"劳驾取一下{color_cn}杯子"][local % 4]
                if local % 4 < 3:
                    action = "FETCH"
                    # The language identifies a recipient, but the valid scene
                    # snapshot intentionally contains no recipient pose/zone.
                    # Correct behavior is clarification, not fabricated delivery.
                    status, theme, skills = "NEEDS_CLARIFICATION", None, []
                else:
                    action, skills = "GRASP", ["Reach", "Grasp"]

            cases.append(make_case(i, category, instruction, obs(objects), action=action,
                                   theme=theme, destination=destination, obstacles=obstacles,
                                   status=status, skills=skills, constraints=constraints,
                                   sequence=sequence))
            i += 1
    return {"dataset_id": "cn_generalization_v1", "version": "1.0.0",
            "language": "zh-CN", "scenes": ["daily", "industrial"],
            "categories": CATEGORIES, "cases": cases}


if __name__ == "__main__":
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": data["dataset_id"], "cases": len(data["cases"]),
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, ensure_ascii=False))
