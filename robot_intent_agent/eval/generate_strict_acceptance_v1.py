"""Generate the frozen 600-case strict downstream acceptance dataset.

The generator is deterministic.  Every case carries explicit input-validity
metadata and exact downstream expectations; malformed-input cases are marked
intentional and are never confused with dataset defects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent
OUT = ROOT / "strict_acceptance_v1_1.json"

CATEGORY_COUNTS = {
    "basic_action": 50,
    "role_binding": 35,
    "attribute_spatial": 40,
    "multi_object_ambiguity": 30,
    "negation_obstacle": 30,
    "numeric_constraints": 25,
    "condition_sequence": 30,
    "missing_conflict_safety": 25,
    "colloquial_noise": 20,
    "context_reference": 15,
}

SCENES = {
    "daily": {
        "targets": [("cup", "杯子"), ("bottle", "瓶子"), ("box", "盒子"), ("book", "书"), ("medicine_bottle", "药瓶")],
        "destinations": [("tray", "托盘"), ("table", "桌子"), ("cabinet", "柜子"), ("bin", "收纳箱")],
        "obstacles": [("glass", "玻璃杯"), ("vase", "花瓶"), ("hot_kettle", "热水壶")],
    },
    "industrial": {
        "targets": [("workpiece", "工件"), ("part", "零件"), ("bearing", "轴承"), ("gear", "齿轮"), ("component", "组件")],
        "destinations": [("tray", "托盘"), ("inspection_zone", "检测区"), ("parts_bin", "料箱"), ("workbench", "工位")],
        "obstacles": [("welding_zone", "焊接区"), ("fixture", "夹具"), ("hot_surface", "高温台")],
    },
}

COLORS = [("red", "红色"), ("blue", "蓝色"), ("green", "绿色"), ("yellow", "黄色")]

READY_COUNTS = {
    "basic_action": 35, "role_binding": 24, "attribute_spatial": 28,
    "multi_object_ambiguity": 21, "negation_obstacle": 21,
    "numeric_constraints": 17, "condition_sequence": 21,
    "missing_conflict_safety": 18, "colloquial_noise": 14,
    "context_reference": 11,
}
CLARIFY_COUNTS = {
    "basic_action": 8, "role_binding": 5, "attribute_spatial": 6,
    "multi_object_ambiguity": 5, "negation_obstacle": 5,
    "numeric_constraints": 4, "condition_sequence": 5,
    "missing_conflict_safety": 3, "colloquial_noise": 2,
    "context_reference": 2,
}


def obj(object_id: str, category: str, color: str, x: float, *, y: float = 0.15, support=False):
    size_by_category = {
        "table": (0.80, 0.72, 0.60), "workbench": (1.00, 0.80, 0.70),
        "inspection_zone": (0.60, 0.04, 0.50), "welding_zone": (0.80, 0.05, 0.80),
        "tray": (0.35, 0.04, 0.25), "parts_bin": (0.40, 0.25, 0.30),
        "bin": (0.40, 0.25, 0.30), "cabinet": (0.70, 1.20, 0.45),
        "fixture": (0.25, 0.15, 0.20), "hot_surface": (0.50, 0.08, 0.40),
    }
    width, height, depth = size_by_category.get(category, (0.06, 0.08, 0.06))
    material = ("glass" if category == "glass" else "metal" if category in {
        "workpiece", "part", "bearing", "gear", "component", "fixture", "hot_surface"
    } else "plastic")
    affordances = ["support_surface", "container", "fixed"] if support else ["graspable", "movable"]
    return {
        "object_id": object_id,
        "category_candidates": [{"name": category, "score": 0.98}],
        "appearance": {"color": color, "material": material},
        "pose": {"frame_id": "robot_base", "position": {"x": x, "y": y, "z": 0.05},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "geometry": {"size": {"width": width, "height": height, "depth": depth}, "unit": "m"},
        "affordances": affordances,
        "confidence": 0.98,
    }


def make_case(scene_name: str, category: str, local_i: int, global_i: int):
    spec = SCENES[scene_name]
    target_cat, target_cn = spec["targets"][local_i % len(spec["targets"])]
    dest_cat, dest_cn = spec["destinations"][local_i % len(spec["destinations"])]
    obs_cat, obs_cn = spec["obstacles"][local_i % len(spec["obstacles"])]
    color, color_cn = COLORS[local_i % len(COLORS)]
    prefix = "D" if scene_name == "daily" else "I"
    target_id, dest_id, obstacle_id = f"{prefix}_target_{global_i:03d}", f"{prefix}_dest_{global_i:03d}", f"{prefix}_obs_{global_i:03d}"

    # Exact per-scene totals: 210 READY, 45 clarification, 45 blocked.
    outcome = ("READY" if local_i < READY_COUNTS[category]
               else "NEEDS_CLARIFICATION" if local_i < READY_COUNTS[category] + CLARIFY_COUNTS[category]
               else "BLOCKED")
    # Difficulty is based on semantic composition, not file position. Totals:
    # 180 simple, 300 medium, 120 complex across both scenes.
    if category in {"basic_action", "attribute_spatial"}:
        difficulty = "simple"
    elif category in {"condition_sequence", "context_reference"}:
        difficulty = "complex"
    elif category == "colloquial_noise":
        difficulty = "medium" if local_i < 5 else "complex"
    else:
        difficulty = "medium"

    # Non-mentioned scene objects are kept away from the direct reach corridor;
    # otherwise geometry would create an accidental obstacle and a dataset defect.
    objects = [obj(target_id, target_cat, color, 0.35), obj(dest_id, dest_cat, "gray", 0.65, y=-0.45, support=True),
               obj(obstacle_id, obs_cat, "orange", 0.50, y=0.55)]
    expected = {
        "plan_status": outcome,
        "execution_allowed": outcome == "READY",
        "action": "GRASP",
        "theme_entity_id": target_id if outcome == "READY" else None,
        "destination_entity_id": None,
        "obstacle_entity_ids": [],
        "required_skills": ["Reach", "Grasp"] if outcome == "READY" else [],
        "constraints": [],
    }

    if outcome == "NEEDS_CLARIFICATION":
        second_id = f"{prefix}_target_alt_{global_i:03d}"
        objects.append(obj(second_id, target_cat, color, 0.42))
        instruction = [f"把{target_cn}拿起来", f"请拿一下{target_cn}", f"抓取那个{target_cn}",
                       f"帮我拿起{target_cn}", f"把现场的{target_cn}取起来"][local_i % 5]
    elif outcome == "BLOCKED":
        objects = [o for o in objects if o["object_id"] != target_id]
        instruction = [f"把现场不存在的紫色{target_cn}拿起来", f"请抓取紫色{target_cn}",
                       f"拿起感知里没有的紫色{target_cn}", f"把那个紫色{target_cn}取起来",
                       f"帮我找出并拿起紫色{target_cn}"][local_i % 5]
    else:
        instruction = [f"把{color_cn}{target_cn}拿起来", f"请抓取{color_cn}{target_cn}",
                       f"拿起那个{color_cn}{target_cn}", f"将{color_cn}{target_cn}抓稳",
                       f"帮我取起{color_cn}{target_cn}"][local_i % 5]

    if category == "role_binding" and outcome == "READY":
        instruction = [f"把{color_cn}{target_cn}放到{dest_cn}里", f"请将{color_cn}{target_cn}放进{dest_cn}",
                       f"拿起{color_cn}{target_cn}并放到{dest_cn}中", f"把{color_cn}{target_cn}摆放在{dest_cn}里",
                       f"将{color_cn}{target_cn}抓起后放入{dest_cn}"][local_i % 5]
        expected.update(action="PLACE", destination_entity_id=dest_id, required_skills=["Reach", "Place"])
    elif category == "attribute_spatial" and outcome == "READY":
        instruction = [f"拿起左侧的{color_cn}{target_cn}", f"请抓取左边那个{color_cn}{target_cn}",
                       f"把位于左侧的{color_cn}{target_cn}拿起来", f"取起左手边的{color_cn}{target_cn}",
                       f"将左边的{color_cn}{target_cn}抓稳"][local_i % 5]
    elif category == "multi_object_ambiguity":
        alt_id = f"{prefix}_other_{global_i:03d}"
        objects.append(obj(alt_id, target_cat, "white", 0.55))
        if outcome == "READY":
            instruction = [f"拿起{color_cn}的{target_cn}", f"两个里面请抓{color_cn}{target_cn}",
                           f"选择{color_cn}{target_cn}并拿起来", f"别拿白色的，拿起{color_cn}{target_cn}",
                           f"请取颜色为{color_cn}的{target_cn}"][local_i % 5]
            # A negatively selected object is still a semantically relevant
            # exclusion and must be retained for downstream safety planning.
            if local_i % 5 == 3:
                expected["obstacle_entity_ids"] = [alt_id]
                expected["required_skills"] = ["PlanPath", "Reach", "Grasp"]
    elif category == "negation_obstacle" and outcome == "READY":
        instruction = [f"拿起{color_cn}{target_cn}，不要碰到{obs_cn}", f"请避开{obs_cn}抓取{color_cn}{target_cn}",
                       f"在不接触{obs_cn}的情况下拿起{color_cn}{target_cn}", f"抓{color_cn}{target_cn}，路径别经过{obs_cn}",
                       f"取起{color_cn}{target_cn}，操作时绕开{obs_cn}"][local_i % 5]
        expected["obstacle_entity_ids"] = [obstacle_id]
        expected["required_skills"] = ["PlanPath", "Reach", "Grasp"]
    elif category == "numeric_constraints" and outcome == "READY":
        force = float(2 + local_i % 3)
        instruction = [f"用不超过{force:g}N的力拿起{color_cn}{target_cn}", f"抓取{color_cn}{target_cn}，夹持力最多{force:g}N",
                       f"拿起{color_cn}{target_cn}，力量不要超过{force:g}牛顿", f"以最大{force:g}N的抓力取起{color_cn}{target_cn}",
                       f"请轻拿{color_cn}{target_cn}，力的上限是{force:g}N"][local_i % 5]
        expected["constraints"] = [{"parameter": "force_n", "operator": "max", "value": force, "unit": "N"}]
    elif category == "condition_sequence" and outcome == "READY":
        instruction = [f"先拿起{color_cn}{target_cn}，然后把它放到{dest_cn}里", f"抓取{color_cn}{target_cn}后再放进{dest_cn}",
                       f"第一步拿起{color_cn}{target_cn}，第二步放入{dest_cn}", f"请先抓{color_cn}{target_cn}，完成后将它放到{dest_cn}",
                       f"拿到{color_cn}{target_cn}以后，把它放进{dest_cn}"][local_i % 5]
        expected.update(action="PLACE", destination_entity_id=dest_id, required_skills=["Reach", "Place"], sequence_required=True)
    elif category == "missing_conflict_safety" and outcome == "READY":
        instruction = f"轻轻拿起{color_cn}{target_cn}"
    elif category == "colloquial_noise" and outcome == "READY":
        instruction = f"麻烦把那个{color_cn}的{target_cn}拿一下"
        if difficulty == "complex":
            instruction += f"，劲儿别超过3N，也别碰{obs_cn}"
            expected["constraints"] = [{"parameter": "force_n", "operator": "max", "value": 3.0, "unit": "N"}]
            expected["obstacle_entity_ids"] = [obstacle_id]
            expected["required_skills"] = ["PlanPath", "Reach", "Grasp"]
    elif category == "context_reference" and outcome == "READY":
        instruction = f"看见{color_cn}{target_cn}了吗？把它拿起来，过程中避开{obs_cn}，抓力不要超过3N"
        expected["constraints"] = [{"parameter": "force_n", "operator": "max", "value": 3.0, "unit": "N"}]
        expected["obstacle_entity_ids"] = [obstacle_id]
        expected["required_skills"] = ["PlanPath", "Reach", "Grasp"]

    # Preserve the advertised category for clarification/block cases. The v1
    # generator replaced these with generic missing-target text, contaminating
    # category scores.
    if outcome == "NEEDS_CLARIFICATION":
        if category == "role_binding":
            instruction = f"把{target_cn}放到{dest_cn}里"
        elif category == "attribute_spatial":
            instruction = f"拿起左侧的{target_cn}"
        elif category == "negation_obstacle":
            instruction = f"拿起{target_cn}，不要碰到{obs_cn}"
        elif category == "numeric_constraints":
            instruction = f"用不超过3N的力拿起{target_cn}"
        elif category == "condition_sequence":
            instruction = f"先拿起{target_cn}，然后把它放到{dest_cn}里"
        elif category == "context_reference":
            instruction = f"看到{target_cn}了吗？把它拿起来"
    elif outcome == "BLOCKED":
        if category == "missing_conflict_safety":
            # Missing target is safely handled by clarification OR blocking.
            expected["accepted_plan_statuses"] = ["NEEDS_CLARIFICATION", "BLOCKED"]
        else:
            # Create a real semantic conflict with a complete, realistic input.
            if target_id not in {o["object_id"] for o in objects}:
                objects.insert(0, obj(target_id, target_cat, color, 0.35))
            instruction = f"拿起{color_cn}{target_cn}，同时不要抓取{color_cn}{target_cn}"
            if category == "role_binding":
                instruction = f"把{color_cn}{target_cn}放到{dest_cn}，同时不要移动{color_cn}{target_cn}"
            elif category == "negation_obstacle":
                instruction += f"，并且避开{obs_cn}"
            elif category == "numeric_constraints":
                instruction = f"以至少5N且不超过2N的力拿起{color_cn}{target_cn}"
            elif category == "condition_sequence":
                instruction = f"先拿起{color_cn}{target_cn}再放到{dest_cn}，同时不要抓取它"
            expected["accepted_plan_statuses"] = ["BLOCKED"]

    expected.setdefault("accepted_plan_statuses", [outcome])

    return {
        "case_id": f"SAV1-{prefix}-{global_i+1:03d}",
        "scene": scene_name,
        "category": category,
        "difficulty": difficulty,
        "instruction": instruction,
        "observation_json": {
            "schema_version": "perception.v1", "frame_id": "robot_base", "unit": "m",
            "timestamp": f"2026-08-11T00:{global_i//60:02d}:{global_i%60:02d}+08:00",
            "objects": objects,
            "robot_state": {"available_skills": ["Reach", "Grasp", "Place", "PlanPath"],
                            "gripper": "empty", "homed": True},
        },
        "expected": expected,
        "input_contract": {
            "intentionally_missing_target": outcome == "BLOCKED" and category == "missing_conflict_safety",
            "intentional_ambiguity": outcome == "NEEDS_CLARIFICATION",
            "valid_sample": True,
        },
    }


def generate():
    cases = []
    for scene in ("daily", "industrial"):
        global_i = 0
        for category, count in CATEGORY_COUNTS.items():
            for local_i in range(count):
                cases.append(make_case(scene, category, local_i, global_i))
                global_i += 1
    payload = {
        "dataset_id": "strict_acceptance_v1_1",
        "frozen": True,
        "scoring": "whole-case downstream dispatchability",
        "cases": cases,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    OUT.write_bytes(raw)
    (ROOT / "strict_acceptance_v1_1.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii")
    return payload


if __name__ == "__main__":
    data = generate()
    print(f"generated={len(data['cases'])} path={OUT.name}")
