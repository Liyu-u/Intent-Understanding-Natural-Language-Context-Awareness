"""Generate the frozen in-workspace generalization acceptance set.

The generator uses only the frozen capability contract and scene-defined
expected IDs.  It never calls the system under test.  This gives an
independent oracle from model output, while the repository location means
the resulting set is not a third-party sequestered holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).parent
OUT = HERE / "generalization_blind_v1.json"
CONTRACT = HERE.parent / "domain" / "capability_contract_v1.json"

FORMAL_ACTIONS = [
    "GRASP", "DYNAMIC_GRASP", "PLACE", "FETCH", "TRANSFER",
    "HANDOVER", "PUSH", "POUR", "STACK", "WAIT",
]
PUBLIC_ACTION = {
    "GRASP": "grasp", "DYNAMIC_GRASP": "dynamic_grasp",
    "PLACE": "pick_and_place", "FETCH": "fetch", "TRANSFER": "transfer",
    "HANDOVER": "handover", "PUSH": "push", "POUR": "pour",
    "STACK": "stack", "WAIT": "wait",
}
ACTION_SKILLS = {
    "GRASP": ["Reach", "Grasp"],
    "DYNAMIC_GRASP": ["WaitUntilStable", "Reach", "DynamicGrasp"],
    "PLACE": ["Reach", "Grasp", "Transport", "Place"],
    "FETCH": ["Reach", "Grasp", "Fetch"],
    "TRANSFER": ["Reach", "Grasp", "Transport", "Place"],
    "HANDOVER": ["Reach", "Grasp", "MoveToHandoverZone", "Handover"],
    "PUSH": ["Reach", "Push", "Release"],
    "POUR": ["Reach", "Grasp", "MoveTo", "Pour", "Release"],
    "STACK": ["Reach", "Grasp", "MoveTo", "Stack", "Release"],
    "WAIT": ["WaitUntil"],
}
READY_ACTIONS = {"GRASP", "DYNAMIC_GRASP", "PLACE", "TRANSFER", "PUSH", "POUR", "STACK"}
AXIS_QUOTAS = {
    "lexical_expression": 160,
    "syntax_variation": 120,
    "compositional": 200,
    "scene_grounding": 120,
    "constraint_safety": 160,
    "ood_rejection": 40,
}
COLORS = [("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("黄色", "yellow")]
TARGETS = [
    ("cup", "杯子"), ("bottle", "瓶子"), ("box", "盒子"),
    ("book", "书本"), ("medicine_bottle", "药瓶"), ("workpiece", "工件"),
]


def opaque_id(case_no: int, role: str, salt: int = 0) -> str:
    raw = f"generalization-v1:{case_no}:{role}:{salt}".encode("utf-8")
    return f"u_{hashlib.sha256(raw).hexdigest()[:10]}"


def object_record(
    object_id: str,
    category: str,
    color: str,
    x: float,
    y: float,
    *,
    surface: bool = False,
    moving: bool = False,
    material: str = "plastic",
    affordances: list[str] | None = None,
) -> dict[str, Any]:
    if surface:
        size = {"width": 0.45, "height": 0.05, "depth": 0.30}
        object_affordances = ["support_surface", "fixed", "container"]
    else:
        size = {"width": 0.06, "height": 0.08, "depth": 0.06}
        object_affordances = ["graspable", "movable"]
    if affordances is not None:
        object_affordances = affordances
    return {
        "object_id": object_id,
        "category_candidates": [{"name": category, "score": 0.98}],
        "appearance": {"color": color, "material": material},
        "pose": {
            "frame_id": "robot_base",
            "position": {"x": x, "y": y, "z": 0.05},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "geometry": {"size": size, "unit": "m"},
        "affordances": object_affordances,
        "tracking": {
            "state": "moving" if moving else "stationary",
            "velocity": {"x": 0.03 if moving else 0.0, "y": 0.0, "z": 0.0},
            "velocity_confidence": 0.95 if moving else 0.0,
        },
        "confidence": 0.98,
    }


def action_spec(action: str, case_no: int) -> dict[str, Any]:
    color_cn, color = COLORS[case_no % len(COLORS)]
    category, category_cn = TARGETS[case_no % len(TARGETS)]
    target_id = opaque_id(case_no, "theme")
    dest_id = opaque_id(case_no, "destination")
    obstacle_id = opaque_id(case_no, "obstacle")
    target = object_record(
        target_id, category, color, 0.30, 0.10,
        moving=action in {"DYNAMIC_GRASP", "WAIT"}, material="metal" if category == "workpiece" else "plastic",
        affordances=(
            ["pushable", "movable"] if action == "PUSH" else
            ["pourable", "graspable", "movable"] if action == "POUR" else
            ["stackable", "graspable", "movable"] if action == "STACK" else
            ["graspable", "movable"]
        ),
    )
    destination = object_record(
        dest_id, "tray", "gray", 0.65, -0.35, surface=True,
        affordances=["support_surface", "fixed", "container", "destination_stable"]
        if action != "POUR" else ["support_surface", "fixed", "container", "destination_container"],
    )
    obstacle = object_record(obstacle_id, "fixture", "orange", 0.48, 0.35, surface=True)
    return {
        "color_cn": color_cn, "color": color, "category": category,
        "category_cn": category_cn, "target_id": target_id, "dest_id": dest_id,
        "obstacle_id": obstacle_id, "target": target, "destination": destination,
        "obstacle": obstacle,
    }


def base_instruction(action: str, spec: dict[str, Any], variant: int) -> str:
    c, t = spec["color_cn"], spec["category_cn"]
    if action == "GRASP":
        return [f"把{c}{t}提起来", f"请将{c}{t}稳稳拿住", f"把桌面上的{c}{t}取离原位", f"麻烦夹住{c}{t}"][variant % 4]
    if action == "DYNAMIC_GRASP":
        return [f"抓住正在移动的{c}{t}", f"把移动中的{c}{t}稳住", f"等它靠近后抓住{c}{t}"][variant % 3]
    if action == "PLACE":
        return [f"把{c}{t}放到托盘上", f"将{c}{t}安放进托盘", f"把{c}{t}摆到灰色托盘里"][variant % 3]
    if action == "FETCH":
        return [f"把{c}{t}拿过来", f"将{c}{t}取到手边", f"帮我把{c}{t}带过来"][variant % 3]
    if action == "TRANSFER":
        return [f"把{c}{t}搬运到托盘", f"将{c}{t}移送到灰色托盘", f"把{c}{t}转移至托盘"][variant % 3]
    if action == "HANDOVER":
        return [f"把{c}{t}递给我", f"请将{c}{t}交到操作员手里", f"把{c}{t}传给用户"][variant % 3]
    if action == "PUSH":
        return [f"把{c}{t}推到托盘旁", f"沿直线把{c}{t}推过去", f"将{c}{t}向托盘方向推动"][variant % 3]
    if action == "POUR":
        return [f"把{c}{t}里的东西倒入托盘", f"握住{c}{t}向托盘倾倒", f"将{c}{t}中的内容倒进托盘"][variant % 3]
    if action == "STACK":
        return [f"把{c}{t}叠放到托盘上", f"将{c}{t}堆到托盘顶部", f"把{c}{t}整齐码放在托盘上"][variant % 3]
    return ["等到目标稳定后再继续", "请等待目标停止移动", "保持等待直到场景稳定"][variant % 3]


def build_observation(spec: dict[str, Any], action: str, axis: str, local: int) -> dict[str, Any]:
    objects = [spec["target"]]
    if action in {"PLACE", "TRANSFER", "PUSH", "POUR", "STACK"}:
        objects.append(spec["destination"])
    if axis in {"scene_grounding", "constraint_safety"}:
        objects.append(spec["obstacle"])
    if axis == "scene_grounding":
        extra = object_record(opaque_id(local + 9000, "distractor"), spec["category"], "white", 0.42, 0.10)
        objects.append(extra)
        if local % 2:
            objects = list(reversed(objects))
    if axis == "constraint_safety" and local % 4 == 2 and action != "WAIT":
        # A real ambiguity case: remove the discriminating color from the
        # command and expose two same-category candidates.
        objects.append(object_record(
            opaque_id(local + 9100, "ambiguous"), spec["category"], "white", 0.42, 0.10,
            moving=action == "DYNAMIC_GRASP",
            material="metal" if spec["category"] == "workpiece" else "plastic",
            affordances=spec["target"]["affordances"],
        ))
    return {
        "schema_version": "perception.v1",
        "message_type": "perception_observation",
        "observation_id": f"gen_obs_{opaque_id(local + 5000, 'obs')}",
        "scene_id": f"gen_scene_{local % 2}",
        "frame_id": "robot_base",
        "unit": "m",
        "timestamp": f"2026-08-13T12:{local % 60:02d}:00Z",
        "objects": objects,
        "robot_state": {"available_skills": ACTION_SKILLS.get(action, [])},
    }


def expected_for(action: str, spec: dict[str, Any], instruction: str, axis: str, local: int) -> dict[str, Any]:
    # FETCH/HANDOVER need a real delivery/handover pose or zone.  The public
    # observation contract intentionally does not fabricate one, so these
    # cases test correct clarification rather than false execution.
    ready = action in READY_ACTIONS and action not in {"FETCH", "HANDOVER"}
    status = "READY" if ready else "NEEDS_CLARIFICATION"
    theme = spec["target_id"] if ready else None
    destination = spec["dest_id"] if ready and action in {"PLACE", "TRANSFER", "POUR", "STACK"} else None
    constraints: list[dict[str, Any]] = []
    obstacle_ids: list[str] = []
    if axis == "constraint_safety":
        if local % 4 == 0 and action in {"GRASP", "DYNAMIC_GRASP", "PLACE", "TRANSFER"}:
            instruction = instruction + "，力度不超过2N"
            constraints.append({"parameter": "force_n", "operator": "max", "value": 2.0, "unit": "N"})
        elif local % 4 == 1:
            instruction = instruction + "，过程中不要碰橙色夹具"
            obstacle_ids = [spec["obstacle_id"]]
        elif local % 4 == 2:
            status, theme, destination = "NEEDS_CLARIFICATION", None, None
    return {
        "plan_status": status,
        "accepted_plan_statuses": [status] if status == "READY" else ["NEEDS_CLARIFICATION", "BLOCKED"],
        "execution_allowed": status == "READY",
        "action": action,
        "public_action": PUBLIC_ACTION[action],
        "theme_entity_id": theme,
        "destination_entity_id": destination,
        "obstacle_entity_ids": obstacle_ids,
        "required_skills": ACTION_SKILLS[action] if status == "READY" else [],
        "constraints": constraints,
        "instruction_after_constraint": instruction,
    }


def make_case(case_no: int, axis: str, action: str, local: int, difficulty: str) -> dict[str, Any]:
    spec = action_spec(action, case_no)
    instruction = base_instruction(action, spec, local)
    if axis == "syntax_variation":
        instruction = [f"现在请{instruction}", f"完成准备后，{instruction}", f"如果可以的话，{instruction}"][local % 3]
    elif axis == "compositional":
        if action in {"PLACE", "TRANSFER", "PUSH", "POUR", "STACK"}:
            instruction = f"先处理好{spec['color_cn']}{spec['category_cn']}，随后{instruction}"
        elif action == "WAIT":
            instruction = "在目标仍在变化时先保持等待，直到它稳定下来"
        else:
            instruction = f"先确认目标位置，再{instruction}"
    if axis == "constraint_safety" and local % 4 == 2 and action != "WAIT":
        instruction = {
            "GRASP": "把那个同类物体拿起来",
            "DYNAMIC_GRASP": "抓住正在移动的那个目标",
            "PLACE": "把那个物体放到托盘上",
            "TRANSFER": "把那个物体搬运到托盘",
            "PUSH": "把那个物体推过去",
            "POUR": "把那个容器里的东西倒入托盘",
            "STACK": "把那个物体叠到托盘上",
        }.get(action, instruction)
    expected = expected_for(action, spec, instruction, axis, local)
    instruction = expected.pop("instruction_after_constraint")
    observation = build_observation(spec, action, axis, local)
    return {
        "case_id": f"GEN_{case_no:04d}",
        "generalization_axis": [axis],
        "family_id": f"{axis}:{action.lower()}:{local % 8:02d}",
        "scene": "daily" if case_no % 2 else "industrial",
        "difficulty": difficulty,
        "category": axis,
        "instruction": instruction,
        "observation_json": observation,
        "expected": expected,
        "input_contract": {
            "valid_sample": True,
            "intentionally_missing_target": expected["theme_entity_id"] is None and expected["execution_allowed"],
            "intentional_ambiguity": expected["plan_status"] != "READY" and axis == "constraint_safety",
            "evaluation_truth_present": False,
        },
    }


def make_ood_case(case_no: int, local: int, difficulty: str) -> dict[str, Any]:
    spec = action_spec("GRASP", case_no)
    unsupported = ["清洗", "切割", "焊接", "读取温度"][local % 4]
    instruction = f"请{unsupported}{spec['color_cn']}{spec['category_cn']}"
    observation = build_observation(spec, "GRASP", "ood_rejection", local)
    return {
        "case_id": f"GEN_{case_no:04d}",
        "generalization_axis": ["ood_rejection"],
        "family_id": f"ood:unsupported:{local:02d}",
        "scene": "daily" if case_no % 2 else "industrial",
        "difficulty": difficulty,
        "category": "ood_rejection",
        "instruction": instruction,
        "observation_json": observation,
        "expected": {
            "plan_status": "BLOCKED",
            "accepted_plan_statuses": ["BLOCKED", "NEEDS_CLARIFICATION"],
            "execution_allowed": False,
            "action": "CUSTOM",
            "public_action": "unsupported",
            "theme_entity_id": None,
            "destination_entity_id": None,
            "obstacle_entity_ids": [],
            "required_skills": [],
            "constraints": [],
        },
        "input_contract": {"valid_sample": True, "domain_out_of_scope": True},
    }


def build() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    difficulty_plan = ["simple"] * 200 + ["medium"] * 360 + ["complex"] * 240
    case_no = 1
    for axis, per_action in (
        ("lexical_expression", 16), ("syntax_variation", 12),
        ("compositional", 20), ("scene_grounding", 12),
        ("constraint_safety", 16),
    ):
        for action in FORMAL_ACTIONS:
            for local in range(per_action):
                cases.append(make_case(case_no, axis, action, local, difficulty_plan[case_no - 1]))
                case_no += 1
    for local in range(AXIS_QUOTAS["ood_rejection"]):
        cases.append(make_ood_case(case_no, local, difficulty_plan[case_no - 1]))
        case_no += 1
    return {
        "dataset_id": "generalization_blind_v1",
        "version": "1.0.0",
        "frozen": True,
        "generation_protocol": "contract-fixed independent scene oracle; no system-under-test calls",
        "capability_contract": "capability_contract.v1",
        "axes": AXIS_QUOTAS,
        "cases": cases,
    }


if __name__ == "__main__":
    payload = build()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (HERE / "generalization_blind_v1.sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"dataset": payload["dataset_id"], "cases": len(payload["cases"]), "sha256": digest}, ensure_ascii=False))
