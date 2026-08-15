"""Generate the final independent Chinese open-language acceptance set.

The generator is deliberately separate from the historical regression-set
generator.  It uses the frozen capability contract and an independent scene
oracle, and never calls the system under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).parent
OUT = HERE / "open_language_acceptance_v1.json"
CONTRACT = HERE.parent / "domain" / "capability_contract_v1.json"

FORMAL_ACTIONS = [
    "GRASP", "DYNAMIC_GRASP", "PLACE", "FETCH", "TRANSFER",
    "HANDOVER", "PUSH", "POUR", "STACK", "WAIT",
]
PUBLIC_ACTION = {
    "GRASP": "grasp", "DYNAMIC_GRASP": "dynamic_grasp",
    "PLACE": "pick_and_place", "FETCH": "fetch", "TRANSFER": "transfer",
    "HANDOVER": "handover", "PUSH": "push", "POUR": "pour",
    "STACK": "stack", "WAIT": "wait", "CUSTOM": "unsupported",
}
SKILLS = {
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

COLORS = [
    ("红色", "red"), ("蓝色", "blue"), ("绿色", "green"),
    ("黄色", "yellow"), ("白色", "white"), ("黑色", "black"),
]
TARGETS = [
    ("cup", "杯子"), ("bottle", "瓶子"), ("box", "盒子"),
    ("book", "书本"), ("medicine_bottle", "药瓶"), ("workpiece", "工件"),
]
DESTINATIONS = [
    ("tray", "托盘"),
]
LOCATIONS = ["靠近左侧边缘", "位于中间偏后", "靠近右侧挡板", "在操作区前方"]
SIZES = ["偏小的", "中等大小的", "尺寸较大的", "细长的", "矮胖的"]


def opaque_id(case_no: int, role: str, salt: int = 0) -> str:
    raw = f"open-language-v1:{case_no}:{role}:{salt}".encode("utf-8")
    return f"obj_{hashlib.sha256(raw).hexdigest()[:12]}"


def object_record(
    object_id: str,
    category: str,
    color: str,
    x: float,
    y: float,
    *,
    surface: bool = False,
    moving: bool = False,
    affordances: list[str] | None = None,
    material: str = "plastic",
) -> dict[str, Any]:
    if surface:
        size = {"width": 0.45, "height": 0.05, "depth": 0.30}
        defaults = ["support_surface", "fixed", "container"]
    else:
        size = {"width": 0.06, "height": 0.08, "depth": 0.06}
        defaults = ["graspable", "movable"]
    return {
        "object_id": object_id,
        "category_candidates": [{"name": category, "score": 0.98}],
        "appearance": {"color": color, "material": material},
        "pose": {
            "frame_id": "robot_base",
            "position": {"x": x, "y": y, "z": 0.05},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "geometry": {"type": "oriented_bbox_3d", "size": size, "unit": "m"},
        "affordances": affordances or defaults,
        "tracking": {
            "state": "moving" if moving else "stationary",
            "velocity": {"x": 0.04 if moving else 0.0, "y": 0.0, "z": 0.0},
            "velocity_confidence": 0.95 if moving else 0.0,
        },
        "confidence": 0.98,
    }


def scene_for(case_no: int, action: str, *, obstacle: bool = False,
              ambiguous: bool = False, distractor: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    color_cn, color = COLORS[case_no % len(COLORS)]
    target_cat, target_cn = TARGETS[case_no % len(TARGETS)]
    dest_cat, dest_cn = DESTINATIONS[0]
    target_id = opaque_id(case_no, "theme")
    dest_id = opaque_id(case_no, "destination")
    target_affordances = {
        "PUSH": ["pushable", "movable"],
        "POUR": ["pourable", "graspable", "movable"],
        "STACK": ["stackable", "graspable", "movable"],
    }.get(action, ["graspable", "movable"])
    target = object_record(
        target_id, target_cat, color, 0.25 + (case_no % 4) * 0.04,
        0.10, moving=action == "DYNAMIC_GRASP", affordances=target_affordances,
        material="metal" if target_cat == "workpiece" else "plastic",
    )
    dest_affordances = ["support_surface", "fixed", "container"]
    if action == "POUR":
        dest_affordances.append("destination_container")
    if action == "STACK":
        dest_affordances.append("destination_stable")
    destination = object_record(
        dest_id, dest_cat, "gray", 0.62, -0.28, surface=True,
        affordances=dest_affordances,
    )
    objects = [target]
    roles = {"theme": target_id, "destination": dest_id,
             "color_cn": color_cn, "color": color, "target_cn": target_cn,
             "dest_cn": dest_cn, "dest_cat": dest_cat}
    if action in {"PLACE", "FETCH", "TRANSFER", "POUR", "STACK"}:
        objects.append(destination)
    if action == "HANDOVER":
        recipient_id = opaque_id(case_no, "recipient")
        objects.append(object_record(recipient_id, "operator", "none", 0.70, 0.15,
                                     affordances=["recipient", "reachable"]))
        roles["recipient"] = recipient_id
    if action == "FETCH":
        destination["affordances"].append("robot_receive_zone")
    if obstacle:
        obstacle_id = opaque_id(case_no, "obstacle")
        objects.append(object_record(obstacle_id, "fixture", "orange", 0.44, 0.32,
                                     surface=True))
        roles["obstacle"] = obstacle_id
    if distractor:
        distractor_id = opaque_id(case_no, "distractor")
        distractor_color = color if ambiguous else "white"
        objects.append(object_record(distractor_id, target_cat, distractor_color, 0.43, 0.10,
                                     affordances=target_affordances))
        roles["distractor"] = distractor_id
    if case_no % 2:
        objects.reverse()
    return {
        "schema_version": "1.0.0",
        "message_type": "perception_observation",
        "observation_id": opaque_id(case_no, "observation"),
        "scene_id": opaque_id(case_no, "scene"),
        "timestamp": "2026-08-13T12:00:00Z",
        "clock_domain": "unix_utc",
        "coordinate_system": "robot_base",
        "frame_id": "robot_base",
        "unit": "m",
        "objects": objects,
        "relations": [],
        "robot_state": {"available_skills": SKILLS.get(action, [])},
    }, roles


LEXICAL = {
    "GRASP": [
        "稳稳托起{c}{t}", "让夹具把{c}{t}牢牢控住", "把{c}{t}从原处提离",
        "将{c}{t}收入夹持范围", "把{c}{t}轻轻擎在夹爪里", "对{c}{t}完成取持",
        "把{c}{t}拿稳后悬空", "将{c}{t}从台面上提走", "把{c}{t}可靠地夹持起来",
        "让机械手握住{c}{t}", "把{c}{t}从接触面抬开", "对准后控制住{c}{t}",
        "把{c}{t}提到空中", "将{c}{t}稳妥地取离台面", "夹爪把{c}{t}抱住", "把{c}{t}拿在手上",
    ],
    "DYNAMIC_GRASP": [
        "趁{c}{t}移动时把它截住", "跟住运动中的{c}{t}并将其控稳", "对正在位移的{c}{t}实施截取",
        "等{c}{t}进入夹持窗口后收住它", "把滑动中的{c}{t}拦下并抓牢", "跟踪{c}{t}的移动轨迹后取持",
        "在{c}{t}尚未停下时把它稳住", "对移动目标{c}{t}完成动态取持", "迎着移动中的{c}{t}把它接住",
        "让夹爪追上{c}{t}再把它控住", "捕获正在运动的{c}{t}", "把行进中的{c}{t}截留在夹爪内",
    ],
    "PLACE": [
        "把{c}{t}安顿到{d}里", "让{c}{t}落在{d}的承托面上", "将{c}{t}归置进{d}",
        "把{c}{t}送到{d}的内部", "把{c}{t}稳妥地安放于{d}", "让{c}{t}在{d}上就位",
        "把{c}{t}转手放入{d}", "将{c}{t}放回{d}的承载区域", "把{c}{t}安排在{d}中",
        "把{c}{t}卸到{d}", "令{c}{t}最终停在{d}上", "把{c}{t}搁置于{d}",
        "将{c}{t}平稳地交给{d}承托", "把{c}{t}收进{d}里", "使{c}{t}落位到{d}", "把{c}{t}摆放妥当于{d}",
    ],
    "FETCH": [
        "把{c}{t}取到收取区", "将{c}{t}带回机器人接收位", "把{c}{t}从现场拿来",
        "把{c}{t}送回指定接收处", "将{c}{t}带到回收托盘", "把{c}{t}取回来放入接收区",
        "帮忙把{c}{t}弄到收取位置", "把{c}{t}从那里带回这边", "将{c}{t}带至机器人身边",
        "把{c}{t}搬回接收台", "把{c}{t}取出并送回收取处", "让{c}{t}回到机器人接收区",
        "把{c}{t}带来交给接收位", "把现场的{c}{t}收回", "取回{c}{t}并放到接收区", "把{c}{t}搬到回收位置",
    ],
    "TRANSFER": [
        "把{c}{t}调运到{d}", "将{c}{t}转送至{d}", "把{c}{t}从当前工位移交给{d}",
        "把{c}{t}输送到{d}", "将{c}{t}改送至{d}", "把{c}{t}运往{d}",
        "完成{c}{t}到{d}的移送", "把{c}{t}调拨进{d}", "将{c}{t}送入{d}的作业位",
        "把{c}{t}从这里转运到{d}", "让{c}{t}换到{d}所在工位", "将{c}{t}转交至{d}",
        "把{c}{t}移交到{d}一侧", "把{c}{t}从当前处所搬至{d}", "将{c}{t}输送进{d}", "把{c}{t}调到{d}处",
    ],
    "HANDOVER": [
        "把{c}{t}递到操作员手边", "将{c}{t}交由操作员接收", "把{c}{t}送到工作人员手里",
        "把{c}{t}传递给现场操作员", "让操作员接过{c}{t}", "将{c}{t}递交给人手",
        "把{c}{t}交到接收者手中", "把{c}{t}送到操作人员面前", "将{c}{t}交付给操作员",
        "让机械手把{c}{t}递给工作人员", "把{c}{t}移到操作员可接取的位置", "向操作员交付{c}{t}",
    ],
    "PUSH": [
        "把{c}{t}沿台面推行一段", "让{c}{t}在平面上向前移动", "把{c}{t}推离当前位置",
        "沿直线推动{c}{t}", "使{c}{t}向作业区方向滑行", "把{c}{t}推到指定方向",
        "推动{c}{t}越过当前标记", "把{c}{t}向前顶过去", "让{c}{t}在台面上移位",
        "把{c}{t}推向右侧作业带", "把{c}{t}沿路径推走", "对{c}{t}施加推移动作",
        "把{c}{t}从挡板旁推开", "推动{c}{t}离开原来的位置", "让{c}{t}向目标方向挪动", "将{c}{t}推行至前方",
    ],
    "POUR": [
        "把{c}{t}中的内容灌进{d}", "将{c}{t}里的东西转注到{d}", "把{c}{t}朝向{d}完成倾注",
        "让{c}{t}向{d}释放内部物料", "把{c}{t}的内容物倒进{d}", "将{c}{t}内的液体注入{d}",
        "把{c}{t}倾向{d}并完成转倒", "把容器{c}{t}里的东西灌入{d}", "向{d}倾空{c}{t}",
        "把{c}{t}中的物料导入{d}", "将{c}{t}倒向{d}的容纳区域", "完成{c}{t}到{d}的内容转移",
        "把{c}{t}里的材料倾入{d}", "让{c}{t}向{d}完成注料", "把{c}{t}翻向{d}并倒出内容", "把{c}{t}中的东西灌到{d}",
    ],
    "STACK": [
        "把{c}{t}叠置到{d}上方", "将{c}{t}码到{d}的顶部", "把{c}{t}压在{d}的承托面上形成堆叠",
        "让{c}{t}与{d}上下相叠", "把{c}{t}摞到{d}上", "将{c}{t}加到{d}这一摞的最上面",
        "把{c}{t}放成位于{d}之上的一层", "把{c}{t}和{d}叠合起来", "将{c}{t}码放在{d}顶部",
        "把{c}{t}垒到{d}上方", "让{c}{t}成为{d}上面的那一件", "把{c}{t}整齐地堆在{d}上",
        "将{c}{t}安置成{d}的上层物体", "把{c}{t}叠加到{d}之上", "把{c}{t}放到{d}的顶层", "完成{c}{t}在{d}上的堆码",
    ],
    "WAIT": [
        "先不动作，等目标完全静止", "保持当前状态直到运动停止", "等场景中的运动目标稳定下来",
        "在目标还有位移时暂缓操作", "等传送过程结束后再继续", "让系统等到目标不再变化",
        "暂时保持等待，直到工位稳定", "在目标稳住以前不要开始下一步", "等移动状态消失后再处理",
        "先观察一会儿，直到目标停止", "待场景恢复稳定后继续任务", "把后续动作延后到目标静止",
    ],
}


def instruction_for(action: str, axis: str, local: int, roles: dict[str, str]) -> str:
    c, t, d = roles["color_cn"], roles["target_cn"], roles["dest_cn"]
    templates = LEXICAL[action]
    base = templates[local % len(templates)].format(c=c, t=t, d=d)
    location = LOCATIONS[(local + len(action)) % len(LOCATIONS)]
    size = SIZES[(local // 2) % len(SIZES)]
    if action != "WAIT" and axis == "lexical_expression":
        return f"请{base}，目标是{location}的{size}{c}{t}"
    if axis == "syntax_variation":
        wrappers = [
            f"完成确认后再执行：{base}", f"如果现场条件满足，请{base}",
            f"当前优先事项是，{base}", f"麻烦先把目标辨认好，然后{base}",
        ]
        return wrappers[local % len(wrappers)]
    if axis == "compositional":
        extras = [
            "过程中不要碰橙色夹具", "全程保持轻柔", "完成后再释放夹爪",
            "先确认目标仍然可见", "动作之间保持先后顺序", "不要把旁边的同类物体混进去",
        ]
        return f"先确认{location}的{size}{base}，{extras[local % len(extras)]}"
    if axis == "scene_grounding":
        return f"在现场目标中，操作{location}的{size}{base}"
    if axis == "constraint_safety":
        safety = [
            "用力不要超过2N", "速度保持在0.15m/s以内", "不要接触橙色夹具",
            "如果目标不明确就先询问", "不能越过安全边界", "只处理指定颜色的那个",
        ][local % 6]
        return f"{base}，并且{safety}"
    return base


def expected_for(action: str, roles: dict[str, str], axis: str, local: int,
                 instruction: str) -> dict[str, Any]:
    ready = action not in {"FETCH", "HANDOVER"} or True
    status = "READY"
    theme = roles.get("theme")
    destination = roles.get("destination") if action in {"PLACE", "FETCH", "TRANSFER", "POUR", "STACK"} else None
    if axis == "scene_grounding" and local % 6 == 5:
        status, theme, destination = "NEEDS_CLARIFICATION", None, None
    if axis == "constraint_safety" and local % 6 == 3:
        status, theme, destination = "NEEDS_CLARIFICATION", None, None
    obstacles = [roles["obstacle"]] if "橙色夹具" in instruction and "obstacle" in roles else []
    constraints = []
    if axis == "constraint_safety":
        if local % 6 == 0:
            constraints = [{"parameter": "force_n", "operator": "max", "value": 2.0, "unit": "N"}]
        elif local % 6 == 1:
            constraints = [{"parameter": "velocity_ms", "operator": "max", "value": 0.15, "unit": "m/s"}]
    return {
        "plan_status": status,
        "accepted_plan_statuses": [status] if status == "READY" else ["NEEDS_CLARIFICATION", "BLOCKED"],
        "execution_allowed": status == "READY",
        "action": action,
        "public_action": PUBLIC_ACTION[action],
        "theme_entity_id": theme if status == "READY" else None,
        "destination_entity_id": destination if status == "READY" else None,
        "obstacle_entity_ids": obstacles,
        "required_skills": SKILLS[action] if status == "READY" else [],
        "constraints": constraints,
        "source": "independent_scene_oracle",
    }


def build_case(case_no: int, axis: str, action: str, local: int,
               difficulty: str) -> dict[str, Any]:
    obstacle = axis in {"compositional", "constraint_safety"}
    ambiguous = (axis == "scene_grounding" and local % 6 == 5) or (
        axis == "constraint_safety" and local % 6 == 3
    )
    observation, roles = scene_for(
        case_no, action, obstacle=obstacle, ambiguous=ambiguous,
        distractor=axis == "scene_grounding" or ambiguous,
    )
    instruction = instruction_for(action, axis, local, roles)
    expected = expected_for(action, roles, axis, local, instruction)
    return {
        "case_id": f"OPEN-GEN-{case_no:04d}",
        "generalization_axis": [axis],
        "family_id": f"open:{axis}:{action.lower()}:{local:02d}",
        "scene": "daily" if case_no % 2 else "industrial",
        "difficulty": difficulty,
        "category": axis,
        "instruction": instruction,
        "observation_json": observation,
        "expected": expected,
        "input_contract": {
            "valid_sample": True,
            "evaluation_truth_present": False,
            "independent_language_author": "synthetic_independent_protocol",
        },
    }


def build_ood(case_no: int, local: int, difficulty: str) -> dict[str, Any]:
    observation, roles = scene_for(case_no, "GRASP", obstacle=False)
    unsupported = ["给杯子拍照", "把瓶子清洗干净", "把工件焊接起来", "测量盒子的温度"][local % 4]
    return {
        "case_id": f"OPEN-GEN-{case_no:04d}",
        "generalization_axis": ["ood_rejection"],
        "family_id": f"open:ood:{local:02d}",
        "scene": "daily" if case_no % 2 else "industrial",
        "difficulty": difficulty,
        "category": "ood_rejection",
        "instruction": unsupported,
        "observation_json": observation,
        "expected": {
            "plan_status": "BLOCKED", "accepted_plan_statuses": ["BLOCKED", "NEEDS_CLARIFICATION"],
            "execution_allowed": False, "action": "CUSTOM", "public_action": "unsupported",
            "theme_entity_id": None, "destination_entity_id": None,
            "obstacle_entity_ids": [], "required_skills": [], "constraints": [],
            "source": "independent_scene_oracle",
        },
        "input_contract": {"valid_sample": True, "domain_out_of_scope": True,
                           "evaluation_truth_present": False},
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
                cases.append(build_case(case_no, axis, action, local, difficulty_plan[case_no - 1]))
                case_no += 1
    for local in range(40):
        cases.append(build_ood(case_no, local, difficulty_plan[case_no - 1]))
        case_no += 1
    return {
        "dataset_id": "open_language_acceptance_v1",
        "version": "1.0.0",
        "frozen": True,
        "generation_protocol": "new Chinese language forms plus independent scene oracle; no system-under-test calls; held-out after generation",
        "capability_contract": "capability_contract.v1",
        "axes": {"lexical_expression": 160, "syntax_variation": 120,
                 "compositional": 200, "scene_grounding": 120,
                 "constraint_safety": 160, "ood_rejection": 40},
        "cases": cases,
    }


if __name__ == "__main__":
    payload = build()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(json.dumps({"dataset": payload["dataset_id"], "cases": len(payload["cases"]),
                      "sha256": digest}, ensure_ascii=False))
