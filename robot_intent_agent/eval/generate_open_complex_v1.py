"""Generate a frozen 500-case open-language complex intent benchmark.

The expected answers are constructed from scenario semantics, never from the
system output.  The benchmark is intentionally separate from controlled
acceptance data and must not be used as a rule-development fixture.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "open_complex_v1.json"
SHA = HERE / "open_complex_v1.sha256"
SEED = 20260811

SCENES = {
    "daily": {
        "targets": [("cup", "杯子"), ("bottle", "瓶子"), ("box", "盒子"),
                    ("book", "书"), ("medicine_bottle", "药瓶")],
        "destinations": [("tray", "托盘"), ("table", "桌面"),
                         ("cabinet", "柜子"), ("bin", "收纳箱")],
        "obstacles": [("glass", "玻璃杯"), ("vase", "花瓶"),
                      ("hot_kettle", "热水壶")],
    },
    "industrial": {
        "targets": [("workpiece", "工件"), ("part", "零件"),
                    ("bearing", "轴承"), ("gear", "齿轮"),
                    ("component", "组件")],
        "destinations": [("tray", "托盘"), ("inspection_zone", "检测区"),
                         ("parts_bin", "料箱"), ("workbench", "工位")],
        "obstacles": [("welding_zone", "焊接区"), ("fixture", "夹具"),
                      ("hot_surface", "高温台")],
    },
}

COLORS = [("red", "红色"), ("blue", "蓝色"), ("green", "绿色"),
          ("yellow", "黄色"), ("white", "白色")]

CATEGORY_COUNTS = {
    "open_paraphrase": 60,
    "composite_pick_place": 60,
    "context_pronoun": 50,
    "negative_contrast": 50,
    "multi_constraint": 50,
    "multi_object_ambiguity": 50,
    "explicit_conflict": 40,
    "conditional_sequence": 50,
    "industrial_role_binding": 50,
    "perception_robustness": 40,
}


def _obj(oid: str, category: str, color: str, x: float, y: float,
         *, support: bool = False, confidence: float = 0.97,
         affordances: list[str] | None = None) -> dict:
    sizes = {
        "table": (0.80, 0.72, 0.60), "workbench": (1.00, 0.80, 0.70),
        "inspection_zone": (0.60, 0.04, 0.50), "tray": (0.35, 0.04, 0.25),
        "parts_bin": (0.40, 0.25, 0.30), "bin": (0.40, 0.25, 0.30),
        "cabinet": (0.70, 1.20, 0.45), "welding_zone": (0.80, 0.05, 0.80),
        "fixture": (0.25, 0.15, 0.20), "hot_surface": (0.50, 0.08, 0.40),
    }
    w, h, d = sizes.get(category, (0.06, 0.08, 0.06))
    material = "metal" if category in {
        "workpiece", "part", "bearing", "gear", "component", "fixture", "hot_surface"
    } else "glass" if category == "glass" else "plastic"
    affs = affordances if affordances is not None else (
        ["support_surface", "container", "fixed"] if support else ["graspable", "movable"]
    )
    return {
        "object_id": oid,
        "category_candidates": [{"name": category, "score": confidence}],
        "appearance": {"color": color, "material": material},
        "pose": {"frame_id": "robot_base", "position": {"x": x, "y": y, "z": 0.05},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "geometry": {"size": {"width": w, "height": h, "depth": d}, "unit": "m"},
        "affordances": affs,
        "confidence": confidence,
    }


def _expected(status: str, action: str = "GRASP", theme: str | None = None,
              destination: str | None = None, obstacles: list[str] | None = None,
              skills: list[str] | None = None, constraints: list[dict] | None = None,
              sequence: bool = False) -> dict:
    return {
        "plan_status": status,
        "accepted_plan_statuses": [status],
        "execution_allowed": status == "READY",
        "action": action,
        "theme_entity_id": theme if status == "READY" else None,
        "destination_entity_id": destination if status == "READY" else None,
        "obstacle_entity_ids": obstacles or [],
        "required_skills": skills or [],
        "constraints": constraints or [],
        "sequence_required": sequence,
    }


def _case(scene: str, category: str, local_i: int, global_i: int, rng: random.Random) -> dict:
    spec = SCENES[scene]
    target_cat, target_cn = spec["targets"][local_i % len(spec["targets"])]
    dest_cat, dest_cn = spec["destinations"][(local_i * 3 + 1) % len(spec["destinations"])]
    obs_cat, obs_cn = spec["obstacles"][(local_i * 2 + 1) % len(spec["obstacles"])]
    color, color_cn = COLORS[(local_i * 2 + global_i) % len(COLORS)]
    prefix = "D" if scene == "daily" else "I"
    cid = f"OCV1-{prefix}-{global_i + 1:03d}"
    tid, did, oid = f"{cid}-target", f"{cid}-dest", f"{cid}-obstacle"
    objects = [
        _obj(tid, target_cat, color, 0.34, -0.12),
        _obj(did, dest_cat, "gray", 0.66, -0.48, support=True),
        _obj(oid, obs_cat, "orange", 0.50, 0.52),
    ]
    discourse = ["麻烦你", "劳驾", "现在请", "帮我", "如果方便的话", "请稳妥地"][local_i % 6]
    difficulty = "complex" if category not in {"open_paraphrase", "multi_object_ambiguity"} else "medium"

    if category == "open_paraphrase":
        templates = [
            "{p}把视野里那件{c}{t}取起来", "{p}夹稳那只{c}{t}",
            "{p}将靠近机械臂的{c}{t}拿好", "{p}拿一下现场的{c}{t}",
            "{p}选中{c}{t}后抓牢", "{p}把{c}{t}从当前位置提起",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, c=color_cn, t=target_cn)
        exp = _expected("READY", theme=tid, skills=["Reach", "Grasp"])

    elif category == "composite_pick_place":
        templates = [
            "{p}先取起{c}{t}，确认抓稳以后放进{d}",
            "{p}完成{c}{t}的抓取，接着把它摆到{d}",
            "{p}把{c}{t}拿离原位并安放在{d}中",
            "{p}第一步夹住{c}{t}，第二步转放到{d}",
            "{p}抓稳{c}{t}之后，再将其放至{d}",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, c=color_cn, t=target_cn, d=dest_cn)
        exp = _expected("READY", "PLACE", tid, did, skills=["Reach", "Grasp", "Place"], sequence=True)

    elif category == "context_pronoun":
        templates = [
            "你看到{c}{t}了吧？把它拿起来，途中绕开{obs}，抓力上限为3N",
            "刚才提到的{c}{t}，请将其取起；别碰{obs}，力量不要超过3牛顿",
            "目标是{c}{t}。拿起它时避开{obs}，夹持力至多3N",
            "现场那件{c}{t}就是目标，把这个拿稳，路径别经过{obs}，劲儿别超过3N",
            "关于{c}{t}，请完成抓取；过程中不要接触{obs}，最大抓力3N",
        ]
        instruction = templates[local_i % len(templates)].format(c=color_cn, t=target_cn, obs=obs_cn)
        constraint = [{"parameter": "force_n", "operator": "max", "value": 3.0, "unit": "N"}]
        exp = _expected("READY", theme=tid, obstacles=[oid], skills=["PlanPath", "Reach", "Grasp"], constraints=constraint)

    elif category == "negative_contrast":
        alt_id = f"{cid}-excluded"
        objects.append(_obj(alt_id, target_cat, "white", 0.40, 0.18))
        templates = [
            "白色那个不要动，{p}拿起{c}{t}", "别选白色的，目标是{c}{t}，{p}把它抓起来",
            "排除白色{t}后，{p}取起{c}的那个", "{p}抓{c}{t}，白色同类保持原位",
            "不要碰白色{t}，只拿{c}{t}",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, c=color_cn, t=target_cn)
        exp = _expected("READY", theme=tid, obstacles=[alt_id], skills=["PlanPath", "Reach", "Grasp"])

    elif category == "multi_constraint":
        force = float(2 + local_i % 3)
        speed = [0.08, 0.10, 0.12][local_i % 3]
        templates = [
            "{p}以不高于{f:g}N的抓力拿起{c}{t}，移动速度不要超过{s:g}m/s",
            "{p}抓取{c}{t}；夹持力上限{f:g}牛顿，速度至多{s:g}m/s",
            "{p}把{c}{t}取起来，力别超过{f:g}N，同时限速{s:g}m/s",
            "{p}在最大抓力{f:g}N、最大速度{s:g}m/s的约束下拿起{c}{t}",
            "{p}轻拿{c}{t}，抓力不大于{f:g}N，运动速度不大于{s:g}m/s",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, f=force, s=speed, c=color_cn, t=target_cn)
        constraints = [
            {"parameter": "force_n", "operator": "max", "value": force, "unit": "N"},
            {"parameter": "velocity_ms", "operator": "max", "value": speed, "unit": "m/s"},
        ]
        exp = _expected("READY", theme=tid, skills=["Reach", "Grasp"], constraints=constraints)

    elif category == "multi_object_ambiguity":
        alt_id = f"{cid}-alternative"
        objects.append(_obj(alt_id, target_cat, color, 0.42, -0.12))
        templates = [
            "{p}拿起那个{c}{t}", "{p}把{c}{t}取起来", "{p}抓一个{c}{t}",
            "{p}选中{c}{t}并拿起", "{p}处理一下{c}{t}，先把它抓住",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, c=color_cn, t=target_cn)
        exp = _expected("NEEDS_CLARIFICATION")

    elif category == "explicit_conflict":
        templates = [
            "{p}拿起{c}{t}，但同时禁止抓取它",
            "{p}把{c}{t}抓起来，并且不要移动这件东西",
            "{p}完成{c}{t}的抓取，不过不允许机械臂接触它",
            "{p}取起{c}{t}，同时保持它完全不被拿起",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, c=color_cn, t=target_cn)
        exp = _expected("BLOCKED")

    elif category == "conditional_sequence":
        templates = [
            "{p}确认{c}{t}静止后先抓住它，然后放进{d}",
            "{p}等{c}{t}稳定下来，再取起并放到{d}",
            "{p}若{c}{t}当前没有移动，就先夹住它，随后放入{d}",
            "{p}在{c}{t}处于静止状态时，完成抓取后再摆放至{d}",
            "{p}待{c}{t}停止运动，依次执行拿起和放入{d}",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, c=color_cn, t=target_cn, d=dest_cn)
        exp = _expected("READY", "PLACE", tid, did, skills=["Reach", "Grasp", "Place"], sequence=True)

    elif category == "industrial_role_binding":
        templates = [
            "{p}避开{obs}，把{c}{t}上料到{d}",
            "{p}将{c}{t}转放至{d}，路径不得经过{obs}",
            "{p}抓取{c}{t}后送入{d}，全程绕开{obs}",
            "{p}完成{c}{t}到{d}的搬放，机械臂不要靠近{obs}",
            "{p}把{c}{t}从当前位置移放到{d}，规划时排除{obs}",
        ]
        instruction = templates[local_i % len(templates)].format(p=discourse, c=color_cn, t=target_cn, d=dest_cn, obs=obs_cn)
        exp = _expected("READY", "PLACE", tid, did, [oid], ["PlanPath", "Reach", "Place"])

    else:  # perception_robustness
        variant = local_i % 4
        if variant == 0:
            objects = [x for x in objects if x["object_id"] != tid]
            instruction = f"{discourse}拿起{color_cn}{target_cn}"
            exp = _expected("BLOCKED")
            exp["accepted_plan_statuses"] = ["BLOCKED", "NEEDS_CLARIFICATION"]
        elif variant == 1:
            alt_id = f"{cid}-low-confidence-peer"
            objects[0]["category_candidates"][0]["score"] = 0.51
            objects[0]["confidence"] = 0.51
            objects.append(_obj(alt_id, target_cat, color, 0.40, -0.12, confidence=0.50))
            instruction = f"{discourse}拿起{color_cn}{target_cn}"
            exp = _expected("NEEDS_CLARIFICATION")
        elif variant == 2:
            objects[0]["affordances"] = ["fixed"]
            instruction = f"{discourse}抓起固定在台面上的{color_cn}{target_cn}"
            exp = _expected("BLOCKED")
        else:
            objects[0]["pose"]["position"]["x"] = 2.4
            instruction = f"{discourse}拿起远处的{color_cn}{target_cn}"
            exp = _expected("BLOCKED")

    return {
        "case_id": cid,
        "scene": scene,
        "category": category,
        "difficulty": difficulty,
        "instruction": instruction,
        "observation_json": {
            "schema_version": "perception.v1", "frame_id": "robot_base", "unit": "m",
            "timestamp": f"2026-08-11T10:{global_i // 60:02d}:{global_i % 60:02d}+08:00",
            "objects": objects,
            "robot_state": {"available_skills": ["Reach", "Grasp", "Place", "PlanPath"],
                            "gripper": "empty", "homed": True},
        },
        "expected": exp,
        "input_contract": {
            "valid_json": True,
            "valid_schema": True,
            "perception_reliability": "degraded" if category == "perception_robustness" else "normal",
            "generated_independently_of_system_output": True,
        },
    }


def generate() -> dict:
    rng = random.Random(SEED)
    cases = []
    for scene in ("daily", "industrial"):
        index = 0
        for category, total in CATEGORY_COUNTS.items():
            per_scene = total // 2
            for local_i in range(per_scene):
                cases.append(_case(scene, category, local_i, index, rng))
                index += 1
    dataset = {
        "dataset_id": "open_complex_v1",
        "version": "1.0.0",
        "frozen": True,
        "seed": SEED,
        "description": "500-case open-language complex intent and degraded-perception benchmark",
        "cases": cases,
    }
    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    SHA.write_text(f"{sha}  {OUT.name}\n", encoding="utf-8")
    return dataset


if __name__ == "__main__":
    data = generate()
    print(f"generated={len(data['cases'])} path={OUT.name}")
