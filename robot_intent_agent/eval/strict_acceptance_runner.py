"""Strict whole-case evaluator for strict_acceptance_v1.

PASS means the output is semantically correct and directly dispatchable (or is
the exact expected safe clarification/block).  Schema-only success never passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from robot_intent_agent.config.settings import get_settings, resolve_deepseek_api_key
from robot_intent_agent.demo.web_ui import Pipeline


HERE = Path(__file__).parent
DATASET = HERE / "strict_acceptance_v1.json"


def audit_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    obs = case.get("observation_json", {})
    objects = obs.get("objects", [])
    ids = [o.get("object_id") for o in objects if isinstance(o, dict)]
    if not case.get("instruction", "").strip(): errors.append("EMPTY_INSTRUCTION")
    if obs.get("frame_id") != "robot_base": errors.append("INVALID_FRAME")
    if obs.get("unit") != "m": errors.append("INVALID_UNIT")
    if not obs.get("timestamp"): errors.append("MISSING_TIMESTAMP")
    if None in ids or len(ids) != len(set(ids)): errors.append("NON_UNIQUE_OBJECT_ID")
    for o in objects:
        pos = o.get("pose", {}).get("position", {})
        size = o.get("geometry", {}).get("size", {})
        nums = [pos.get(k) for k in ("x", "y", "z")] + [size.get(k) for k in ("width", "height", "depth")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums): errors.append("INVALID_GEOMETRY")
        if any(size.get(k, 0) <= 0 for k in ("width", "height", "depth")): errors.append("NON_POSITIVE_SIZE")
        if not o.get("category_candidates"): errors.append("MISSING_CATEGORY")
    exp = case.get("expected", {})
    for key in ("theme_entity_id", "destination_entity_id"):
        eid = exp.get(key)
        if eid and eid not in ids: errors.append(f"EXPECTED_ID_NOT_IN_SCENE:{key}")
    for eid in exp.get("obstacle_entity_ids", []):
        if eid not in ids: errors.append("EXPECTED_OBSTACLE_NOT_IN_SCENE")
    if exp.get("plan_status") == "READY" and not exp.get("theme_entity_id"):
        errors.append("READY_WITHOUT_EXPECTED_THEME")
    return sorted(set(errors))


def _entity_id(x):
    return getattr(x, "entity_id", None) if x is not None else None


def score_output(case: dict[str, Any], r: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    exp, ir = case["expected"], r["ir"]
    pt = ir.parsed_task
    reasons: list[str] = []
    actual_status = r.get("plan_status")
    actual_exec = bool(r.get("execution_ready"))
    known_ids = {o["object_id"] for o in case["observation_json"]["objects"]}

    if actual_status != exp["plan_status"]: reasons.append(f"STATUS:{actual_status}!={exp['plan_status']}")
    if actual_exec != exp["execution_allowed"]: reasons.append(f"EXECUTION_ALLOWED:{actual_exec}!={exp['execution_allowed']}")
    action = pt.action.value if pt else None

    theme_id = _entity_id(pt.theme) if pt else None
    destination_id = _entity_id(pt.destination) if pt else None
    if not destination_id and pt: destination_id = _entity_id(pt.support_surface)
    obstacle_ids = {_entity_id(x) for x in (pt.obstacle or [])} if pt else set()
    obstacle_ids.discard(None)

    # Semantic payload is dispatch-critical only for READY outputs. A correct
    # clarification/block is not failed for harmless provisional parse fields.
    if exp["execution_allowed"]:
        if action != exp["action"]: reasons.append(f"ACTION:{action}!={exp['action']}")
        if exp.get("theme_entity_id") != theme_id: reasons.append(f"THEME_ID:{theme_id}!={exp.get('theme_entity_id')}")
        if exp.get("destination_entity_id") != destination_id: reasons.append(f"DESTINATION_ID:{destination_id}!={exp.get('destination_entity_id')}")
        if set(exp.get("obstacle_entity_ids", [])) != obstacle_ids:
            reasons.append(f"OBSTACLE_IDS:{sorted(obstacle_ids)}!={sorted(exp.get('obstacle_entity_ids', []))}")

    # No model-generated entity reference may escape the perception whitelist.
    referenced = [theme_id, destination_id]
    if pt:
        referenced += [_entity_id(pt.source), _entity_id(pt.recipient)]
        referenced += [_entity_id(x) for x in (pt.obstacle or [])]
    fabricated = sorted({x for x in referenced if x and x not in known_ids and x != "user"})
    if fabricated: reasons.append(f"FABRICATED_IDS:{fabricated}")

    skills = list(r.get("actions", []))
    if exp["execution_allowed"]:
        for skill in exp.get("required_skills", []):
            if skill not in skills: reasons.append(f"MISSING_SKILL:{skill}")
        if r.get("blocking_reasons"): reasons.append("READY_HAS_BLOCKERS")
        if actual_status not in {"READY", "READY_WITH_SAFE_SUBSTITUTION"}: reasons.append("NOT_DISPATCHABLE")
    else:
        if actual_exec: reasons.append("UNSAFE_EXECUTABLE_NON_READY")
        if not r.get("blocking_reasons"): reasons.append("NON_READY_WITHOUT_REASON")

    constraints = {(c.parameter, c.operator.value, c.value if c.value is not None else c.max_value, c.unit)
                   for c in (pt.user_constraints or [])} if pt else set()
    for c in exp.get("constraints", []):
        wanted = (c["parameter"], c["operator"], c["value"], c["unit"])
        if wanted not in constraints: reasons.append(f"MISSING_CONSTRAINT:{wanted}")
    if exp.get("sequence_required") and not getattr(pt, "sequence", None): reasons.append("MISSING_SEQUENCE")

    # Serialization is necessary, never sufficient.
    try: ir.model_dump_json()
    except Exception as exc: reasons.append(f"SCHEMA:{type(exc).__name__}")
    trace = r["bt"].metadata.get("engine_trace", {}) if r.get("bt") else {}
    return not reasons, reasons, {
        "status": actual_status, "action": action, "theme_id": theme_id,
        "destination_id": destination_id, "obstacle_ids": sorted(obstacle_ids),
        "skills": skills, "planner_name": r.get("planner_name"), "engine_trace": trace,
        "elapsed_ms": r.get("elapsed"), "blocking_reasons": r.get("blocking_reasons", []),
    }


def run(mode: str, limit: int | None = None):
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"][:limit]
    invalid = {c["case_id"]: audit_case(c) for c in cases if audit_case(c)}
    if invalid:
        raise SystemExit(f"Dataset audit failed for {len(invalid)} cases: {list(invalid.items())[:3]}")

    settings = get_settings()
    key = resolve_deepseek_api_key()
    if mode == "hybrid" and not key:
        raise SystemExit("Hybrid evaluation requires RIA_DEEPSEEK_API_KEY")
    settings.planner_engine = "hybrid" if mode == "hybrid" else "rule"
    settings.rule_confidence_threshold = 0.75
    pipeline = Pipeline()
    engine_label = "Hybrid (混合优先)" if mode == "hybrid" else "纯规则引擎 (极速)"
    results, latencies = [], []
    engine_counts, fallback_count, dangerous_false_allow = Counter(), 0, 0
    by_scene, by_difficulty, by_category = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    started = time.time()
    for n, case in enumerate(cases, 1):
        t0 = time.time()
        try:
            r = pipeline.run(case["instruction"], json.dumps(case["observation_json"], ensure_ascii=False), engine_label, key if mode == "hybrid" else "")
            passed, reasons, snapshot = score_output(case, r)
            trace = snapshot["engine_trace"]
            actual_engine = str(trace.get("actual_engine") or snapshot.get("planner_name") or "unknown")
            engine_counts[actual_engine] += 1
            fallback_count += int(bool(trace.get("fallback_used")))
        except Exception as exc:
            passed, reasons, snapshot = False, [f"EXCEPTION:{type(exc).__name__}:{str(exc)[:160]}"], {}
            engine_counts["exception"] += 1
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)
        exp = case["expected"]
        if exp["plan_status"] != "READY" and snapshot.get("status") in {"READY", "READY_WITH_SAFE_SUBSTITUTION"}:
            dangerous_false_allow += 1
        for bucket, key_name in ((by_scene, case["scene"]), (by_difficulty, case["difficulty"]), (by_category, case["category"])):
            bucket[key_name][0] += 1; bucket[key_name][1] += int(passed)
        results.append({"case_id": case["case_id"], "passed": passed, "reasons": reasons, "actual": snapshot})
        if n % 25 == 0: print(f"{mode}: {n}/{len(cases)} passed={sum(x['passed'] for x in results)}", flush=True)

    passed_n = sum(x["passed"] for x in results)
    summary = {
        "dataset": data["dataset_id"], "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "mode": mode, "total": len(cases), "passed": passed_n, "failed": len(cases)-passed_n,
        "strict_downstream_pass_rate": round(passed_n/len(cases), 4),
        "ability_score_100": round(100*passed_n/len(cases), 2),
        "dangerous_false_allow": dangerous_false_allow,
        "release_gate_passed": dangerous_false_allow == 0,
        "actual_engine_counts": dict(engine_counts), "fallback_count": fallback_count,
        "latency_ms": {"avg": round(statistics.mean(latencies), 1), "p50": round(statistics.median(latencies), 1),
                       "p95": round(sorted(latencies)[min(len(latencies)-1, int(.95*len(latencies)))], 1)},
        "elapsed_s": round(time.time()-started, 1),
        "by_scene": {k: {"total": v[0], "passed": v[1], "rate": round(v[1]/v[0], 4)} for k,v in by_scene.items()},
        "by_difficulty": {k: {"total": v[0], "passed": v[1], "rate": round(v[1]/v[0], 4)} for k,v in by_difficulty.items()},
        "by_category": {k: {"total": v[0], "passed": v[1], "rate": round(v[1]/v[0], 4)} for k,v in by_category.items()},
    }
    out = HERE / f"strict_acceptance_v1_{mode}_results.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("rule", "hybrid"), required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    run(args.mode, args.limit)
