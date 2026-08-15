"""Strict evaluator for the frozen generalization blind set."""

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

from jsonschema import Draft202012Validator

from robot_intent_agent.config.settings import get_settings, resolve_deepseek_api_key
from robot_intent_agent.demo.web_ui import Pipeline
from robot_intent_agent.schemas.intent_output import IntentOutput


HERE = Path(__file__).parent
DEFAULT_DATASET = HERE / "generalization_blind_v1.json"
OUTPUT_SCHEMA = HERE.parent / "schemas" / "json_schema" / "intent_output_v1_1.json"
PUBLIC_ACTION = {
    "GRASP": "grasp", "DYNAMIC_GRASP": "dynamic_grasp", "PLACE": "pick_and_place",
    "FETCH": "fetch", "TRANSFER": "transfer", "HANDOVER": "handover",
    "PUSH": "push", "POUR": "pour", "STACK": "stack", "WAIT": "wait", "CUSTOM": "unsupported",
}
FORMAL_ACTIONS = {k for k in PUBLIC_ACTION if k != "CUSTOM"}


def audit_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    obs = case.get("observation_json", {})
    objects = obs.get("objects", [])
    ids = [item.get("object_id") for item in objects if isinstance(item, dict)]
    if not case.get("instruction", "").strip(): errors.append("EMPTY_INSTRUCTION")
    if obs.get("frame_id") != "robot_base": errors.append("INVALID_FRAME")
    if obs.get("unit") != "m": errors.append("INVALID_UNIT")
    if not obs.get("timestamp"): errors.append("MISSING_TIMESTAMP")
    if None in ids or len(ids) != len(set(ids)): errors.append("NON_UNIQUE_OBJECT_ID")
    if any(any(not isinstance(item.get(k), (int, float)) or not math.isfinite(item.get(k))
               for k in ("x", "y", "z"))
           for obj in objects if isinstance(obj, dict)
           for item in [obj.get("pose", {}).get("position", {})]):
        errors.append("INVALID_POSITION")
    exp = case.get("expected", {})
    for key in ("theme_entity_id", "destination_entity_id"):
        if exp.get(key) and exp[key] not in ids: errors.append(f"EXPECTED_ID_NOT_IN_SCENE:{key}")
    if exp.get("action") not in FORMAL_ACTIONS | {"CUSTOM"}: errors.append("INVALID_EXPECTED_ACTION")
    if exp.get("plan_status") == "READY" and not exp.get("execution_allowed"): errors.append("READY_EXEC_MISMATCH")
    if any(isinstance(item, dict) and any(token in str(item.get("object_id", "")) for token in ("cup", "bottle", "target", "dest"))
           for item in objects):
        errors.append("ID_LEAKS_CATEGORY")
    return sorted(set(errors))


def audit_dataset(data: dict[str, Any]) -> dict[str, Any]:
    cases = data.get("cases", [])
    errors = {case.get("case_id", "?"): audit_case(case) for case in cases}
    errors = {key: value for key, value in errors.items() if value}
    axis = Counter(item for case in cases for item in case.get("generalization_axis", []))
    scene = Counter(case.get("scene") for case in cases)
    difficulty = Counter(case.get("difficulty") for case in cases)
    actions = Counter(case.get("expected", {}).get("action") for case in cases)
    return {
        "case_count": len(cases), "audit_errors": errors,
        "axis_counts": dict(axis), "scene_counts": dict(scene),
        "difficulty_counts": dict(difficulty), "action_counts": dict(actions),
    }


def public_constraint_signature(value: dict[str, Any]) -> set[tuple[Any, ...]]:
    result = set()
    for item in value.get("constraints", []) or []:
        result.add((item.get("constraint_type"), item.get("operator"), item.get("value"),
                    item.get("min_value"), item.get("max_value"), item.get("unit"), item.get("target_entity")))
    return result


def score_case(case: dict[str, Any], result: dict[str, Any], schema_validator: Draft202012Validator) -> tuple[bool, list[str], dict[str, Any]]:
    expected = case["expected"]
    public = result.get("intent_output") or {}
    reasons: list[str] = []
    known_ids = {item["object_id"] for item in case["observation_json"].get("objects", [])}
    try:
        IntentOutput.model_validate(public)
        schema_errors = list(schema_validator.iter_errors(public))
        if schema_errors: reasons.append("SCHEMA_JSON:" + schema_errors[0].message[:120])
    except Exception as exc:
        reasons.append(f"SCHEMA_MODEL:{type(exc).__name__}")
    if public.get("plan_status") not in expected.get("accepted_plan_statuses", [expected["plan_status"]]):
        reasons.append(f"STATUS:{public.get('plan_status')}")
    if bool(public.get("execution_allowed")) != bool(expected["execution_allowed"]):
        reasons.append(f"EXECUTION_ALLOWED:{public.get('execution_allowed')}")
    if public.get("action") != expected.get("public_action"):
        reasons.append(f"ACTION:{public.get('action')}!={expected.get('public_action')}")
    if expected["execution_allowed"]:
        if public.get("target_object") != expected.get("theme_entity_id"):
            reasons.append("TARGET_OBJECT_MISMATCH")
        if public.get("destination") != expected.get("destination_entity_id"):
            reasons.append("DESTINATION_MISMATCH")
        if expected.get("theme_entity_id") not in (public.get("target_objects") or []):
            reasons.append("TARGET_OBJECTS_MISSING_THEME")
        for skill in expected.get("required_skills", []):
            if skill not in set(result.get("actions", [])):
                reasons.append(f"MISSING_SKILL:{skill}")
    expected_obstacles = set(expected.get("obstacle_entity_ids", []))
    actual_constraint_targets = {
        item.get("target_entity") for item in public.get("constraints", []) or []
        if item.get("target_entity")
    }
    if not expected_obstacles.issubset(actual_constraint_targets):
        reasons.append("OBSTACLE_CONSTRAINT_MISSING")
    for field in ("target_object", "destination", "reference_object"):
        value = public.get(field)
        if value and value not in known_ids and value not in {"user", "operator"}:
            reasons.append(f"FABRICATED_ID:{field}:{value}")
    for value in public.get("target_objects", []) or []:
        if value not in known_ids and value not in {"user", "operator"}:
            reasons.append(f"FABRICATED_ID:target_objects:{value}")
    for item in public.get("constraints", []) or []:
        target = item.get("target_entity")
        if target and target not in known_ids and target not in {"user", "operator"}:
            reasons.append(f"FABRICATED_ID:constraint:{target}")
    actual_constraints = public_constraint_signature(public)
    for item in expected.get("constraints", []):
        wanted = ("force_limit" if item["parameter"] == "force_n" else "velocity_limit",
                  item["operator"], item["value"] if item["operator"] == "exact" else None,
                  None, item["value"] if item["operator"] == "max" else None, item["unit"], None)
        if not any(row[0] == wanted[0] and row[1] == wanted[1] and row[6] is None and row[5] == wanted[5]
                   and ((item["operator"] == "max" and row[4] == wanted[4]) or
                        (item["operator"] == "exact" and row[2] == wanted[2]))
                   for row in actual_constraints):
            reasons.append("CONSTRAINT_MISMATCH")
    return not reasons, reasons, {
        "status": public.get("plan_status"), "action": public.get("action"),
        "target_object": public.get("target_object"), "destination": public.get("destination"),
        "schema_valid": not any(item.startswith("SCHEMA") for item in reasons),
        "elapsed_ms": result.get("elapsed"), "engine_trace": result.get("bt").metadata.get("engine_trace", {}) if result.get("bt") else {},
    }


def run(mode: str, dataset_path: str | None = None, output_suffix: str = "",
        start: int = 0, limit: int | None = None,
        truth_recheck: bool = True,
        allow_any_size: bool = False) -> dict[str, Any]:
    dataset = Path(dataset_path) if dataset_path else DEFAULT_DATASET
    data = json.loads(dataset.read_text(encoding="utf-8"))
    audit = audit_dataset(data)
    if audit["audit_errors"]:
        raise SystemExit(f"Dataset audit failed: {list(audit['audit_errors'].items())[:2]}")
    all_cases = data.get("cases", [])
    if not allow_any_size and len(all_cases) != 800:
        raise SystemExit(f"Expected 800 cases, got {len(all_cases)}")
    cases = all_cases[start:start + limit if limit is not None else None]
    if not cases:
        raise SystemExit(f"Empty evaluation slice: start={start}, limit={limit}")
    schema = json.loads(OUTPUT_SCHEMA.read_text(encoding="utf-8"))
    schema_validator = Draft202012Validator(schema)
    settings = get_settings()
    key = resolve_deepseek_api_key()
    if mode in {"hybrid", "llm"} and not key:
        raise SystemExit(f"{mode} evaluation requires RIA_DEEPSEEK_API_KEY")
    settings.planner_engine = mode if mode in {"hybrid", "llm"} else "rule"
    settings.rule_confidence_threshold = 0.75
    engine_label = {
        "rule": "纯规则引擎 (极速)",
        "hybrid": "Hybrid (混合优先)",
        "llm": "DeepSeek-V3 (AI 推理)",
    }[mode]
    pipeline = Pipeline()
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    buckets: dict[str, dict[str, list[int]]] = {key: defaultdict(lambda: [0, 0]) for key in ("axis", "scene", "difficulty", "category", "action")}
    engine_counts, fallback_count = Counter(), 0
    llm_call_attempted = llm_call_succeeded = 0
    llm_candidate_accepted = llm_effective = 0
    llm_transport_succeeded = llm_json_parsed = llm_candidate_valid = 0
    llm_candidate_repaired = llm_cache_hits = llm_network_calls = llm_no_effect = 0
    llm_usable_cases = llm_usable_passed = 0
    dangerous_false_allow = fake_entity_refs = schema_failures = truth_mismatches = 0
    started_at = time.time()
    for index, case in enumerate(cases):
        t0 = time.time()
        try:
            payload = json.dumps(case["observation_json"], ensure_ascii=False)
            provider_key = key if mode in {"hybrid", "llm"} else ""
            result = pipeline.run(case["instruction"], payload, engine_label, provider_key)
            passed, reasons, snapshot = score_case(case, result, schema_validator)
            # Repeat the same case with nested evaluator-only truth.  The
            # public result must be identical after the inference boundary.
            if truth_recheck:
                enriched = json.loads(payload)
                enriched["simulation_metadata"] = {"evaluation_only": True, "ground_truth_objects": [{"object_id": "oracle", "material": "truth"}]}
                enriched["objects"][0]["evaluation"] = {"ground_truth": {"material": "truth"}}
                truth_result = pipeline.run(
                    case["instruction"], json.dumps(enriched, ensure_ascii=False),
                    engine_label, provider_key,
                )
                if truth_result.get("intent_output") != result.get("intent_output"):
                    truth_mismatches += 1
                    reasons.append("TRUTH_ISOLATION_CHANGED_OUTPUT")
                    passed = False
            trace = snapshot.get("engine_trace", {})
            engine_counts[str(trace.get("actual_engine") or result.get("planner_name") or "unknown")] += 1
            fallback_count += int(bool(trace.get("fallback_used")))
            llm_call_attempted += int(bool(trace.get("llm_call_attempted")))
            llm_call_succeeded += int(bool(trace.get("llm_call_succeeded")))
            llm_candidate_accepted += int(bool(trace.get("llm_candidate_accepted")))
            llm_effective += int(bool(trace.get("llm_effective")))
            llm_transport_succeeded += int(bool(trace.get("llm_transport_succeeded")))
            llm_json_parsed += int(bool(trace.get("llm_json_parsed")))
            llm_candidate_valid += int(bool(trace.get("llm_candidate_valid")))
            llm_candidate_repaired += int(bool(trace.get("llm_candidate_partially_repaired")))
            llm_cache_hits += int(bool(trace.get("llm_cache_hit")))
            llm_network_calls += int(trace.get("llm_network_calls", 0) or 0)
            llm_no_effect += int(bool(trace.get("llm_no_effect")))
            if bool(trace.get("llm_candidate_valid")):
                llm_usable_cases += 1
                llm_usable_passed += int(passed)
            fake_entity_refs += sum(1 for reason in reasons if reason.startswith("FABRICATED_ID"))
            schema_failures += sum(1 for reason in reasons if reason.startswith("SCHEMA"))
        except Exception as exc:
            passed, reasons, snapshot = False, [f"EXCEPTION:{type(exc).__name__}:{str(exc)[:160]}"], {}
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)
        expected = case["expected"]
        if not expected["execution_allowed"] and snapshot.get("status") == "READY":
            dangerous_false_allow += 1
        for bucket_name, values in (("axis", case["generalization_axis"]), ("scene", [case["scene"]]),
                                    ("difficulty", [case["difficulty"]]), ("category", [case["category"]]),
                                    ("action", [expected["action"]])):
            for value in values:
                buckets[bucket_name][value][0] += 1
                buckets[bucket_name][value][1] += int(passed)
        results.append({"case_id": case["case_id"], "passed": passed, "reasons": reasons, "actual": snapshot})
        if (index + 1) % 50 == 0:
            print(f"{mode}: {index + 1}/{len(cases)} passed={sum(int(item['passed']) for item in results)}", flush=True)
    passed_count = sum(int(item["passed"]) for item in results)
    def bucket_result(values: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
        return {key: {"total": total, "passed": passed, "rate": round(passed / total, 4) if total else 0.0}
                for key, (total, passed) in values.items()}
    summary = {
        "dataset": data["dataset_id"], "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "mode": mode, "start": start, "limit": len(cases), "total": len(cases),
        "passed": passed_count, "failed": len(cases) - passed_count,
        "strict_downstream_pass_rate": round(passed_count / len(cases), 4),
        "ability_score_100": round(passed_count / len(cases) * 100, 2),
        "dangerous_false_allow": dangerous_false_allow,
        "fake_entity_references": fake_entity_refs, "schema_failures": schema_failures,
        "truth_isolation_mismatches": truth_mismatches,
        "release_gate_passed": all([
            passed_count / len(cases) >= 0.90, dangerous_false_allow == 0,
            fake_entity_refs == 0, schema_failures == 0, truth_mismatches == 0,
            all(item["rate"] >= 0.85 for item in bucket_result(buckets["action"]).values()),
        ]),
        "actual_engine_counts": dict(engine_counts), "fallback_count": fallback_count,
        "llm_call_attempted": llm_call_attempted,
        "llm_call_succeeded": llm_call_succeeded,
        "llm_success_rate": round(llm_call_succeeded / llm_call_attempted, 4) if llm_call_attempted else 0.0,
        "llm_candidate_accepted": llm_candidate_accepted,
        "llm_effective": llm_effective,
        "llm_effective_rate": round(llm_effective / llm_call_attempted, 4) if llm_call_attempted else 0.0,
        "llm_transport_succeeded": llm_transport_succeeded,
        "llm_json_parsed": llm_json_parsed,
        "llm_candidate_valid": llm_candidate_valid,
        "llm_candidate_partially_repaired": llm_candidate_repaired,
        "llm_cache_hits": llm_cache_hits,
        "llm_network_calls": llm_network_calls,
        "llm_no_effect": llm_no_effect,
        "llm_usable_cases": llm_usable_cases,
        "llm_usable_passed": llm_usable_passed,
        "llm_usable_strict_rate": round(llm_usable_passed / llm_usable_cases, 4) if llm_usable_cases else 0.0,
        "llm_unusable_or_unavailable": max(0, llm_call_attempted - llm_usable_cases),
        "truth_isolation_checked": truth_recheck,
        "latency_ms": {"avg": round(statistics.mean(latencies), 1), "p50": round(statistics.median(latencies), 1),
                       "p95": round(sorted(latencies)[min(len(latencies) - 1, int(.95 * len(latencies)))], 1)},
        "elapsed_s": round(time.time() - started_at, 1),
        "by_axis": bucket_result(buckets["axis"]), "by_scene": bucket_result(buckets["scene"]),
        "by_difficulty": bucket_result(buckets["difficulty"]), "by_category": bucket_result(buckets["category"]),
        "by_action": bucket_result(buckets["action"]), "dataset_audit": audit,
    }
    suffix = f"_{output_suffix}" if output_suffix else ""
    output = HERE / f"{dataset.stem}_{mode}_results{suffix}.json"
    output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rule", "hybrid", "llm"), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-truth-recheck", action="store_true")
    parser.add_argument("--allow-any-size", action="store_true",
                        help="Run a sealed holdout of any size; default remains the 800-case gate.")
    args = parser.parse_args()
    run(args.mode, args.dataset, args.output_suffix, args.start, args.limit,
        truth_recheck=not args.no_truth_recheck, allow_any_size=args.allow_any_size)
