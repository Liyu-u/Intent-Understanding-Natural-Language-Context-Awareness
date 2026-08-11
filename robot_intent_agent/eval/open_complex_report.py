"""Build a dimension-level report for the frozen open_complex_v1 benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
DATASET = HERE / "open_complex_v1.json"


def _rate(passed: int, total: int) -> dict:
    return {"passed": passed, "total": total,
            "rate": round(passed / total, 4) if total else None}


def analyze(mode: str) -> dict:
    cases = {c["case_id"]: c for c in json.loads(DATASET.read_text(encoding="utf-8"))["cases"]}
    result_path = HERE / f"open_complex_v1_{mode}_results.json"
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    rows = artifact["results"]
    dimensions = {
        "action_understanding": [0, 0],
        "theme_grounding": [0, 0],
        "destination_role": [0, 0],
        "obstacle_negation": [0, 0],
        "numeric_constraints": [0, 0],
        "sequence_preservation": [0, 0],
        "safe_non_execution": [0, 0],
        "strict_downstream_ready": [0, 0],
    }
    failure_codes = Counter()
    for row in rows:
        case = cases[row["case_id"]]
        expected = case["expected"]
        reasons = row.get("reasons", [])
        codes = {r.split(":", 1)[0] for r in reasons}
        for code in codes:
            failure_codes[code] += 1

        if expected["execution_allowed"]:
            for name, bad_code in (
                ("action_understanding", "ACTION"),
                ("theme_grounding", "THEME_ID"),
            ):
                dimensions[name][1] += 1
                dimensions[name][0] += int(bad_code not in codes)
            if expected.get("destination_entity_id"):
                dimensions["destination_role"][1] += 1
                dimensions["destination_role"][0] += int("DESTINATION_ID" not in codes)
            if expected.get("obstacle_entity_ids"):
                dimensions["obstacle_negation"][1] += 1
                dimensions["obstacle_negation"][0] += int("OBSTACLE_IDS" not in codes)
            if expected.get("constraints"):
                dimensions["numeric_constraints"][1] += 1
                dimensions["numeric_constraints"][0] += int("MISSING_CONSTRAINT" not in codes)
            if expected.get("sequence_required"):
                dimensions["sequence_preservation"][1] += 1
                dimensions["sequence_preservation"][0] += int("MISSING_SEQUENCE" not in codes)
        else:
            dimensions["safe_non_execution"][1] += 1
            unsafe = "UNSAFE_EXECUTABLE_NON_READY" in codes
            dimensions["safe_non_execution"][0] += int(not unsafe)

        dimensions["strict_downstream_ready"][1] += 1
        dimensions["strict_downstream_ready"][0] += int(row.get("passed", False))

    return {
        "mode": mode,
        "source_result": result_path.name,
        "summary": artifact["summary"],
        "dimensions": {k: _rate(v[0], v[1]) for k, v in dimensions.items()},
        "top_failure_codes": dict(failure_codes.most_common()),
    }


def main() -> None:
    report = {
        "dataset": "open_complex_v1",
        "interpretation": {
            "strict_downstream_ready": "Only fully correct, directly consumable JSON passes.",
            "safe_non_execution": "Ambiguous, conflicting, or unreliable perception must not execute.",
            "warning": "This benchmark is frozen after the first rule/hybrid measurement and must not become a rule fixture.",
        },
        "rule": analyze("rule"),
        "hybrid": analyze("hybrid"),
    }
    out = HERE / "open_complex_v1_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
