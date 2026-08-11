import json
from collections import Counter
from pathlib import Path

from robot_intent_agent.eval.strict_acceptance_runner import audit_case


DATASET = Path(__file__).parents[1] / "eval" / "strict_acceptance_v1_1.json"


def _cases():
    return json.loads(DATASET.read_text(encoding="utf-8"))["cases"]


def test_strict_dataset_has_required_frozen_distribution():
    cases = _cases()
    assert len(cases) == 600
    assert Counter(c["scene"] for c in cases) == {"daily": 300, "industrial": 300}
    assert Counter(c["difficulty"] for c in cases) == {"simple": 180, "medium": 300, "complex": 120}
    assert Counter(c["expected"]["plan_status"] for c in cases) == {
        "READY": 420, "NEEDS_CLARIFICATION": 90, "BLOCKED": 90,
    }


def test_every_strict_dataset_input_passes_preflight_audit():
    failures = {c["case_id"]: audit_case(c) for c in _cases() if audit_case(c)}
    assert failures == {}


def test_ready_cases_have_dispatch_contract_and_real_perception_ids():
    for case in _cases():
        expected = case["expected"]
        if expected["plan_status"] != "READY":
            continue
        known = {o["object_id"] for o in case["observation_json"]["objects"]}
        assert expected["execution_allowed"] is True
        assert expected["theme_entity_id"] in known
        assert expected["required_skills"]
        if expected["destination_entity_id"]:
            assert expected["destination_entity_id"] in known
        assert set(expected["obstacle_entity_ids"]).issubset(known)


def test_category_labels_are_present_in_non_ready_language():
    cues = {
        "role_binding": ("放到", "放进", "放入", "摆放"),
        "negation_obstacle": ("不要碰", "避开"),
        "numeric_constraints": ("N", "牛顿"),
        "condition_sequence": ("先", "然后", "再"),
    }
    for case in _cases():
        if case["expected"]["plan_status"] == "READY":
            continue
        if case["category"] not in cues:
            continue
        assert any(cue in case["instruction"] for cue in cues[case["category"]])


def test_large_surfaces_have_physically_plausible_dimensions():
    minimum_width = {"table": 0.5, "workbench": 0.5, "inspection_zone": 0.4}
    for case in _cases():
        for obj in case["observation_json"]["objects"]:
            category = obj["category_candidates"][0]["name"]
            if category in minimum_width:
                assert obj["geometry"]["size"]["width"] >= minimum_width[category]
