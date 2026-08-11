import hashlib
import json
from collections import Counter
from pathlib import Path

from robot_intent_agent.eval.strict_acceptance_runner import audit_case


EVAL = Path(__file__).parent.parent / "eval"
DATASET = EVAL / "open_complex_v1.json"


def _cases():
    return json.loads(DATASET.read_text(encoding="utf-8"))["cases"]


def test_open_complex_dataset_is_large_balanced_and_complex():
    cases = _cases()
    assert len(cases) == 500
    assert Counter(c["scene"] for c in cases) == {"daily": 250, "industrial": 250}
    assert Counter(c["difficulty"] for c in cases) == {"complex": 390, "medium": 110}
    assert len({c["instruction"] for c in cases}) >= 450


def test_open_complex_input_contract_has_no_format_defects():
    failures = {c["case_id"]: audit_case(c) for c in _cases() if audit_case(c)}
    assert failures == {}


def test_open_complex_categories_cover_required_capabilities():
    categories = Counter(c["category"] for c in _cases())
    assert set(categories) == {
        "open_paraphrase", "composite_pick_place", "context_pronoun",
        "negative_contrast", "multi_constraint", "multi_object_ambiguity",
        "explicit_conflict", "conditional_sequence", "industrial_role_binding",
        "perception_robustness",
    }
    assert min(categories.values()) >= 40


def test_open_complex_hash_is_frozen():
    expected = (EVAL / "open_complex_v1.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == expected
