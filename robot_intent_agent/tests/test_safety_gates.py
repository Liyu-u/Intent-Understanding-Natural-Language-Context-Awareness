"""Regression tests for deterministic safety and perception gates."""
from types import SimpleNamespace

from robot_intent_agent.safety.action_conflict_checker import find_action_constraint_conflicts
from robot_intent_agent.safety.perception_quality import assess_perception_quality


def test_conflicting_numeric_bounds_block():
    task = SimpleNamespace(
        action="GRASP", theme=SimpleNamespace(mention="cup"), destination=None,
        support_surface=None, recipient=None, prohibitions=[], instruction="抓取杯子",
        user_constraints=[
            SimpleNamespace(parameter="force_n", operator="min", value=5, min_value=None, max_value=None, is_hard=True),
            SimpleNamespace(parameter="force_n", operator="max", value=2, min_value=None, max_value=None, is_hard=True),
        ],
    )
    assert any("force_n_min_gt_max" in reason for reason in find_action_constraint_conflicts(task))


def test_invalid_perception_is_blocked_and_ambiguous_is_clarification():
    invalid = assess_perception_quality({"objects": [{"category_candidates": [], "pose": {"position": {"x": 99}}}]})
    assert invalid["status"] == "BLOCKED"
    ambiguous = assess_perception_quality({"objects": [{"category_candidates": [
        {"name": "cup", "score": 0.61}, {"name": "bottle", "score": 0.56}
    ]}]})
    assert ambiguous["status"] == "NEEDS_CLARIFICATION"
