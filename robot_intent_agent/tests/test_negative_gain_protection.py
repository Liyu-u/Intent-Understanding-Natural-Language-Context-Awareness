from robot_intent_agent.planner.llm_planner import HybridRouter
from robot_intent_agent.planner.llm_planner import LLMPlanner


def test_negative_gain_protection_rejects_dropped_safety_fields():
    reasons = HybridRouter._negative_gain_reasons(
        {
            "obstacle": [{"mention": "glass cup"}],
            "user_constraints": [{"parameter": "force_n", "operator": "max", "max_value": 3.0}],
            "conditions": [{"condition_id": "c1"}],
            "steps": [{"step_index": 0}, {"step_index": 1}],
        },
        {
            "obstacle": [],
            "user_constraints": [],
            "conditions": [],
            "steps": [{"step_index": 0}],
        },
    )
    assert "DROPPED_RULE_OBSTACLE" in reasons
    assert "DROPPED_OR_CHANGED_RULE_CONSTRAINTS" in reasons
    assert "DROPPED_RULE_SEQUENCE_STEPS" in reasons


def test_semantic_only_accepts_frame_when_transport_bt_is_incomplete():
    planner = LLMPlanner(api_key="test-key")
    bt = planner._parse_response(
        {
            "intent_frame": {
                "schema_version": "1.0.0",
                "action": "FETCH",
                "theme": {"mention": "red bottle", "category": "bottle"},
            },
            "behavior_tree": {
                "type": "sequence",
                "name": "incomplete",
                "children": [{"type": "action", "name": "reach", "skill_name": "Reach", "target": "red bottle"}],
            },
        },
        "bring the red bottle",
        semantic_only=True,
    )
    assert bt.metadata["parsed_task"]["action"] == "FETCH"
