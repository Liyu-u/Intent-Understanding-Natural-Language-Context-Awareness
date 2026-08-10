from robot_intent_agent.eval.downstream_json_quality import assess_downstream_json


PERCEPTION = {"objects": [{"object_id": "obj-1"}, {"object_id": "zone-1"}]}


def test_ready_requires_real_perception_ids():
    ok = assess_downstream_json({"plan_status": "READY", "target_object_id": "obj-1",
                                 "destination_id": "zone-1"}, PERCEPTION)
    assert ok.usable


def test_ready_rejects_fabricated_id():
    bad = assess_downstream_json({"plan_status": "READY", "target_object_id": "invented"}, PERCEPTION)
    assert not bad.usable
    assert "FABRICATED_TARGET_ID:invented" in bad.reasons


def test_blocked_is_valid_only_with_auditable_reason():
    assert assess_downstream_json({"plan_status": "BLOCKED", "blocking_reasons": ["FORCE_CONFLICT"]}, PERCEPTION).usable
    assert not assess_downstream_json({"plan_status": "BLOCKED"}, PERCEPTION).usable
