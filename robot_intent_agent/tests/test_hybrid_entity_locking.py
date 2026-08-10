"""Regression tests for the Hybrid semantic/grounding boundary."""

from robot_intent_agent.planner.llm_planner import normalize_intent_frame
from robot_intent_agent.schemas.intent_frame import IntentFrame
from robot_intent_agent.scene_builder import RawObjectPercept, SemanticSceneBuilder
from robot_intent_agent.task_semantics import load_parsed_task_from_bt


def _scene():
    return SemanticSceneBuilder().build([
        RawObjectPercept(
            name="red cup", object_id="cup-01", x=0.1, y=0.1, z=0.05,
            width=0.07, height=0.1, depth=0.07, color="red", material="plastic",
        ),
        RawObjectPercept(
            name="tray", object_id="tray-01", x=0.3, y=0.1, z=0.02,
            width=0.3, height=0.03, depth=0.2, color="white", material="plastic",
        ),
    ])


def test_place_frame_derives_support_surface_without_id():
    frame = IntentFrame.model_validate({
        "schema_version": "1.0.0",
        "action": "PLACE",
        "theme": {"mention": "red cup", "category": "cup"},
        "destination": {"mention": "tray", "category": "tray", "required_affordances": ["support_surface"]},
    })
    normalized = normalize_intent_frame(frame)
    assert normalized["support_surface"]["mention"] == "tray"
    assert normalized["support_surface"]["entity_id"] is None


def test_hybrid_clears_and_rebinds_llm_entity_id():
    frame = IntentFrame.model_validate({
        "schema_version": "1.0.0",
        "action": "GRASP",
        "theme": {"mention": "red cup", "category": "cup"},
    })
    parsed = normalize_intent_frame(frame)
    # Simulate a malicious/legacy LLM response that invents an existing ID.
    parsed["theme"]["entity_id"] = "tray-01"
    scene = _scene()
    task = load_parsed_task_from_bt(
        "抓住 red cup",
        {"parsed_task": parsed, "engine_trace": {"actual_engine": "DeepSeek"}},
        scene=scene,
    )
    expected_id = next(obj.id for obj in scene.objects if obj.name == "red cup")
    assert task.theme.entity_id == expected_id
    assert any("cleared_llm_entity_id:theme=tray-01" in n for n in task.notes)
