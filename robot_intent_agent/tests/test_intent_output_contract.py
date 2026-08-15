from __future__ import annotations

import json
from pathlib import Path

from robot_intent_agent.demo.web_ui import Pipeline
from robot_intent_agent.intent_output_adapter import build_intent_output
from robot_intent_agent.schemas.intent_output import IntentOutput
from robot_intent_agent.schemas.perception_observation import inference_observation


OBSERVATION = {
    "schema_version": "1.0.0",
    "message_type": "perception_observation",
    "observation_id": "obs_contract_001",
    "scene_id": "scene_contract_001",
    "coordinate_system": "robot_base",
    "objects": [
        {
            "object_id": "obj_001",
            "category_candidates": [{"name": "cup", "score": 0.93}],
            "pose": {"position": {"x": 0.35, "y": 0.12, "z": 0.75}},
            "geometry": {"type": "oriented_bbox_3d", "size": {
                "width": 0.06, "height": 0.12, "depth": 0.06,
            }},
            "appearance": {"color_candidates": [{"name": "transparent", "score": 0.88}]},
        }
    ],
}


def run_grasp(observation: dict) -> dict:
    return Pipeline().run(
        "\u6293\u4f4f\u676f\u5b50",
        json.dumps(observation, ensure_ascii=False),
        "\u7eaf\u89c4\u5219\u5f15\u64ce (\u6781\u901f)",
        "",
    )


def test_public_output_keeps_flat_contract_and_real_ids():
    result = run_grasp(OBSERVATION)
    output = result["intent_output"]
    validated = IntentOutput.model_validate(output)

    expected_fields = {
        "schema_version", "intent_id", "observation_id", "scene_id", "action",
        "target_object", "destination", "target_objects", "reference_object",
        "attributes", "resolved_attributes", "sort_criterion", "constraints",
        "plan_status", "execution_allowed", "confidence", "clarification",
        "errors", "unsupported_capabilities",
    }
    assert set(output) == expected_fields
    assert validated.target_object == "obj_001"
    assert validated.target_objects == ["obj_001"]
    assert validated.action == "grasp"
    assert validated.observation_id == "obs_contract_001"
    assert result["ir"].intent_output == output
    # A category ontology may legitimately infer a material from the public
    # category.  Leakage is tested by comparing enriched and clean outputs,
    # not by banning a valid inferred value.


def test_evaluation_truth_is_removed_recursively_and_does_not_change_output():
    enriched = json.loads(json.dumps(OBSERVATION))
    enriched["simulation_metadata"] = {
        "evaluation_only": True,
        "ground_truth_objects": [{
            "object_id": "obj_001", "material": "glass", "mass_kg": 0.2,
        }],
    }
    enriched["objects"][0]["evaluation"] = {
        "ground_truth": {"material": "glass"},
        "ground_truth_objects": [{"material": "glass"}],
    }
    clean = inference_observation(enriched)
    assert "simulation_metadata" not in clean
    assert "evaluation" not in clean["objects"][0]
    assert "simulation_metadata" in enriched
    assert "evaluation" in enriched["objects"][0]

    base_output = run_grasp(OBSERVATION)["intent_output"]
    truth_output = run_grasp(enriched)["intent_output"]
    assert truth_output == base_output


def test_adapter_blocks_a_fabricated_grounded_id():
    result = run_grasp(OBSERVATION)
    result["ir"].parsed_task.theme.entity_id = "invented_object"
    output = build_intent_output(
        result["ir"],
        observation=OBSERVATION,
        observation_id=OBSERVATION["observation_id"],
        scene_id=OBSERVATION["scene_id"],
    )
    assert output.plan_status == "BLOCKED"
    assert output.execution_allowed is False
    assert any(item.get("code") == "OUTPUT_UNKNOWN_ENTITY_ID" for item in output.errors)


def test_versioned_schema_declares_all_public_fields():
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "json_schema" / "intent_output_v1_1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$id"] == "intent_output.v1.1"
    assert set(schema["required"]) == {
        "schema_version", "intent_id", "observation_id", "scene_id", "action",
        "target_object", "destination", "target_objects", "reference_object",
        "attributes", "resolved_attributes", "sort_criterion", "constraints",
        "plan_status", "execution_allowed", "confidence", "clarification",
        "errors", "unsupported_capabilities",
    }
