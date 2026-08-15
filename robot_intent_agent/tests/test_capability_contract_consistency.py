"""The frozen capability contract is the source of truth for the domain."""

import json
from pathlib import Path

from robot_intent_agent.domain.action_schemas import ACTION_SCHEMAS
from robot_intent_agent.schemas.intent_frame import ActionKind
from robot_intent_agent.task_semantics import TaskActionKind


ROOT = Path(__file__).parents[1]


def _contract():
    return json.loads((ROOT / "domain" / "capability_contract_v1.json").read_text(encoding="utf-8"))


def test_contract_and_runtime_action_enums_are_identical():
    contract_actions = set(_contract()["actions"])
    runtime_actions = set(ACTION_SCHEMAS) - {"CUSTOM"}
    assert contract_actions == runtime_actions
    assert contract_actions == {item.value for item in ActionKind} - {"CUSTOM"}
    assert contract_actions == {item.value for item in TaskActionKind} - {"CUSTOM"}


def test_contract_roles_and_fetch_alternative_role_are_explicit():
    contract = _contract()
    assert "condition" in contract["roles"]
    fetch = ACTION_SCHEMAS["FETCH"]
    assert fetch.required_roles == ("theme",)
    assert ("destination", "recipient") in fetch.required_any_roles


def test_llm_only_candidate_can_supply_missing_rule_event_and_entity():
    from robot_intent_agent.schemas.semantic_task_graph import (
        EvidenceSpan, SemanticCandidate, SemanticEntity, SemanticEvent,
        SemanticTaskGraph,
    )
    from robot_intent_agent.semantic_reasoner.semantic_fusion import SemanticFusion

    evidence = EvidenceSpan(value="送到我这里", source_text="送到我这里", start=0, end=6)
    graph = SemanticTaskGraph(
        instruction="送到我这里",
        entities=[
            SemanticEntity(local_ref="llm-theme", mention="透明容器", category="container",
                           attributes={"color": "transparent"}, evidence=[evidence]),
            SemanticEntity(local_ref="llm-user", mention="我这里", category="human",
                           attributes={}, evidence=[evidence]),
        ],
        events=[SemanticEvent(event_id="llm-event-1", action="FETCH",
                              theme_ref="llm-theme", recipient_ref="llm-user",
                              evidence_span="送到我这里", evidence=[evidence])],
    )
    candidate = SemanticCandidate.from_graph(graph, confidence=0.95, source="llm")
    rule = SemanticCandidate.from_graph(
        SemanticTaskGraph(instruction="送到我这里"), confidence=0.1, source="rule"
    )
    fused, audit = SemanticFusion().fuse(rule, candidate, "送到我这里")
    assert fused.graph.events[0].action == "FETCH"
    assert fused.graph.events[0].theme_ref == "llm-theme"
    assert any(item.decision == "ACCEPT_LLM_DELTA" for item in audit)
