"""Focused tests for the domain-restricted semantic compiler boundary."""

from robot_intent_agent.constraint.constraint_compiler import compile_feasible_domain
from robot_intent_agent.domain.action_schemas import get_action_schema
from robot_intent_agent.semantic_parser.rule_semantic_parser import RuleSemanticParser
from robot_intent_agent.semantic_parser.action_parser import parse_action_candidates
from robot_intent_agent.semantic_compiler import SemanticCompiler
from robot_intent_agent.schemas.semantic_task_graph import (
    SemanticCandidate,
    SemanticEntity,
    SemanticEvent,
    SemanticTaskGraph,
)
from robot_intent_agent.grounding.spatial_reasoner import SpatialReasoner
from robot_intent_agent.semantic_reasoner.semantic_fusion import SemanticFusion
from robot_intent_agent.scene_builder import RawObjectPercept, SemanticSceneBuilder


def test_industrial_transfer_uses_destination_not_recipient():
    candidate = RuleSemanticParser().parse("把工件上料到检测区")
    graph = candidate.graph
    assert graph.events[0].action == "TRANSFER"
    assert graph.events[0].theme_ref
    assert graph.events[0].destination_ref
    assert graph.events[0].recipient_ref is None
    assert "destination" in get_action_schema("TRANSFER").required_roles
    assert "recipient" in get_action_schema("TRANSFER").forbidden_roles


def test_wait_and_grasp_preserve_two_events_and_order():
    graph = RuleSemanticParser().parse("等传送带停止再抓工件").graph
    assert [event.action for event in graph.events] == ["WAIT", "GRASP"]
    assert graph.relations[0].type == "BEFORE"


def test_negation_is_independent_prohibition():
    graph = RuleSemanticParser().parse("别拿红色的，拿蓝色杯子").graph
    assert graph.prohibitions
    assert graph.prohibitions[0].type == "FORBID_ACTION"
    assert graph.prohibitions[0].target_ref


def test_numeric_compiler_intersects_domains_without_averaging():
    result = compile_feasible_domain("force_n", [
        {"operator": "min", "value": 1, "source": "robot"},
        {"operator": "max", "value": 10, "source": "robot"},
        {"operator": "max", "value": 2, "source": "object"},
    ])
    assert (result.min_value, result.max_value) == (1.0, 2.0)
    assert result.selected_value == 1.0


def test_specialized_action_wins_over_embedded_generic_verb():
    graph = RuleSemanticParser().parse("握住绿色杯子向托盘倾倒").graph
    assert [event.action for event in graph.events] == ["POUR"]
    assert graph.events[0].theme_ref and graph.events[0].destination_ref


def test_surface_delivery_is_place_not_transfer():
    graph = RuleSemanticParser().parse("请把红色杯子送到托盘的内部").graph
    assert [event.action for event in graph.events] == ["PLACE"]
    assert graph.events[0].theme_ref and graph.events[0].destination_ref


def test_tool_agent_phrase_binds_the_direct_object_as_theme():
    graph = RuleSemanticParser().parse("让夹具把红色杯子牢牢控住").graph
    assert graph.events[0].action == "GRASP"
    theme = graph.entity(graph.events[0].theme_ref)
    assert theme is not None
    assert theme.category == "cup"
    assert theme.attributes.get("color") == "red"


def test_stack_pair_and_direction_first_pour_bind_distinct_roles():
    stack = RuleSemanticParser().parse("把白色药瓶和托盘叠合起来").graph
    assert stack.events[0].action == "STACK"
    assert stack.entity(stack.events[0].theme_ref).category == "medicine_bottle"
    assert stack.entity(stack.events[0].destination_ref).category == "tray"

    pour = RuleSemanticParser().parse("向托盘倾空黄色书本").graph
    assert pour.events[0].action == "POUR"
    assert pour.entity(pour.events[0].theme_ref).category == "book"
    assert pour.entity(pour.events[0].destination_ref).category == "tray"


def test_fetch_prefers_unique_declared_receive_zone():
    scene = SemanticSceneBuilder().build([
        RawObjectPercept(
            name="fixture", object_id="fixture-01", x=0.40, y=0.30, z=0.05,
            width=0.20, height=0.10, depth=0.20,
            extra_attrs={"_upstream_affordances": ["fixed", "container"]},
        ),
        RawObjectPercept(
            name="tray", object_id="receive-01", x=0.60, y=-0.20, z=0.05,
            width=0.20, height=0.05, depth=0.20,
            extra_attrs={"_upstream_affordances": ["fixed", "container", "robot_receive_zone"]},
        ),
        RawObjectPercept(
            name="blue bottle", object_id="bottle-01", x=0.20, y=0.10, z=0.05,
            width=0.05, height=0.10, depth=0.05, color="blue",
        ),
    ])
    result = SemanticCompiler().compile("把蓝色瓶子带到机器人身边", scene=scene)
    event = result.graph.events[0]
    assert event.action == "FETCH"
    receive_id = next(
        item.id for item in scene.objects
        if item.attributes.get("_perception_object_id") == "receive-01"
    )
    assert result.graph.entity(event.destination_ref).entity_id == receive_id


def test_vague_same_class_peer_reference_does_not_create_fake_prohibition():
    graph = RuleSemanticParser().parse(
        "把绿色盒子递交给人手，不要把旁边的同类物体混进来"
    ).graph
    assert graph.events[0].action == "HANDOVER"
    assert graph.events[0].obstacle_refs == []
    assert graph.prohibitions == []


def test_unsupported_sensing_verb_is_not_downgraded_to_grasp():
    assert parse_action_candidates("请读取温度红色盒子") == []
    assert RuleSemanticParser().parse("请读取温度红色盒子").graph.events == []


def test_llm_unknown_role_reference_falls_back_to_rule_graph():
    class FakeLLM:
        is_available = True

        def semantic_candidates(self, instruction, scene=None, memory_context=None):
            graph = SemanticTaskGraph(
                instruction=instruction,
                events=[SemanticEvent(
                    event_id="llm-event-1", action="STACK",
                    theme_ref="scene-object-1", destination_ref="scene-object-2",
                    evidence_span="叠",
                )],
            )
            return [SemanticCandidate.from_graph(graph, confidence=0.95, source="llm")]

    result = SemanticCompiler(FakeLLM()).compile(
        "把蓝色书本叠放到托盘上", mode="hybrid"
    )
    assert result.engine_trace["fallback_used"] is True
    assert result.engine_trace["actual_engine"] == "RuleEngine"
    assert result.graph.events[0].theme_ref in {"entity-1", "entity-2"}
    assert not result.graph.validate_local_references()


def test_llm_custom_action_with_supported_evidence_is_repaired():
    from robot_intent_agent.planner.llm_planner import parse_semantic_candidates

    raw = {"candidates": [{
        "entities": [
            {"local_ref": "e1", "mention": "绿色药瓶", "category": "medicine_bottle",
             "evidence_spans": ["绿色药瓶"]},
            {"local_ref": "e2", "mention": "托盘", "category": "tray",
             "evidence_spans": ["托盘"]},
        ],
        "events": [{"event_id": "a1", "action": "CUSTOM", "theme_ref": "e1",
                     "destination_ref": "e2", "evidence_span": "堆"}],
    }]}
    candidates = parse_semantic_candidates(raw, "将绿色药瓶堆到托盘顶部")
    assert candidates
    assert candidates[0].graph.events[0].action == "STACK"


def test_llm_event_aliases_are_normalized_inside_prohibition_scope():
    from robot_intent_agent.planner.llm_planner import parse_semantic_candidates

    raw = {"candidates": [{
        "entities": [
            {"local_ref": "e1", "mention": "红色杯子", "category": "cup",
             "evidence_spans": ["红色杯子"]},
        ],
        "events": [{"event_id": "a1", "action": "GRASP", "theme_ref": "e1",
                     "evidence_span": "拿起"}],
        "prohibitions": [{"prohibition_id": "p1", "type": "NO_CONTACT",
                           "target_ref": "e1", "scope_event_ids": ["evt-grasp-1"],
                           "evidence_span": "不要碰"}],
    }]}
    candidates = parse_semantic_candidates(raw, "拿起红色杯子，不要碰红色杯子")
    assert candidates
    assert candidates[0].graph.prohibitions[0].scope_event_ids == ["a1"]
    assert not candidates[0].graph.validate_local_references()


def test_llm_candidate_cache_avoids_duplicate_provider_call(monkeypatch):
    from robot_intent_agent.planner.llm_planner import LLMPlanner

    raw = {"candidates": [{
        "entities": [{"local_ref": "e1", "mention": "红色杯子", "category": "cup",
                       "evidence_spans": ["红色杯子"]}],
        "events": [{"event_id": "a1", "action": "GRASP", "theme_ref": "e1",
                     "evidence_span": "拿起"}],
    }]}
    calls = []
    planner = LLMPlanner(api_key="test-key")
    monkeypatch.setattr(planner, "_call_api", lambda message: calls.append(message) or raw)
    first = planner.semantic_candidates("拿起红色杯子")
    second = planner.semantic_candidates("拿起红色杯子")
    assert first and second
    assert len(calls) == 1
    assert planner.last_call_metadata["cache_hit"] is True


def test_provider_execution_identity_is_rejected_before_fusion():
    class FakeLLM:
        is_available = True

        def semantic_candidates(self, instruction, scene=None, memory_context=None):
            graph = SemanticTaskGraph(
                instruction=instruction,
                entities=[],
                events=[SemanticEvent(
                    event_id="e1", action="GRASP", evidence_span="抓起",
                    parameters={"execution_allowed": True},
                )],
            )
            graph.entities.append(__import__(
                "robot_intent_agent.schemas.semantic_task_graph",
                fromlist=["SemanticEntity"],
            ).SemanticEntity(
                local_ref="e1-target", mention="杯子", category="cup", entity_id="obj-forged",
            ))
            graph.events[0].theme_ref = "e1-target"
            return [SemanticCandidate.from_graph(graph, confidence=0.99, source="llm")]

    result = SemanticCompiler(FakeLLM()).compile("抓起杯子", mode="hybrid")
    assert result.engine_trace["fallback_used"] is True
    assert "FORBIDDEN_PROVIDER_FIELD" in result.engine_trace["fallback_reason"]
    assert result.engine_trace["llm_candidate_accepted"] is False


def test_llm_specialized_action_changes_rule_custom_result_without_whole_result_gate():
    class FakeLLM:
        is_available = True

        def semantic_candidates(self, instruction, scene=None, memory_context=None):
            graph = SemanticTaskGraph(
                instruction=instruction,
                entities=[],
                events=[SemanticEvent(
                    event_id="e1", action="FETCH", theme_ref="target",
                    recipient_ref="user", evidence_span="送到我这里",
                )],
            )
            from robot_intent_agent.schemas.semantic_task_graph import SemanticEntity
            graph.entities.extend([
                SemanticEntity(local_ref="target", mention="容器", category="container",
                               evidence_spans=["容器"]),
                SemanticEntity(local_ref="user", mention="我这里", category="human",
                               evidence_spans=["我这里"]),
            ])
            return [SemanticCandidate.from_graph(graph, confidence=0.95, source="llm")]

    result = SemanticCompiler(FakeLLM()).compile("把容器送到我这里", mode="hybrid")
    assert result.graph.events[0].action == "FETCH"
    assert result.engine_trace["llm_candidate_accepted"] is True
    assert result.engine_trace["llm_effective"] is True
    assert result.fused_candidate is not None


def test_fusion_rewrites_provider_entity_refs_and_preserves_rule_sequence():
    rule = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="抓住蓝色瓶子",
        entities=[SemanticEntity(local_ref="entity-1", mention="位于后方的蓝色瓶子",
                                 category="bottle", attributes={"color": "blue"})],
        events=[SemanticEvent(event_id="event-1", sequence_index=0, action="GRASP",
                              theme_ref="entity-1", evidence_span="抓住")],
    ), source="rule")
    llm = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="抓住蓝色瓶子",
        entities=[SemanticEntity(local_ref="e1", mention="蓝色瓶子", category="bottle",
                                 candidate_key="scene-object-1")],
        events=[SemanticEvent(event_id="llm-event-1", sequence_index=1, action="GRASP",
                              theme_ref="e1", evidence_span="稳稳托起")],
    ), source="llm")

    fused, _ = SemanticFusion().fuse(rule, llm, "请稳稳托起蓝色瓶子")

    assert fused.graph.events[0].theme_ref == "entity-1"
    assert fused.graph.events[0].sequence_index == 0
    assert not fused.graph.validate_local_references()


def test_supported_rule_action_is_not_overwritten_by_llm_reclassification():
    """LLM repair may enrich a supported action, but not reclassify it."""
    rule = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="把瓶子推开",
        events=[SemanticEvent(event_id="event-1", action="PUSH",
                              evidence_span="推开")],
    ), source="rule")
    llm = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="把瓶子推开",
        events=[SemanticEvent(event_id="event-1", action="GRASP",
                              evidence_span="把瓶子")],
    ), source="llm")

    fused, _ = SemanticFusion().fuse(rule, llm, "把瓶子推开")

    assert fused.graph.events[0].action == "PUSH"


def test_llm_does_not_promote_plain_spatial_phrase_to_obstacle():
    rule = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="把蓝色瓶子推开",
        events=[SemanticEvent(event_id="event-1", action="PUSH",
                              theme_ref="target", evidence_span="推开")],
        entities=[SemanticEntity(local_ref="target", mention="蓝色瓶子",
                                  category="bottle")],
    ), source="rule")
    llm = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="把蓝色瓶子推开",
        entities=[SemanticEntity(local_ref="target", mention="蓝色瓶子",
                                 category="bottle"),
                  SemanticEntity(local_ref="front", mention="操作区前方",
                                 category="region")],
        events=[SemanticEvent(event_id="event-1", action="PUSH",
                              theme_ref="target", obstacle_refs=["front"],
                              evidence_span="把蓝色瓶子推开")],
    ), source="llm")

    fused, _ = SemanticFusion().fuse(rule, llm, "把蓝色瓶子推开")

    assert fused.graph.events[0].obstacle_refs == []


def test_llm_explicit_no_contact_can_add_obstacle_reference():
    rule = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="把蓝色瓶子推开，不要碰橙色夹具",
        events=[SemanticEvent(event_id="event-1", action="PUSH",
                              theme_ref="target", evidence_span="推开")],
        entities=[SemanticEntity(local_ref="target", mention="蓝色瓶子",
                                  category="bottle")],
    ), source="rule")
    llm = SemanticCandidate.from_graph(SemanticTaskGraph(
        instruction="把蓝色瓶子推开，不要碰橙色夹具",
        entities=[SemanticEntity(local_ref="target", mention="蓝色瓶子",
                                 category="bottle"),
                  SemanticEntity(local_ref="obstacle", mention="橙色夹具",
                                 category="fixture")],
        events=[SemanticEvent(event_id="event-1", action="PUSH",
                              theme_ref="target", obstacle_refs=["obstacle"],
                              evidence_span="把蓝色瓶子推开")],
    ), source="llm")

    fused, _ = SemanticFusion().fuse(
        rule, llm, "把蓝色瓶子推开，不要碰橙色夹具"
    )

    assert fused.graph.events[0].obstacle_refs == ["obstacle"]


def test_unique_rule_grounding_protects_role_from_provider_replacement():
    scene = SemanticSceneBuilder().build([
        RawObjectPercept(name="red cup", object_id="cup-01", x=0.10, y=0.10, z=0.05,
                         width=0.07, height=0.10, depth=0.07, color="red"),
        RawObjectPercept(name="blue cup", object_id="cup-02", x=0.40, y=0.10, z=0.05,
                         width=0.07, height=0.10, depth=0.07, color="blue"),
    ])
    rule = SemanticTaskGraph(
        instruction="抓住红色杯子",
        entities=[SemanticEntity(local_ref="rule-target", mention="红色杯子",
                                 category="cup", attributes={"color": "red"})],
        events=[SemanticEvent(event_id="event-1", action="GRASP",
                              theme_ref="rule-target", evidence_span="抓住")],
    )
    fused = rule.model_copy(deep=True)
    fused.entities.append(SemanticEntity(local_ref="llm-target", mention="蓝色杯子",
                                         category="cup", attributes={"color": "blue"}))
    fused.events[0].theme_ref = "llm-target"

    compiler = SemanticCompiler()
    protected = compiler._protect_rule_grounded_roles(rule, fused, scene)

    assert protected
    assert fused.events[0].theme_ref == "rule-target"
