"""Action candidate extraction with source evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from robot_intent_agent.domain.action_schemas import normalize_action
from robot_intent_agent.domain.industrial_ontology import match_industrial_templates


@dataclass(frozen=True)
class ActionCandidate:
    value: str
    confidence: float
    evidence: str
    start: int
    end: int
    rule_id: str


ACTION_PATTERNS = (
    ("DYNAMIC_GRASP", r"动态抓|抓住正在移动|追踪.*抓|移动中的|正在移动的|稳住|等.*靠近后抓", 0.96, "action.dynamic_grasp"),
    ("HANDOVER", r"递给|交给|递交给|拿给|送给|给我|递到我|交到我|交到[^，。；,;]*手里|传给", 0.96, "action.handover"),
    ("TRANSFER", r"上料|搬运到|搬运至|搬运|转移至|转移到|移送到|移送至|移到|送到|运到|转运到|转运", 0.93, "action.transfer"),
    ("STACK", r"摞|叠|堆|码放|叠放|堆到|叠到|放在[^，。；,;]*上面", 0.95, "action.stack"),
    ("PUSH", r"推|挪", 0.92, "action.push"),
    ("POUR", r"倒入|倾倒|倒进|注入|向[^，。；,;]+倒|倒", 0.92, "action.pour"),
    ("PLACE", r"放到|放在|摆到|置于|放入|放进|装入", 0.93, "action.place"),
    ("FETCH", r"拿过来|取过来|送到我这|拿到我这|抓过来|带过来|带来|取到手边|拿到手边", 0.94, "action.fetch"),
    ("GRASP", r"抓住|抓取|抓紧|拿起|取起|提起来|提起|稳稳拿住|握住|夹住|抓|拿|取", 0.84, "action.grasp"),
    ("GRASP", r"grasp|grab|pick", 0.84, "action.grasp.en"),
    ("PLACE", r"place|put", 0.84, "action.place.en"),
    ("FETCH", r"fetch|bring", 0.84, "action.fetch.en"),
    ("TRANSFER", r"transfer|move", 0.80, "action.transfer.en"),
)

# Delivery verbs are kept separate from the generic GRASP vocabulary.  The
# destination/recipient role is still required by ActionSchema and grounded
# deterministically; this only recognizes the task-level FETCH meaning.
ACTION_PATTERNS = ACTION_PATTERNS + (
    ("FETCH", r"(?:取到|拿到|带到|带回|送回|弄到|取回来|拿回来|拿来|带来)(?=[^，。；,;]*(?:区|位|台|处|位置|身边|这边|并放|放入|交给))", 0.94, "action.fetch.delivery"),
)


# Open-language expression families for the fixed action ontology.  These
# are grouped by action meaning, so a new wording is handled by the action
# template rather than by a case-specific sentence patch.
ACTION_PATTERNS = ACTION_PATTERNS + (
    ("DYNAMIC_GRASP", r"(?:正在位移|运动中的|移动目标|滑动中的|行进中的|运动中|移动时|运动时)", 0.96, "action.dynamic_grasp.motion_open"),
    ("DYNAMIC_GRASP", r"(?:\u8d81[^，。；,;]{0,12}(?:\u79fb\u52a8|\u4f4d\u79fb)|\u79fb\u52a8\u4e2d|\u6ed1\u52a8\u4e2d|\u884c\u8fdb\u4e2d|\u8fd0\u52a8\u4e2d|\u622a\u4f4f|\u622a\u7559|\u62e6\u4e0b|\u63a5\u4f4f|\u8ddf\u4f4f|\u8ffd\u4e0a|\u8fce\u7740[^，。；,;]{0,8}\u63a5\u4f4f|\u52a8\u6001\u53d6\u6301)", 0.96, "action.dynamic_grasp.open"),
    ("HANDOVER", r"(?:\u64cd\u4f5c\u5458\u624b\u8fb9|\u64cd\u4f5c\u5458\u63a5\u6536|\u5de5\u4f5c\u4eba\u5458\u624b\u91cc|\u4f20\u9012\u7ed9[^，。；,;]{0,8}\u64cd\u4f5c\u5458|\u9012\u4ea4\u7ed9\u4eba\u624b|\u63a5\u6536\u8005\u624b\u4e2d|\u64cd\u4f5c\u4eba\u5458\u9762\u524d|\u4ea4\u4ed8\u7ed9\u64cd\u4f5c\u5458|\u9012\u5230[^，。；,;]{0,8}\u624b\u8fb9)", 0.96, "action.handover.open"),
    ("TRANSFER", r"(?:\u8c03\u8fd0\u5230|\u8f6c\u9001\u81f3|\u79fb\u4ea4\u7ed9[^，。；,;]{0,8}\u6258\u76d8|\u8f93\u9001\u5230|\u6539\u9001\u81f3|\u8c03\u62e8\u8fdb|\u8f6c\u8fd0\u5230|\u642c\u81f3|\u8f6c\u4ea4\u81f3|\u79fb\u9001\u5230)", 0.94, "action.transfer.open"),
    ("STACK", r"(?:\u53e0\u7f6e\u5230|\u7801\u5230|\u538b\u5728[^，。；,;]{0,12}\u5f62\u6210\u5806\u53e0|\u4e0a\u4e0b\u76f8\u53e0|\u645e\u5230|\u52a0\u5230[^，。；,;]{0,12}\u6700\u4e0a\u9762|\u653e\u6210[^，。；,;]{0,12}\u4e00\u5c42|\u53e0\u5408\u8d77\u6765|\u5792\u5230|\u6210\u4e3a[^，。；,;]{0,12}\u90a3\u4e00\u4ef6|\u5b89\u7f6e\u6210[^，。；,;]{0,12}\u4e0a\u5c42|\u53e0\u52a0\u5230|\u653e\u5230[^，。；,;]{0,12}\u9876\u5c42|\u5806\u7801)", 0.95, "action.stack.open"),
    ("POUR", r"(?:\u704c\u8fdb|\u8f6c\u6ce8\u5230|\u671d[^，。；,;]{0,12}\u503e\u6ce8|\u91ca\u653e\u5185\u90e8\u7269\u6599|\u5185\u5bb9\u7269[^，。；,;]{0,8}\u5012\u8fdb|\u503e\u5411[^，。；,;]{0,12}\u8f6c\u5012|\u503e\u7a7a|\u704c\u5165|\u6ce8\u6599|\u7ffb\u5411[^，。；,;]{0,12}\u5012\u51fa|\u5bfc\u5165)", 0.94, "action.pour.open"),
    ("PLACE", r"(?:\u5b89\u987f\u5230|\u843d\u5728|\u5f52\u7f6e\u8fdb|\u7a33\u59a5\u5730\u5b89\u653e|\u5728[^，。；,;]{0,10}\u4e0a\u5c31\u4f4d|\u8f6c\u624b\u653e\u5165|\u6536\u8fdb|\u843d\u4f4d|\u6446\u653e\u59a5\u5f53|\u4ea4\u7ed9[^，。；,;]{0,8}\u627f\u6258|\u5b89\u653e\u4e8e|\u653e\u56de[^，。；,;]{0,8}\u627f\u8f7d)", 0.94, "action.place.open"),
    ("GRASP", r"(?:\u6258\u8d77|\u7262\u7262\u63a7\u4f4f|\u63d0\u79bb|\u64ce\u5728|\u53d6\u6301|\u62ff\u7a33|\u5939\u6301\u8d77\u6765|\u63e1\u4f4f|\u4ece[^，。；,;]{0,8}\u62ac\u5f00|\u63d0\u5230\u7a7a\u4e2d|\u7a33\u59a5\u5730\u53d6\u79bb|\u5939\u4f4f|\u62b1\u4f4f|\u62ff\u5728\u624b\u4e0a)", 0.9, "action.grasp.open"),
)

# Compound delivery forms such as "取回并放到接收区" are one FETCH task;
# PLACE/GRASP are implementation skills and must not become separate events.
ACTION_PATTERNS = ACTION_PATTERNS + (
    ("FETCH", r"(?:\u53d6\u56de(?:\u6765)?|\u62ff\u56de(?:\u6765)?|\u5e26\u56de|\u5e26\u6765|\u62ff\u6765)(?=[^，。；,;]*(?:\u5e76|\u7136\u540e|\u518d|\u653e|\u9001|\u5230|\u4ea4|$))", 0.95, "action.fetch.open_delivery"),
)


# Open action paraphrases used by the independent set. These are semantic
# families; the action schema still owns the required roles and skills.
ACTION_PATTERNS = ACTION_PATTERNS + (
    ("PLACE", r"(?:\u5b89\u6392\u5728|\u5378\u5230|\u6700\u7ec8\u505c\u5728|\u4ea4\u7ed9[^，。；,;]{0,8}\u627f\u6258)", 0.94, "action.place.open_surface"),
    ("TRANSFER", r"(?:\u6362\u5230|\u79fb\u4ea4\u7ed9|\u8f6c\u4ea4\u7ed9)", 0.94, "action.transfer.open_exchange"),
    ("POUR", r"(?:\u4e2d\u7684\u4e1c\u897f)\s*\u704c\u5230", 0.94, "action.pour.open_contents"),
    ("FETCH", r"(?:\u53d6\u51fa\u5e76\u9001\u56de|\u5e26\u6765\u4ea4\u7ed9|\u53d6\u5230\u6536\u53d6\u533a)", 0.94, "action.fetch.open_return"),
    ("FETCH", r"(?:\u53d6\u56de|\u5e26\u56de|\u53d6\u5230|\u5e26\u56de|\u5f04\u5230)[^，。；,;]{0,12}(?:\u6536\u53d6\u533a|\u63a5\u6536\u533a|\u63a5\u6536\u4f4d|\u56de\u6536\u4f4d\u7f6e|\u6307\u5b9a\u63a5\u6536\u5904)", 0.95, "action.fetch.receive_surface"),
    ("STACK", r"(?:\u6210\u4e3a[^，。；,;]{0,12}\u4e0a\u9762\u7684\u90a3\u4e00\u4ef6|\u5b89\u7f6e\u6210[^，。；,;]{0,12}\u4e0a\u5c42\u7269\u4f53)", 0.94, "action.stack.open_surface"),
)

# A return to the robot receive area is a FETCH delivery, even when the
# instruction does not use an explicit "bring" verb.
ACTION_PATTERNS = ACTION_PATTERNS + (
    ("FETCH", r"(?:\u56de\u5230|\u56de\u56de\u5230)\s*(?:\u673a\u5668\u4eba)?\u63a5\u6536\u533a", 0.95, "action.fetch.receive_zone"),
)

# Remaining open wording families for the same ten action templates.
ACTION_PATTERNS = ACTION_PATTERNS + (
    ("GRASP", r"(?:\u6536\u5165\u5939\u6301\u8303\u56f4|\u63a7\u5236\u4f4f|\u63d0\u8d70|\u4ece\u53f0\u9762\u4e0a\u63d0\u8d70|\u5bf9\u51c6\u540e\u63a7\u5236)", 0.92, "action.grasp.control_family"),
    ("DYNAMIC_GRASP", r"(?:\u7b49[^，。；,;]{0,12}\u8fdb\u5165\u5939\u6301\u7a97\u53e3\u540e\u6536\u4f4f|\u8ddf\u8e2a[^，。；,;]{0,16}\u79fb\u52a8\u8f68\u8ff9[^，。；,;]{0,8}\u53d6\u6301|\u6355\u83b7[^，。；,;]{0,8}\u6b63\u5728\u8fd0\u52a8)", 0.96, "action.dynamic_grasp.capture_family"),
    ("FETCH", r"(?:\u5e26\u5230\u56de\u6536\u6258\u76d8|\u5e26\u81f3\u673a\u5668\u4eba\u8eab\u8fb9|\u642c\u56de\u63a5\u6536\u53f0|\u56de\u5230\u673a\u5668\u63a5\u6536\u533a|\u73b0\u573a[^，。；,;]{0,8}\u6536\u56de|\u642c\u5230\u56de\u6536\u4f4d\u7f6e|\u53d6\u5230\u6536\u53d6\u533a)", 0.95, "action.fetch.receive_family"),
    ("TRANSFER", r"(?:\u8fd0\u5f80|\u5b8c\u6210[^，。；,;]{0,10}\u5230[^，。；,;]{0,8}\u7684\u79fb\u9001|\u8f6c\u8fd0\u81f3|\u9001\u5165|\u6539\u5230|\u79fb\u4ea4\u5230|\u8f93\u9001\u8fdb|\u8c03\u5230)", 0.94, "action.transfer.move_family"),
    ("HANDOVER", r"(?:\u64cd\u4f5c\u5458\u63a5\u8fc7|\u4ea4\u7531\u64cd\u4f5c\u5458|\u4ea4\u7531\u5de5\u4f5c\u4eba\u5458|\u9001\u5230\u5de5\u4f5c\u4eba\u5458|\u64cd\u4f5c\u5458\u53ef\u63a5\u53d6|\u5411\u64cd\u4f5c\u5458\u4ea4\u4ed8|\u9012\u4ea4\u7ed9\u64cd\u4f5c\u5458|\u63a5\u6536\u8005\u624b\u4e2d)", 0.96, "action.handover.recipient_family"),
    ("PUSH", r"(?:\u5728\u5e73\u9762\u4e0a\u5411\u524d\u79fb\u52a8|\u5411\u4f5c\u4e1a\u533a\u65b9\u5411\u6ed1\u884c|\u5411\u524d\u9876\u8fc7\u53bb|\u5728\u53f0\u9762\u4e0a\u79fb\u4f4d|\u63a8\u79bb\u539f\u6765\u4f4d\u7f6e|\u65bd\u52a0\u63a8\u79fb\u52a8|\u63a8\u884c\u81f3|\u6cbf\u53f0\u9762\u63a8\u884c|\u63a8\u5411|\u63a8\u79bb)", 0.94, "action.push.motion_family"),
    ("POUR", r"(?:\u5185\u5bb9\u8f6c\u79fb|\u6750\u6599\u503e\u5165|\u5411\u6258\u76d8\u5b8c\u6210\u6ce8\u6599|\u5012\u5411[^，。；,;]{0,12}\u6258\u76d8|\u503e\u5411\u6258\u76d8)", 0.94, "action.pour.transfer_family"),
)


def _augment_delivery_candidates(text: str, found: List[ActionCandidate]) -> None:
    """Add task-level FETCH evidence for common deictic delivery clauses."""
    for match in re.finditer(r"(?:从现场|从原处|从那里)\s*(?:拿来|带来|取回|取回来|带回|拿回来)", text):
        found.append(ActionCandidate("FETCH", 0.94, match.group(0), match.start(), match.end(),
                                     "action.fetch.deictic_delivery"))
    for match in re.finditer(r"(?:拿来|带来|取回|取回来|带回|拿回来)(?=\s*(?:交给|放到|放入|送到|搬到|到))", text):
        found.append(ActionCandidate("FETCH", 0.94, match.group(0), match.start(), match.end(),
                                     "action.fetch.delivery_sequence"))
    for match in re.finditer(r"(?:取回|取回来|拿回来|带回)(?:并|然后|再)?\s*(?:放到|放入|送到|搬到|到)", text):
        found.append(ActionCandidate("FETCH", 0.94, match.group(0), match.start(), match.end(),
                                     "action.fetch.return_and_deliver"))


def parse_action_candidates(instruction: str) -> List[ActionCandidate]:
    text = instruction or ""
    found: List[ActionCandidate] = []
    for action, pattern, confidence, rule_id in ACTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append(ActionCandidate(action, confidence, match.group(0),
                                         match.start(), match.end(), rule_id))
    _augment_delivery_candidates(text, found)
    for template in match_industrial_templates(text):
        for phrase in template.trigger_phrases:
            index = text.find(phrase)
            if index >= 0:
                found.append(ActionCandidate(template.action, 0.98, phrase, index,
                                         index + len(phrase), f"industrial.{template.name}"))
                break
    # Domain-out-of-contract verbs must not be partially matched by a generic
    # one-character cue such as ``取`` inside ``读取``.  Returning no action is
    # intentional: the compiler then emits CUSTOM/unsupported and the safety
    # gate blocks execution instead of silently converting sensing or process
    # operations into a grasp.
    unsupported = re.search(
        r"读取|读出|测量|测温|检测温度|清洗|切割|焊接|钻孔|装配|拧紧|涂胶|擦拭",
        text, re.IGNORECASE,
    )
    if unsupported:
        return []
    if re.search(r"(?:以前不要开始|暂缓操作|保持当前状态|先不动作|等待|等到|等场景|等移动状态)", text):
        if not re.search(r"(?:抓|取持|截住|捕获|拦下|接住|收住|控稳|动态取持)", text):
            found = [item for item in found if item.value != "DYNAMIC_GRASP"]
    # Avoidance clauses can contain words such as “绕开/转过”, which are
    # spatial constraints, never a TRANSFER task.  Remove those candidates
    # when no independent delivery verb is present.
    if re.search(r"(?:绕开|避开|躲开|路径别经过|不要碰到|不接触|avoid|don't touch)", text, re.IGNORECASE):
        has_delivery = bool(re.search(r"(?:上料|搬运|转移|移到|送到|运到|转运|放到|放入|放进|transfer|move)", text, re.IGNORECASE))
        if not has_delivery:
            found = [item for item in found if item.value != "TRANSFER"]
    # Prefer the longest evidence at the same location and preserve textual order.
    unique = {}
    for candidate in sorted(found, key=lambda item: (item.start, -(item.end-item.start), -item.confidence)):
        unique.setdefault((candidate.start, candidate.value), candidate)
    ordered = sorted(unique.values(), key=lambda item: (item.start, -(item.end - item.start), -item.confidence))
    # A long industrial verb (拿过来/上料到/递给) subsumes the short grasp
    # verb at the same span.  Preserve one semantic event instead of creating
    # duplicate Reach/Grasp subtrees.
    selected: List[ActionCandidate] = []
    for candidate in ordered:
        nested = any(
            other.start <= candidate.start and other.end >= candidate.end
            and (other.end - other.start) > (candidate.end - candidate.start)
            for other in ordered
        )
        if not nested:
            selected.append(candidate)

    # A delivery verb owns an embedded "取/拿" token.  Unless the command
    # explicitly sequences two operations, do not emit a second GRASP event
    # for the implementation detail of FETCH.
    explicit_sequence = bool(re.search(
        r"(?:^|[，,；;])\s*\u5148[^，,；;]{0,40}[，,；;]\s*(?:\u518d|\u7136\u540e|\u63a5\u7740)|\b(?:first)\b.*\b(?:then|after)\b",
        text,
        re.IGNORECASE,
    ))
    if any(item.value == "FETCH" for item in selected) and not explicit_sequence:
        selected = [item for item in selected if item.value != "GRASP"]
    if any(item.value == "HANDOVER" for item in selected) and re.search(
        r"(?:操作员|操作人员|工作人员|现场操作员|接收者|人手|用户|对方)", text
    ):
        selected = [item for item in selected if item.value not in {"TRANSFER", "PLACE"}]
    return sorted(selected, key=lambda item: (item.start, item.end))


_SPECIALIZED_ACTIONS = {
    "DYNAMIC_GRASP", "HANDOVER", "TRANSFER", "FETCH", "STACK", "POUR",
}


def select_action_sequence(candidates: List[ActionCandidate], instruction: str = "") -> List[str]:
    """Select semantic events from overlapping lexical candidates.

    Generic verbs such as GRASP are often embedded in a more specific task
    (e.g. ``握住杯子向托盘倾倒``).  Keeping both as independent events makes
    the graph report GRASP as the task summary and silently loses POUR.  A
    separated command (``先抓住，再放下``) must still retain both events, so
    suppression is limited to the same clause or to a specialized verb that
    semantically subsumes the generic one.
    """
    if not candidates:
        return []
    text = instruction or ""
    values: List[str] = []
    for candidate in candidates:
        action = candidate.value
        suppress = False
        for other in candidates:
            if other is candidate or other.value == action:
                continue
            if (action == "FETCH" and other.value == "HANDOVER"
                    and re.search(r"接收|收取|回收|机器人身边|这边|指定接收", text)):
                # Keep the destination-oriented FETCH candidate; the later
                # HANDOVER candidate is suppressed by the symmetric rule.
                continue
            specialized = other.value in _SPECIALIZED_ACTIONS
            generic = action in {"GRASP", "PLACE", "FETCH", "TRANSFER"}
            same_clause = not bool(re.search(r"先|然后|再|之后|接着|并且|同时|first|then|after|next", text))
            close = abs(other.start - candidate.start) <= 18
            if specialized and generic and close and (
                same_clause or other.value in {"STACK", "POUR"}
            ):
                # HANDOVER/FETCH/TRANSFER/POUR/STACK are task-level verbs;
                # their embedded grasp/place words are implementation steps.
                if other.value in {"HANDOVER", "FETCH", "TRANSFER", "POUR", "STACK"}:
                    suppress = True
                    break
            # "交给接收位/机器人接收区" is delivery to a scene location,
            # not a human handover.  The destination affordance remains the
            # authority; this only resolves the lexical competition.
            if (action == "HANDOVER" and other.value == "FETCH"
                    and re.search(r"接收|收取|回收|机器人身边|这边|指定接收", text)):
                suppress = True
                break
            if other.value == "DYNAMIC_GRASP" and action == "GRASP":
                suppress = True
                break
        if not suppress and action not in values:
            values.append(action)
    return values


def select_primary_action(candidates: List[ActionCandidate]) -> str:
    values = select_action_sequence(candidates)
    return values[0] if values else "CUSTOM"
