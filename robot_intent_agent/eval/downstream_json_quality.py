"""Deterministic gate for deciding whether an intent result is safe to hand downstream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DownstreamQualityResult:
    usable: bool
    score: float
    reasons: List[str] = field(default_factory=list)


def assess_downstream_json(result: Dict[str, Any], perception: Dict[str, Any]) -> DownstreamQualityResult:
    """Score semantic usability, not merely JSON syntax.

    READY outputs must bind all executable entity references to perception IDs.
    BLOCKED/NEEDS_CLARIFICATION outputs are usable only when they explain why.
    """
    reasons: List[str] = []
    known_ids = {
        obj.get("object_id") or obj.get("id")
        for obj in perception.get("objects", [])
        if isinstance(obj, dict)
    }
    known_ids.discard(None)

    status = result.get("plan_status") or result.get("status")
    if status in {"BLOCKED", "NEEDS_CLARIFICATION"}:
        blockers = result.get("blocking_reasons") or result.get("reasons") or []
        if not blockers:
            reasons.append("NON_EXECUTABLE_WITHOUT_REASON")
        return DownstreamQualityResult(not reasons, 1.0 if not reasons else 0.5, reasons)

    if status not in {"READY", "READY_WITH_SAFE_SUBSTITUTION"}:
        reasons.append("UNKNOWN_PLAN_STATUS")

    target_ids = result.get("target_ids") or []
    if result.get("target_object_id"):
        target_ids = [result["target_object_id"], *target_ids]
    if not target_ids:
        reasons.append("MISSING_TARGET_ID")
    for entity_id in target_ids:
        if entity_id not in known_ids:
            reasons.append(f"FABRICATED_TARGET_ID:{entity_id}")

    destination_id = result.get("destination_id")
    if destination_id and destination_id not in known_ids:
        reasons.append(f"FABRICATED_DESTINATION_ID:{destination_id}")

    score = max(0.0, 1.0 - 0.25 * len(reasons))
    return DownstreamQualityResult(not reasons, score, reasons)
