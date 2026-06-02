"""Workflow 向け Material / Operation 観点のパースと正規化。"""

from __future__ import annotations

from typing import Any

from app.agentic_rag_common import QuestionAspect, QuestionAspectKind


def _normalize_kind(raw: Any) -> QuestionAspectKind:
    kind = str(raw or "").strip().lower()
    if kind == "operation":
        return "operation"
    return "material"


def parse_aspect_items(raw_items: Any) -> list[QuestionAspect]:
    if not isinstance(raw_items, list):
        return []
    aspects: list[QuestionAspect] = []
    seen_labels: set[str] = set()
    for item in raw_items:
        if isinstance(item, str):
            label = item.strip()
            if label and label not in seen_labels:
                seen_labels.add(label)
                aspects.append(QuestionAspect(aspect=label, kind="material"))
            continue
        if not isinstance(item, dict):
            continue
        label = str(item.get("aspect") or item.get("name") or "").strip()
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        depends_on_raw = item.get("depends_on") or []
        depends_on: list[str] = []
        if isinstance(depends_on_raw, list):
            depends_on = [str(dep).strip() for dep in depends_on_raw if str(dep).strip()]
        elif isinstance(depends_on_raw, str) and depends_on_raw.strip():
            depends_on = [depends_on_raw.strip()]

        satisfied_raw = item.get("satisfied")
        satisfied = None
        if isinstance(satisfied_raw, bool):
            satisfied = satisfied_raw

        evidence_ids_raw = item.get("evidence_ids") or []
        evidence_ids: list[str] = []
        if isinstance(evidence_ids_raw, list):
            evidence_ids = [str(eid).strip() for eid in evidence_ids_raw if str(eid).strip()]

        aspects.append(
            QuestionAspect(
                aspect=label,
                kind=_normalize_kind(item.get("kind")),
                depends_on=depends_on,
                satisfied=satisfied,
                evidence_ids=evidence_ids,
            )
        )
    return aspects


def merge_dependent_aspects(
    aspects: list[QuestionAspect],
    dependent_labels: list[str],
) -> list[QuestionAspect]:
    merged = list(aspects)
    seen = {item.aspect for item in merged}
    for label in dependent_labels:
        text = str(label or "").strip()
        if text and text not in seen:
            seen.add(text)
            merged.append(QuestionAspect(aspect=text, kind="material"))
    return merged


def material_aspects(aspects: list[QuestionAspect]) -> list[QuestionAspect]:
    return [item for item in aspects if item.kind == "material"]


def operation_aspects(aspects: list[QuestionAspect]) -> list[QuestionAspect]:
    return [item for item in aspects if item.kind == "operation"]


def material_missing_labels(aspects: list[QuestionAspect]) -> list[str]:
    missing = []
    for item in material_aspects(aspects):
        if item.satisfied is False:
            missing.append(item.aspect)
    return missing[:5]


def operation_aspect_labels(aspects: list[QuestionAspect]) -> list[str]:
    return [item.aspect for item in operation_aspects(aspects)]


def reconcile_status(
    aspects: list[QuestionAspect],
    llm_status: str,
    *,
    min_aspects_for_reconcile: int = 2,
) -> str:
    """全 material が satisfied なら insufficient を sufficient に昇格。"""
    materials = material_aspects(aspects)
    if len(aspects) < min_aspects_for_reconcile or not materials:
        return llm_status
    if all(item.satisfied is True for item in materials):
        if llm_status == "insufficient":
            return "sufficient"
    return llm_status


def apply_sufficiency_normalization(
    parsed: dict[str, Any],
    evidence_ids: set[str],
) -> tuple[list[QuestionAspect], str, list[str]]:
    aspects = parse_aspect_items(parsed.get("aspects"))
    llm_status = str(parsed.get("status") or "").strip().lower()
    llm_missing = parsed.get("missing_aspects") or []
    if not isinstance(llm_missing, list):
        llm_missing = []

    if aspects:
        missing = material_missing_labels(aspects)
    else:
        missing = [
            str(item).strip()
            for item in llm_missing
            if str(item).strip()
        ][:5]

    status = reconcile_status(aspects, llm_status) if aspects else llm_status

    supporting_ids = parsed.get("supporting_evidence_ids") or []
    if status == "sufficient" and isinstance(supporting_ids, list) and supporting_ids:
        for item in aspects:
            if item.kind == "material" and item.satisfied is True and not item.evidence_ids:
                item.evidence_ids = [
                    str(eid).strip()
                    for eid in supporting_ids
                    if str(eid).strip() in evidence_ids
                ][:3]

    return aspects, status, missing


def format_aspects_for_prompt(aspects: list[QuestionAspect]) -> str:
    if not aspects:
        return "（未指定）"
    lines = []
    for item in aspects:
        parts = [f"- {item.aspect} ({item.kind})"]
        if item.depends_on:
            parts.append(f"depends_on={', '.join(item.depends_on)}")
        if item.satisfied is not None:
            parts.append(f"satisfied={item.satisfied}")
        if item.evidence_ids:
            parts.append(f"evidence_ids={', '.join(item.evidence_ids)}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def format_operation_aspects_for_followup(aspects: list[QuestionAspect]) -> str:
    labels = operation_aspect_labels(aspects)
    if not labels:
        return "（なし）"
    return "、".join(labels)

def aspect_counts_summary(aspects: list[QuestionAspect]) -> str:
    material_count = len(material_aspects(aspects))
    operation_count = len(operation_aspects(aspects))
    return f"{material_count} material, {operation_count} operation"
