from app.agentic_rag_common import QuestionAspect
from app.workflow_aspects import (
    apply_sufficiency_normalization,
    material_missing_labels,
    merge_dependent_aspects,
    parse_aspect_items,
    reconcile_status,
)


def test_parse_aspect_items_handles_kind_and_depends_on():
    raw = [
        {"aspect": "abstract本文", "kind": "material", "satisfied": True, "evidence_ids": ["723"]},
        {"aspect": "日本語提示", "kind": "operation", "depends_on": ["abstract本文"], "satisfied": True},
    ]
    aspects = parse_aspect_items(raw)
    assert len(aspects) == 2
    assert aspects[0].kind == "material"
    assert aspects[1].depends_on == ["abstract本文"]


def test_merge_dependent_aspects_adds_material_labels():
    aspects = merge_dependent_aspects([], ["属性X"])
    assert len(aspects) == 1
    assert aspects[0].kind == "material"
    assert aspects[0].aspect == "属性X"


def test_material_missing_labels_excludes_operation():
    aspects = [
        QuestionAspect(aspect="abstract本文", kind="material", satisfied=True),
        QuestionAspect(aspect="和訳", kind="operation", satisfied=False),
        QuestionAspect(aspect="定義", kind="material", satisfied=False),
    ]
    assert material_missing_labels(aspects) == ["定義"]


def test_apply_sufficiency_normalization_strips_operation_from_missing():
    parsed = {
        "status": "insufficient",
        "reason": "和訳なし",
        "missing_aspects": ["和訳"],
        "aspects": [
            {"aspect": "論文ID", "kind": "material", "satisfied": True, "evidence_ids": ["192"]},
            {"aspect": "abstract本文", "kind": "material", "satisfied": True, "evidence_ids": ["723"]},
            {"aspect": "和訳", "kind": "operation", "depends_on": ["abstract本文"], "satisfied": True},
        ],
        "supporting_evidence_ids": ["723"],
    }
    aspects, status, missing = apply_sufficiency_normalization(parsed, {"192", "723"})
    assert missing == []
    assert status == "sufficient"


def test_reconcile_promotes_when_all_material_satisfied():
    aspects = [
        QuestionAspect(aspect="a", kind="material", satisfied=True),
        QuestionAspect(aspect="b", kind="material", satisfied=True),
        QuestionAspect(aspect="op", kind="operation", satisfied=False),
    ]
    assert reconcile_status(aspects, "insufficient") == "sufficient"


def test_reconcile_does_not_promote_with_single_aspect():
    aspects = [QuestionAspect(aspect="a", kind="material", satisfied=True)]
    assert reconcile_status(aspects, "insufficient") == "insufficient"
