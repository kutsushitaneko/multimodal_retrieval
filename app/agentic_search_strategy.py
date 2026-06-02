"""ReAct 初回検索向けの軽量ルール分類（LLM 呼び出しなし）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.entity_patterns import extract_entities_from_text
from app.search_query_generator import SearchQueryGenerator

SearchStrategyKind = Literal["identifier", "none"]

_IDENTIFIER_HINT = (
    "【参考: 識別子が検出されました】検索の詳細は plan_and_execute_search が担当します。"
    "識別子は caption_fulltext_search（識別子トークンのみ）、"
    "それ以外の語は caption_vector_search / image_vector_text_search に分ける方針です。"
)

_FULLTEXT_NATURAL_LANGUAGE_WARNING = (
    "このクエリーは自然文に近く、全文（中カッコ OR）向きの識別子が検出されませんでした。"
    "次は caption_vector_search と image_vector_text_search を検討してください。"
)

_FULLTEXT_MIXED_QUERY_WARNING = (
    "識別子と識別子以外の語が混在しています。"
    "fulltext には識別子のみ、それ以外は caption_vector_search で検索してください"
    "（vector には識別子を含めても構いません）。"
)

_query_generator: SearchQueryGenerator | None = None


def _get_query_generator() -> SearchQueryGenerator:
    global _query_generator
    if _query_generator is None:
        _query_generator = SearchQueryGenerator()
    return _query_generator


@dataclass(frozen=True)
class SearchStrategyHint:
    strategy: SearchStrategyKind
    hint_text: str


def _has_identifier_entities(question: str) -> bool:
    return bool(extract_entities_from_text(question))


def classify_question_strategy(question: str) -> SearchStrategyHint:
    """識別子 lookup / その他（ヒントなし）をルールで判定する。推移性は Planner LLM に委譲。"""
    text = (question or "").strip()
    if not text:
        return SearchStrategyHint(strategy="none", hint_text="")

    if _has_identifier_entities(text):
        return SearchStrategyHint(strategy="identifier", hint_text=_IDENTIFIER_HINT)

    return SearchStrategyHint(strategy="none", hint_text="")


def format_first_step_hint_for_prompt(hint: SearchStrategyHint) -> str:
    if hint.hint_text:
        return hint.hint_text
    return (
        "（特になし。検索は plan_and_execute_search を優先してください。"
        "分解・ツール選択・クエリーは Search Planner が担当します。）"
    )


def count_regex_detected_entities(question: str) -> int:
    return len(extract_entities_from_text(question or ""))


def query_has_fulltext_friendly_tokens(query: str) -> bool:
    """クエリーに全文（中カッコ OR）向けの短い識別子が含まれるか。"""
    return bool(_get_query_generator().extract_rule_entities(query))


def _strip_rule_entities_from_query(query: str) -> str:
    """rule ベース識別子文字列を除去した残りを返す。"""
    text = str(query or "")
    entities = _get_query_generator().extract_rule_entities(text)
    if not entities:
        return text.strip()
    remainder = text
    for entity in sorted(entities, key=lambda item: len(str(item.get("text") or "")), reverse=True):
        entity_text = str(entity.get("text") or "")
        if entity_text:
            remainder = remainder.replace(entity_text, " ")
    return re.sub(r"\s+", " ", remainder).strip()


def fulltext_query_mixes_identifier_and_natural_language(query: str) -> bool:
    """fulltext 向け識別子と識別子以外の語が同一 query に混在するか。"""
    if not query_has_fulltext_friendly_tokens(query):
        return False
    return len(_strip_rule_entities_from_query(query)) >= 2


def format_lead_tool_recommendation(lead: str) -> str:
    """Verifier / 確定保留用の lead 1 件に対する推奨 Tool 表記。"""
    text = (lead or "").strip()
    if not text:
        return ""
    if query_has_fulltext_friendly_tokens(text):
        return f'（推奨: caption_fulltext_search で「{text}」）'
    return f'（推奨: caption_vector_search で「{text}」）'


def format_leads_tool_recommendations(leads: list[str]) -> str:
    """複数 lead の推奨 Tool をまとめて返す。"""
    parts = [format_lead_tool_recommendation(lead) for lead in leads if str(lead or "").strip()]
    return " ".join(parts)


def fulltext_natural_language_warning(query: str) -> str:
    """Step 2 以降で全文検索に自然文のみを渡したときのソフトガード文言。"""
    if query_has_fulltext_friendly_tokens(query):
        return ""
    return _FULLTEXT_NATURAL_LANGUAGE_WARNING


def fulltext_mixed_query_warning(query: str) -> str:
    """Step 2 以降で fulltext に識別子と一般語を混在させたときのソフトガード文言。"""
    if fulltext_query_mixes_identifier_and_natural_language(query):
        return _FULLTEXT_MIXED_QUERY_WARNING
    return ""
