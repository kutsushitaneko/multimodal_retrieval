"""ReAct 初回検索向けの軽量ルール分類（LLM 呼び出しなし）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.search_query_generator import SearchQueryGenerator

SearchStrategyKind = Literal["identifier", "transitive", "none"]

_IDENTIFIER_HINT = (
    "【初回検索ヒント: 識別子・完全一致】初回は caption_fulltext_search を優先し、"
    "識別子・固有語はクエリーにそのまま含めてください。"
    "狭い query で足りる場合は単一 Tool でも構いません。広い multi_search は不要です。"
    "2ホップ目以降、evidence や Verifier から得た長い自然文（タイトル・本文・要約）の再検索は "
    "caption_vector_search を使い、caption_fulltext_search は再検索クエリーに識別子トークンが含まれる場合のみ。"
)

_FULLTEXT_NATURAL_LANGUAGE_WARNING = (
    "このクエリーは自然文に近く、全文（中カッコ OR）向きの識別子が検出されませんでした。"
    "次は caption_vector_search を検討してください。"
)

_TRANSITIVE_HINT = (
    "【初回検索ヒント: 推移的 A→X】初回は参照対象 A の特定（第1ホップ）のみ行ってください。"
    "属性 X（座標・日付・数値・定義など）を query_variants に含めないでください。"
    "multi_search を使う場合も第1ホップ用クエリーは1〜2件に絞り、"
    "caption_vector_search と image_vector_text_search を含めてください。"
)

# 推移的質問で「求めている属性」らしき語（抽象パターン）
_ATTRIBUTE_PATTERN = re.compile(
    r"(?:"
    r"座標|緯度|経度|位置|場所|所在地|"
    r"日付|年度|年|月|日|時刻|"
    r"発掘|設立|創業|発売|"
    r"数値|値|金額|料金|価格|人口|面積|"
    r"定義|意味|原因|理由|方法|手順|"
    r"氏名|名前|名称|タイトル|"
    r"出口|番号|コード|ID"
    r")",
    re.IGNORECASE,
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
    entities = _get_query_generator().extract_rule_entities(question)
    return bool(entities)


def _looks_transitive(question: str) -> bool:
    text = (question or "").strip()
    if not text:
        return False
    # 「の」が2回以上かつ、属性らしい語を質問が求めている
    if text.count("の") < 2:
        return False
    return bool(_ATTRIBUTE_PATTERN.search(text))


def classify_question_strategy(question: str) -> SearchStrategyHint:
    """識別子 lookup / 推移的 A→X / その他（ヒントなし）をルールで判定する。"""
    text = (question or "").strip()
    if not text:
        return SearchStrategyHint(strategy="none", hint_text="")

    if _has_identifier_entities(text):
        return SearchStrategyHint(strategy="identifier", hint_text=_IDENTIFIER_HINT)

    if _looks_transitive(text):
        return SearchStrategyHint(strategy="transitive", hint_text=_TRANSITIVE_HINT)

    return SearchStrategyHint(strategy="none", hint_text="")


def format_first_step_hint_for_prompt(hint: SearchStrategyHint) -> str:
    if hint.hint_text:
        return hint.hint_text
    return "（特になし。下記「初回検索の戦略」に従い、質問タイプを判断してください。）"


def query_has_fulltext_friendly_tokens(query: str) -> bool:
    """クエリーに全文（中カッコ OR）向けの短い識別子が含まれるか。"""
    return bool(_get_query_generator().extract_rule_entities(query))


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
