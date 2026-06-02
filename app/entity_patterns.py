"""外部設定の固有表現パターン読み込み。"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from app.fulltext_entity_extractor import FulltextEntityExtractor
from app.paths import CONFIG_ENTITY_PATTERNS


@lru_cache(maxsize=1)
def load_entity_pattern_specs() -> list[dict[str, str]]:
    path = CONFIG_ENTITY_PATTERNS
    if not os.path.isfile(path):
        return _default_pattern_specs()
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _default_pattern_specs()
    patterns = payload.get("patterns") if isinstance(payload, dict) else None
    if not isinstance(patterns, list):
        return _default_pattern_specs()
    specs: list[dict[str, str]] = []
    for item in patterns:
        if not isinstance(item, dict):
            continue
        entity_type = str(item.get("type") or "identifier").strip()
        regex = str(item.get("regex") or "").strip()
        if regex:
            specs.append({"type": entity_type, "regex": regex})
    return specs or _default_pattern_specs()


def _default_pattern_specs() -> list[dict[str, str]]:
    return [
        {"type": "url", "regex": r"https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=]+"},
        {"type": "paper_id", "regex": r"(?<!\d)\d{4}\.\d{4,5}(?!\d)"},
        {"type": "error_code", "regex": r"\b(?:ORA|PLS|SP2|TNS|HTTP|OCI)-?\d{3,6}\b"},
    ]


def extract_entities_from_text(query: str) -> list[dict[str, Any]]:
    """設定ファイルの regex で固有表現候補を抽出する。"""
    text = query or ""
    entities: list[dict[str, Any]] = []
    for spec in load_entity_pattern_specs():
        pattern = spec["regex"]
        entity_type = spec["type"]
        try:
            for match in re.finditer(pattern, text):
                entities.append({"text": match.group(), "type": entity_type})
        except re.error:
            continue
    return FulltextEntityExtractor.normalize_entities(entities)


def format_entities_for_planner_prompt(entities: list[dict[str, Any]]) -> str:
    if not entities:
        return "（検出なし）"
    lines = []
    for entity in entities[:20]:
        text = str(entity.get("text") or "").strip()
        entity_type = str(entity.get("type") or "identifier")
        if text:
            lines.append(f"- {text} ({entity_type})")
    return "\n".join(lines) if lines else "（検出なし）"
