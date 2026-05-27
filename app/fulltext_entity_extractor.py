from __future__ import annotations

import json
import re
from typing import Callable


LLMTextGenerator = Callable[[str], str]


class FulltextEntityExtractor:
    """全文検索を補完する固有表現候補をLLM出力から安全に抽出する。"""

    ALLOWED_TYPES = {
        "product_name",
        "service_name",
        "organization",
        "person",
        "place",
        "error_code",
        "paper_id",
        "url",
        "ip_address",
        "file_name",
        "identifier",
        "version",
        "api_name",
    }
    GENERIC_STOP_ENTITIES = {
        "説明",
        "意味",
        "理由",
        "原因",
        "特徴",
        "詳細",
        "サービス",
        "システム",
        "データ",
        "アプリ",
        "アプリケーション",
    }

    def __init__(self, llm_text_generator: LLMTextGenerator | None = None, *, max_entities: int = 8):
        self.llm_text_generator = llm_text_generator
        self.max_entities = max_entities

    def extract_entities(self, query: str) -> list[dict]:
        query = (query or "").strip()
        if not query or self.llm_text_generator is None:
            return []
        try:
            parsed = self._parse_json(self.llm_text_generator(self._build_prompt(query)))
        except Exception:
            return []
        raw_entities = parsed.get("entities") if isinstance(parsed, dict) else []
        return self.normalize_entities(raw_entities, max_entities=self.max_entities)

    @classmethod
    def normalize_entities(cls, raw_entities, *, max_entities: int = 8) -> list[dict]:
        if not isinstance(raw_entities, list):
            return []
        entities = []
        seen = set()
        for raw_entity in raw_entities:
            if isinstance(raw_entity, dict):
                text = str(raw_entity.get("text") or "").strip()
                entity_type = str(raw_entity.get("type") or "").strip()
            else:
                text = str(raw_entity or "").strip()
                entity_type = "identifier"
            if not cls._is_valid_entity_text(text):
                continue
            if entity_type not in cls.ALLOWED_TYPES:
                continue
            key = re.sub(r"\s+", " ", text).lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append({"text": text, "type": entity_type})
            if len(entities) >= max_entities:
                break
        return cls._remove_contained_entities(entities)

    @staticmethod
    def _remove_contained_entities(entities: list[dict]) -> list[dict]:
        filtered = []
        normalized_texts = [re.sub(r"\s+", " ", entity["text"]).lower() for entity in entities]
        for index, entity in enumerate(entities):
            text = normalized_texts[index]
            if any(
                text != other_text and len(text) < len(other_text) and text in other_text
                for other_text in normalized_texts
            ):
                continue
            filtered.append(entity)
        return filtered

    @classmethod
    def _is_valid_entity_text(cls, text: str) -> bool:
        if not text or len(text) > 80:
            return False
        if text in cls.GENERIC_STOP_ENTITIES:
            return False
        if len(text) == 1:
            return False
        return True

    @staticmethod
    def _parse_json(response_text: str):
        response_text = str(response_text or "").strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if fenced_match:
            response_text = fenced_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
        return json.loads(response_text)

    @staticmethod
    def _build_prompt(query: str) -> str:
        return (
            "あなたは全文検索を補完する固有表現抽出器です。\n"
            "ユーザー質問から、ベクトル検索が苦手な完全一致・固有表現検索に有効な語だけを抽出してください。\n"
            "対象: 製品名、サービス名、組織名、人名、地名、エラーコード、論文ID、URL、IPアドレス、ファイル名、API名、バージョン、識別子。\n"
            "一般語（意味、理由、原因、特徴、詳細、サービス、システム等）は除外してください。\n"
            "SQLや検索クエリーは作らず、必ずJSONのみを返してください。\n"
            "形式: {\"entities\": [{\"text\": \"ORA-00923\", \"type\": \"error_code\"}]}\n\n"
            f"ユーザー質問:\n{query}"
        )
