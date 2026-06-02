"""ReAct 向け検索計画 LLM（コンテキスト分離）。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from app.agentic_rag_common import LLMTextGenerator
from app.entity_patterns import extract_entities_from_text, format_entities_for_planner_prompt
from app.paths import PROMPT_AGENT_REACT_DIR
from app.prompt_loader import load_prompt

VALID_SEARCH_TOOLS = {
    "caption_vector_search",
    "caption_fulltext_search",
    "image_vector_text_search",
    "image_vector_image_search",
}


@dataclass
class SearchTask:
    tool: str
    query: str = ""
    exact_terms: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class SearchPlan:
    is_likely_multihop: bool = False
    current_hop_goal: str = ""
    dependent_aspects: list[str] = field(default_factory=list)
    search_tasks: list[SearchTask] = field(default_factory=list)
    raw_response: str = ""


class SearchPlanner:
    """検索分解・ツール選択・パラメータを専用プロンプトで計画する。"""

    def __init__(
        self,
        llm_text_generator: LLMTextGenerator | None = None,
        *,
        max_search_tasks: int = 8,
        max_retries: int = 1,
    ):
        self.llm_text_generator = llm_text_generator
        self.max_search_tasks = max(1, min(int(max_search_tasks), 12))
        self.max_retries = max(0, min(int(max_retries), 2))

    def plan(
        self,
        question: str,
        *,
        evidence_summary: str = "",
        executed_searches: list[tuple[str, str]] | None = None,
        phase: str = "initial",
        controller_notes: str = "",
        has_uploaded_image: bool = False,
    ) -> tuple[SearchPlan | None, str | None]:
        if self.llm_text_generator is None:
            return None, "Search Planner モデルが設定されていません。"

        question = (question or "").strip()
        if not question:
            return None, "質問が空です。"

        regex_entities = extract_entities_from_text(question)
        executed_summary = self._format_executed_searches(executed_searches)
        prompt = load_prompt(
            os.path.join(PROMPT_AGENT_REACT_DIR, "search_plan.txt"),
            question=question,
            evidence_summary=evidence_summary or "（まだ evidence がありません）",
            executed_summary=executed_summary,
            phase=phase or "initial",
            controller_notes=controller_notes or "（なし）",
            regex_detected_entities=format_entities_for_planner_prompt(regex_entities),
            max_search_tasks=self.max_search_tasks,
        )

        last_error = ""
        for _attempt in range(self.max_retries + 1):
            response_text = ""
            try:
                response_text = str(self.llm_text_generator(prompt) or "")
                parsed = self._parse_llm_json(response_text)
            except Exception as exc:
                last_error = f"Planner JSON解析エラー: {exc}"
                continue
            if not isinstance(parsed, dict):
                last_error = "Planner出力はJSON objectである必要があります。"
                continue
            plan, validation_error = self._build_plan_from_parsed(
                parsed,
                executed_searches=executed_searches,
                has_uploaded_image=has_uploaded_image,
                raw_response=response_text,
            )
            if validation_error:
                last_error = validation_error
                continue
            return plan, None

        return None, last_error or "Planner の計画生成に失敗しました。"

    @staticmethod
    def _format_executed_searches(executed_searches: list[tuple[str, str]] | None) -> str:
        if not executed_searches:
            return "（まだ検索していません）"
        return "\n".join(f"- {action}: {query}" for action, query in executed_searches)

    def _build_plan_from_parsed(
        self,
        parsed: dict[str, Any],
        *,
        executed_searches: list[tuple[str, str]] | None,
        has_uploaded_image: bool,
        raw_response: str,
    ) -> tuple[SearchPlan | None, str | None]:
        executed_keys = {
            self._search_key(action, query)
            for action, query in (executed_searches or [])
        }

        dependent_aspects = []
        aspects = parsed.get("dependent_aspects") or []
        if isinstance(aspects, list):
            dependent_aspects = [str(item).strip() for item in aspects if str(item).strip()]

        raw_tasks = parsed.get("search_tasks") or []
        if not isinstance(raw_tasks, list):
            return None, "search_tasks は配列である必要があります。"

        tasks: list[SearchTask] = []
        seen_task_keys: set[str] = set()
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                continue
            tool = str(raw_task.get("tool") or "").strip()
            if tool not in VALID_SEARCH_TOOLS:
                continue
            if tool == "image_vector_image_search" and not has_uploaded_image:
                continue

            query = str(raw_task.get("query") or "").strip()
            exact_terms_raw = raw_task.get("exact_terms") or []
            exact_terms: list[str] = []
            if isinstance(exact_terms_raw, list):
                exact_terms = [str(term).strip() for term in exact_terms_raw if str(term).strip()]
            elif isinstance(exact_terms_raw, str) and exact_terms_raw.strip():
                exact_terms = [exact_terms_raw.strip()]

            if tool == "caption_fulltext_search" and exact_terms:
                query = query or " OR ".join(exact_terms)
            if tool != "image_vector_image_search" and not query:
                continue

            task_key = self._search_key(tool, query or "アップロード画像")
            if task_key in executed_keys or task_key in seen_task_keys:
                continue
            seen_task_keys.add(task_key)
            tasks.append(
                SearchTask(
                    tool=tool,
                    query=query,
                    exact_terms=exact_terms,
                    reason=str(raw_task.get("reason") or "").strip(),
                )
            )
            if len(tasks) >= self.max_search_tasks:
                break

        if not tasks:
            return None, "有効な search_tasks が1件もありません。"

        return (
            SearchPlan(
                is_likely_multihop=bool(parsed.get("is_likely_multihop")),
                current_hop_goal=str(parsed.get("current_hop_goal") or "").strip(),
                dependent_aspects=dependent_aspects,
                search_tasks=tasks,
                raw_response=raw_response,
            ),
            None,
        )

    @staticmethod
    def _search_key(action: str, query: str) -> str:
        normalized = " ".join(str(query or "").split()).lower()
        return f"{action}::{normalized}"

    @staticmethod
    def _parse_llm_json(response_text: str) -> Any:
        text = str(response_text or "").strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
        return json.loads(text)
