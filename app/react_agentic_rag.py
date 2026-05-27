from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from app.agentic_rag_common import (
    AnswerGenerator,
    Evidence,
    EvidencePool,
    LLMTextGenerator,
    SufficiencyDecision,
    format_documents,
)


@dataclass
class ReactStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str = ""


@dataclass
class ReactAgenticRAGResult:
    answer: str
    selected_evidence: list[Evidence]
    trace: str
    selection_reason: str
    sufficiency: SufficiencyDecision
    all_evidence: list[Evidence]
    steps: list[ReactStep]


class ReactToolRegistry:
    """ReAct Controllerから許可されたActionだけを実行するToolRegistry。"""

    SEARCH_ACTIONS = {
        "caption_vector_search",
        "caption_fulltext_search",
        "image_vector_text_search",
        "image_vector_image_search",
    }

    def __init__(self, pipeline: "ReactAgenticRAGPipeline", uploaded_image=None):
        self.pipeline = pipeline
        self.uploaded_image = uploaded_image

    def execute(
        self,
        action: str,
        action_input: dict[str, Any],
        pool: EvidencePool,
        selected_evidence: list[Evidence],
    ) -> tuple[str, list[Evidence], str, bool, str]:
        final_result = None
        for result in self.iter_execute(action, action_input, pool, selected_evidence):
            final_result = result
        if final_result is not None:
            return final_result
        return f"未定義Actionです: {action}", selected_evidence, "", False, ""

    def iter_execute(
        self,
        action: str,
        action_input: dict[str, Any],
        pool: EvidencePool,
        selected_evidence: list[Evidence],
    ):
        if action == "multi_search":
            yield from self._iter_execute_multi_search(action_input, pool, selected_evidence)
            return
        if action in self.SEARCH_ACTIONS:
            query = str(action_input.get("query") or "").strip()
            yield (
                f"{action} 実行中: {query or 'アップロード画像'}",
                selected_evidence,
                "",
                False,
                "",
            )
            yield self._execute_search(action, action_input, pool, selected_evidence)
            return
        if action == "select_evidence":
            yield "select_evidence 実行中...", selected_evidence, "", False, ""
            yield self._select_evidence(action_input, pool, selected_evidence)
            return
        if action == "generate_final_answer":
            yield "", selected_evidence, "", True, ""
            return
        yield f"未定義Actionです: {action}", selected_evidence, "", False, ""

    def _execute_search(
        self,
        action: str,
        action_input: dict[str, Any],
        pool: EvidencePool,
        selected_evidence: list[Evidence],
    ) -> tuple[str, list[Evidence], str, bool, str]:
        query = str(action_input.get("query") or "").strip()
        if action != "image_vector_image_search" and not query:
            return "検索Actionには action_input.query が必要です。", selected_evidence, "", False, ""

        started_at = time.perf_counter()
        try:
            if action == "caption_vector_search":
                results, _, _, _ = self.pipeline.search_service.search_by_caption(
                    query,
                    "ベクトル検索",
                    self.pipeline.top_k,
                    self.pipeline.vector_threshold,
                    self.pipeline.keyword_threshold,
                )
                pool.add_many(results, query, action)
            elif action == "caption_fulltext_search":
                results, _, _, _ = self.pipeline.search_service.search_by_caption(
                    query,
                    "全文検索",
                    self.pipeline.top_k,
                    self.pipeline.vector_threshold,
                    self.pipeline.keyword_threshold,
                )
                pool.add_many(results, query, action)
            elif action == "image_vector_text_search":
                results, _, _, _ = self.pipeline.search_service.search_by_image_text(
                    query,
                    self.pipeline.top_k,
                    self.pipeline.vector_threshold,
                )
                pool.add_many(results, query, action)
            else:
                if self.uploaded_image is None:
                    return "アップロード画像がないため image_vector_image_search は実行できません。", selected_evidence, "", False, ""
                results, _, _, _ = self.pipeline.search_service.search_by_image_embedding(
                    self.uploaded_image,
                    self.pipeline.top_k,
                    self.pipeline.vector_threshold,
                )
                pool.add_many(results, "アップロード画像", action)
        except Exception as exc:
            return f"{action} エラー [{self.pipeline._elapsed_ms(started_at)}]: {exc}", selected_evidence, "", False, ""

        return (
            f"{action} [{self.pipeline._elapsed_ms(started_at)}]: {query or 'アップロード画像'} -> {len(results)}件, evidence {len(pool.all())} 件",
            selected_evidence,
            "",
            False,
            "",
        )

    def _execute_multi_search(
        self,
        action_input: dict[str, Any],
        pool: EvidencePool,
        selected_evidence: list[Evidence],
    ) -> tuple[str, list[Evidence], str, bool, str]:
        final_result = None
        for result in self._iter_execute_multi_search(action_input, pool, selected_evidence):
            final_result = result
        return final_result or ("multi_search は実行されませんでした。", selected_evidence, "", False, "")

    def _iter_execute_multi_search(
        self,
        action_input: dict[str, Any],
        pool: EvidencePool,
        selected_evidence: list[Evidence],
    ):
        raw_queries = action_input.get("query_variants") or action_input.get("queries") or []
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        if not isinstance(raw_queries, list):
            raw_queries = []
        queries = []
        seen_queries = set()
        for raw_query in raw_queries:
            query = str(raw_query or "").strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            queries.append(query)

        raw_tools = action_input.get("tools") or []
        if isinstance(raw_tools, str):
            raw_tools = [raw_tools]
        if not isinstance(raw_tools, list):
            raw_tools = []
        tools = [str(tool or "").strip() for tool in raw_tools if str(tool or "").strip()]
        invalid_tools = [tool for tool in tools if tool not in self.SEARCH_ACTIONS]
        valid_tools = []
        for tool in tools:
            if tool in self.SEARCH_ACTIONS and tool not in valid_tools:
                valid_tools.append(tool)

        if not queries and "image_vector_image_search" not in valid_tools:
            yield "multi_search には query_variants の非空配列が必要です。", selected_evidence, "", False, ""
            return
        if not valid_tools:
            yield "multi_search には有効な tools が必要です。", selected_evidence, "", False, ""
            return

        started_at = time.perf_counter()
        observations = []
        executed_count = 0
        for tool in valid_tools:
            if tool == "image_vector_image_search":
                yield f"multi_search 実行中: {tool} アップロード画像", selected_evidence, "", False, ""
                observation, selected_evidence, _, _, _ = self._execute_search(tool, {}, pool, selected_evidence)
                observations.append(observation)
                executed_count += 1
                yield observation, selected_evidence, "", False, ""
                continue
            for query in queries:
                yield f"multi_search 実行中: {tool} / {query}", selected_evidence, "", False, ""
                observation, selected_evidence, _, _, _ = self._execute_search(
                    tool,
                    {"query": query},
                    pool,
                    selected_evidence,
                )
                observations.append(observation)
                executed_count += 1
                yield observation, selected_evidence, "", False, ""

        notes = []
        if invalid_tools:
            notes.append(f"無効Toolを無視: {', '.join(invalid_tools)}")
        if len(queries) < len(raw_queries):
            notes.append("空または重複クエリーを除外")
        detail = "\n    ".join(observations)
        note_text = f"\n    {'; '.join(notes)}" if notes else ""
        yield (
            f"multi_search [{self.pipeline._elapsed_ms(started_at)}]: "
            f"queries {len(queries)} 件, tools {len(valid_tools)} 種, calls {executed_count} 回, "
            f"evidence {len(pool.all())} 件\n    {detail}{note_text}",
            selected_evidence,
            "",
            False,
            "",
        )

    def _select_evidence(
        self,
        action_input: dict[str, Any],
        pool: EvidencePool,
        selected_evidence: list[Evidence],
    ) -> tuple[str, list[Evidence], str, bool, str]:
        raw_ids = action_input.get("evidence_ids") or action_input.get("selected_evidence_ids") or []
        if not isinstance(raw_ids, list):
            return "select_evidence には evidence_ids の配列が必要です。", selected_evidence, "", False, ""

        evidence_by_id = {item.id: item for item in pool.all()}
        selected = []
        seen = set()
        missing_ids = []
        for raw_id in raw_ids:
            evidence_id = str(raw_id)
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            if evidence_id not in evidence_by_id:
                missing_ids.append(evidence_id)
                continue
            selected.append(evidence_by_id[evidence_id])

        if not selected:
            return "有効な evidence ID が選択されませんでした。", selected_evidence, "", False, ""

        selected = selected[: self.pipeline.max_selected_evidence]
        reason = str(action_input.get("reason") or f"ReAct Controllerが{len(selected)}件を選択しました。")
        missing_note = f" 無効IDは除外しました: {', '.join(missing_ids)}" if missing_ids else ""
        return f"selected evidence {len(selected)} 件。{missing_note}".strip(), selected, reason, False, ""


class ReactAgenticRAGPipeline:
    """Thought/Action/Observationを繰り返すReAct型Agentic RAGパイプライン。"""

    ALLOWED_ACTIONS = {
        "multi_search",
        "caption_vector_search",
        "caption_fulltext_search",
        "image_vector_text_search",
        "image_vector_image_search",
        "select_evidence",
        "generate_final_answer",
    }

    def __init__(
        self,
        search_service,
        *,
        top_k: int = 8,
        max_steps: int = 8,
        vector_threshold: float = 0.25,
        keyword_threshold: float = 0,
        max_selected_evidence: int = 6,
        controller_llm_text_generator: LLMTextGenerator | None = None,
        controller_model_name: str = "",
        max_consecutive_parse_errors: int = 2,
    ):
        self.search_service = search_service
        self.top_k = self._normalize_int(top_k, 8, 1, 24)
        self.max_steps = self._normalize_int(max_steps, 8, 1, 12)
        self.vector_threshold = vector_threshold
        self.keyword_threshold = keyword_threshold
        self.max_selected_evidence = self._normalize_int(max_selected_evidence, 6, 1, 12)
        self.controller_llm_text_generator = controller_llm_text_generator
        self.controller_model_name = controller_model_name
        self.max_consecutive_parse_errors = self._normalize_int(max_consecutive_parse_errors, 2, 1, 5)

    def run(
        self,
        question: str,
        *,
        uploaded_image=None,
        answer_generator: AnswerGenerator | None = None,
    ) -> ReactAgenticRAGResult:
        final_result = None
        for result in self.run_stream(question, uploaded_image=uploaded_image, answer_generator=answer_generator):
            final_result = result
        if final_result is not None:
            return final_result
        decision = SufficiencyDecision("insufficient", "質問文または画像が必要です。", ["質問文"])
        return ReactAgenticRAGResult("❌ 質問文を入力してください。", [], "", "", decision, [], [])

    def run_stream(
        self,
        question: str,
        *,
        uploaded_image=None,
        answer_generator: AnswerGenerator | None = None,
    ):
        total_started_at = time.perf_counter()
        question = (question or "").strip()
        trace: list[str] = []
        steps: list[ReactStep] = []
        pool = EvidencePool()
        selected_evidence: list[Evidence] = []
        selection_reason = ""
        processing = SufficiencyDecision("processing", "ReActループ処理中です。")

        def build_result(answer: str = "", sufficiency: SufficiencyDecision | None = None):
            return self._build_result(
                answer=answer,
                selected_evidence=selected_evidence,
                trace=trace,
                selection_reason=selection_reason,
                sufficiency=sufficiency or processing,
                all_evidence=pool.all(),
                steps=steps,
            )

        if not question and uploaded_image is None:
            decision = SufficiencyDecision("insufficient", "質問文または画像が必要です。", ["質問文"])
            yield ReactAgenticRAGResult("❌ 質問文を入力してください。", [], "", "", decision, [], [])
            return

        if not question and uploaded_image is not None:
            trace.append("画像のみ入力: 画像ベクトル検索のみ実行します。")
            yield build_result()
            registry = ReactToolRegistry(self, uploaded_image)
            observation, selected_evidence, selection_reason, _, _ = registry.execute(
                "image_vector_image_search",
                {},
                pool,
                selected_evidence,
            )
            trace.append(observation)
            selected_evidence = pool.all()
            selection_reason = "画像のみ入力のため、画像ベクトル検索結果をそのまま表示しました。"
            decision = SufficiencyDecision("sufficient", f"画像類似検索で{len(selected_evidence)}件の候補が見つかりました。")
            answer = (
                f"画像のみ入力として扱い、画像ベクトル検索で類似画像を{len(selected_evidence)}件見つけました。"
                "参照画像ギャラリーに検索結果を表示します。"
            )
            trace.append(f"ReAct Agentic RAG 全体 [{self._elapsed_ms(total_started_at)}]")
            yield build_result(answer=answer, sufficiency=decision)
            return

        if self.controller_llm_text_generator is None:
            decision = SufficiencyDecision("insufficient", "ReAct Controllerモデルが設定されていません。", ["Controllerモデル"])
            yield self._build_result("❌ ReAct Controllerモデルが設定されていません。", [], trace, "", decision, pool.all(), steps)
            return

        registry = ReactToolRegistry(self, uploaded_image)
        if self.controller_model_name:
            trace.append(f"Controllerモデル: {self.controller_model_name}")
            yield build_result()
        consecutive_controller_errors = 0
        for step_index in range(1, self.max_steps + 1):
            started_at = time.perf_counter()
            trace.append(f"Step {step_index}: Controller思考中...")
            yield build_result()
            controller_started_at = time.perf_counter()
            parsed, controller_error = self._call_controller(question, pool.all(), selected_evidence, steps)
            controller_elapsed = self._elapsed_ms(controller_started_at)
            if controller_error:
                step = ReactStep(
                    thought="Controller出力を解釈できませんでした。",
                    action="invalid_controller_response",
                    action_input={},
                    observation=controller_error,
                )
                steps.append(step)
                trace.append(f"Step {step_index} Error [{controller_elapsed}]: {controller_error}")
                self._append_step_trace(trace, step_index, self._elapsed_ms(started_at))
                consecutive_controller_errors += 1
                if consecutive_controller_errors >= self.max_consecutive_parse_errors:
                    trace.append(f"Controller応答エラーが{consecutive_controller_errors}回連続したため停止します。")
                    trace.append(f"ReAct Agentic RAG 全体 [{self._elapsed_ms(total_started_at)}]")
                    decision = SufficiencyDecision("insufficient", "Controller応答エラーが連続しました。", ["Controller JSON出力"])
                    yield build_result(answer="❌ Controller の応答を解釈できなかったため停止しました。", sufficiency=decision)
                    return
                yield build_result()
                continue
            consecutive_controller_errors = 0

            step = ReactStep(
                thought=str(parsed.get("thought") or "").strip(),
                action=str(parsed.get("action") or "").strip(),
                action_input=parsed.get("action_input") if isinstance(parsed.get("action_input"), dict) else {},
            )
            trace.append(
                f"Step {step_index}: Controller応答 [{controller_elapsed}]\n"
                f"  Thought: {step.thought or '（なし）'}\n"
                f"  Action: {step.action}\n"
                f"  Action Input: {json.dumps(step.action_input, ensure_ascii=False)}"
            )
            yield build_result()
            if step.action not in self.ALLOWED_ACTIONS:
                step.observation = f"未定義Actionです: {step.action}"
                steps.append(step)
                trace.append(f"Step {step_index} Observation: {step.observation}")
                self._append_step_trace(trace, step_index, self._elapsed_ms(started_at))
                yield build_result()
                continue

            if step.action == "generate_final_answer":
                if not selected_evidence:
                    step.observation = "generate_final_answer の前に select_evidence で参照 evidence を選択してください。"
                    steps.append(step)
                    trace.append(f"Step {step_index} Observation: {step.observation}")
                    self._append_step_trace(trace, step_index, self._elapsed_ms(started_at))
                    yield build_result()
                    continue
                documents = format_documents(selected_evidence)
                trace.append("回答生成中...")
                yield build_result()
                answer_started_at = time.perf_counter()
                answer = answer_generator(question, selected_evidence, documents) if answer_generator else self.default_answer(question, selected_evidence, documents)
                step.observation = f"回答生成 LLM [{self._elapsed_ms(answer_started_at)}]: 最終回答を生成しました"
                steps.append(step)
                trace.append(f"Step {step_index} Observation: {step.observation}")
                self._append_step_trace(trace, step_index, self._elapsed_ms(started_at))
                trace.append(f"ReAct Agentic RAG 全体 [{self._elapsed_ms(total_started_at)}]")
                decision = SufficiencyDecision("sufficient", "ReAct Controllerが最終回答を生成しました。")
                yield build_result(answer=answer, sufficiency=decision)
                return

            observations = []
            for observation, new_selected, new_reason, _, _ in registry.iter_execute(
                step.action,
                step.action_input,
                pool,
                selected_evidence,
            ):
                selected_evidence = new_selected
                if new_reason:
                    selection_reason = new_reason
                if observation:
                    observations.append(observation)
                    trace.append(f"Step {step_index} Observation: {observation}")
                    yield build_result()
            step.observation = "\n".join(observations)
            steps.append(step)
            self._append_step_trace(trace, step_index, self._elapsed_ms(started_at))
            yield build_result()

        decision = SufficiencyDecision("insufficient", "最大ステップ数に到達しました。", ["ReActステップ"])
        trace.append(f"最大ステップ到達: {self.max_steps} step")
        trace.append(f"ReAct Agentic RAG 全体 [{self._elapsed_ms(total_started_at)}]")
        yield build_result(answer="❌ ReAct Agentic RAG が最大ステップ数に到達しました。", sufficiency=decision)

    def _call_controller(
        self,
        question: str,
        evidence: list[Evidence],
        selected_evidence: list[Evidence],
        steps: list[ReactStep],
    ) -> tuple[dict[str, Any] | None, str | None]:
        prompt = self._build_controller_prompt(question, evidence, selected_evidence, steps)
        response_text = ""
        try:
            response_text = str(self.controller_llm_text_generator(prompt) or "")
            response_error = self._detect_generation_error(response_text)
            if response_error:
                return None, f"Controller呼び出しエラー: {response_error}; raw={self._format_raw_controller_response(response_text)}"
            parsed = self._parse_llm_json(response_text)
        except Exception as exc:
            return None, f"Controller JSON解析エラー: {exc}; raw={self._format_raw_controller_response(response_text)}"
        if not isinstance(parsed, dict):
            return None, f"Controller出力はJSON objectである必要があります。 raw={self._format_raw_controller_response(response_text)}"
        validation_error = self._validate_controller_response(parsed)
        if validation_error:
            return None, f"Controller応答不正: {validation_error}; raw={self._format_raw_controller_response(response_text)}"
        return parsed, None

    def _validate_controller_response(self, parsed: dict[str, Any]) -> str | None:
        thought = str(parsed.get("thought") or "").strip()
        action = str(parsed.get("action") or "").strip()
        action_input = parsed.get("action_input")

        if not thought:
            return "thought が空です。"
        if not action:
            return "action が空です。"
        if action not in self.ALLOWED_ACTIONS:
            return f"未定義Actionです: {action}"
        if not isinstance(action_input, dict):
            return "action_input はJSON objectである必要があります。"

        if action in {"caption_vector_search", "caption_fulltext_search", "image_vector_text_search"}:
            if not str(action_input.get("query") or "").strip():
                return f"{action} には action_input.query が必要です。"
        elif action == "multi_search":
            raw_tools = action_input.get("tools")
            if isinstance(raw_tools, str):
                raw_tools = [raw_tools]
            if not isinstance(raw_tools, list) or not [str(tool or "").strip() for tool in raw_tools if str(tool or "").strip()]:
                return "multi_search には action_input.tools の非空配列が必要です。"

            tools = [str(tool or "").strip() for tool in raw_tools if str(tool or "").strip()]
            raw_queries = action_input.get("query_variants") or action_input.get("queries") or []
            if isinstance(raw_queries, str):
                raw_queries = [raw_queries]
            queries = [str(query or "").strip() for query in raw_queries] if isinstance(raw_queries, list) else []
            if "image_vector_image_search" not in tools and not [query for query in queries if query]:
                return "multi_search には action_input.query_variants の非空配列が必要です。"
        elif action == "select_evidence":
            evidence_ids = action_input.get("evidence_ids") or action_input.get("selected_evidence_ids")
            if not isinstance(evidence_ids, list) or not [str(evidence_id or "").strip() for evidence_id in evidence_ids]:
                return "select_evidence には action_input.evidence_ids の非空配列が必要です。"
        return None

    def _build_controller_prompt(
        self,
        question: str,
        evidence: list[Evidence],
        selected_evidence: list[Evidence],
        steps: list[ReactStep],
    ) -> str:
        evidence_summary = self._format_evidence_summary(evidence)
        selected_ids = ", ".join(item.id for item in selected_evidence) or "なし"
        history = self._format_step_history(steps)
        return (
            "あなたはReAct型マルチモーダルRAG Controllerです。\n"
            "Thoughtで次に必要な判断を書き、Actionで許可されたToolを1つだけ選んでください。\n"
            "明確に完全一致検索が必要な場合は caption_fulltext_search を Action として選択してください。\n"
            "それ以外の場合は、多角的な初回検索では multi_search を1つのActionとして使い、複数クエリーと複数検索手段をまとめて指示してください。\n"
            "必ずJSON objectのみを返してください。空のJSON objectや、空文字のthought/actionは禁止です。\n"
            "必須キー: thought, action, action_input。\n"
            "形式: {\"thought\": \"短い思考\", \"action\": \"許可Action名\", \"action_input\": {\"必要なキー\": \"値\"}}\n\n"
            "許可Action:\n"
            "- multi_search: {\"query_variants\": [\"元質問\", \"分解クエリー\", \"言い換え\", \"固有語\"], \"tools\": [\"caption_vector_search\", \"caption_fulltext_search\", \"image_vector_text_search\"]}\n"
            "- caption_vector_search: {\"query\": \"検索クエリー\"}。意味的類似、言い換え、抽象的質問、関連概念の発見に強い。\n"
            "- caption_fulltext_search: {\"query\": \"検索クエリー\"}。質問中の固有表現をOR完全一致検索に変換し、URL、論文ID、エラーコード、IPアドレス、製品名、固有名詞、文書内テキストなど、ベクトル検索が苦手な語を補完する。\n"
            "- image_vector_text_search: {\"query\": \"検索クエリー\"}。画像内容だけでなく、画像中のテキスト、スライド、スクリーンショット、図表の情報発見にも有効。\n"
            "- image_vector_image_search: {\"reason\": \"入力画像に類似する画像を探す理由\"}。アップロード画像と視覚的に類似する画像を探す。\n"
            "- select_evidence: {\"evidence_ids\": [\"188\", \"544\"], \"reason\": \"選別理由\"}\n"
            "- generate_final_answer: {\"reason\": \"選別済みevidenceで回答できる理由\"}\n\n"
            "進め方:\n"
            "1. 初回検索では原則 multi_search を使い、caption_vector_search と image_vector_text_search を必ず含める。固有名詞、URL、論文ID、エラーコード、IPアドレス、製品名、文書内テキストが重要なら caption_fulltext_search も含める。\n"
            "2. 複合質問では query_variants に、質問の分解、言い換え、専門語、固有語、エラーコードなどを含め、検索の幅を広げる。\n"
            "3. caption_fulltext_search だけに偏らず、テキストベクトル検索と画像ベクトル検索の相補性を使う。\n"
            "4. 追加で不足が見えた場合は、単一検索Actionまたは multi_search を再度使ってよい。\n"
            "5. 回答に十分な候補が揃ったら select_evidence を実行する。\n"
            "6. select_evidence の evidence_ids には、検索候補の evidence_id の値だけを入れる。No. は表示順であり evidence_id ではないため、絶対に evidence_ids に入れない。\n"
            "7. select_evidence 実行前に、指定するIDが「選択可能な evidence_id 一覧」に存在することを確認する。\n"
            "8. 悪い例: {\"evidence_ids\": [\"1\", \"2\"]}。理由: 1, 2 は No. であり evidence_id ではない。\n"
            "9. select_evidence 後に generate_final_answer を実行して終了する。\n"
            "10. 不正なObservationがあれば、次のActionで修正する。\n\n"
            f"ユーザー質問:\n{question}\n\n"
            f"現在のevidence:\n{evidence_summary}\n\n"
            f"選択済みevidence: {selected_ids}\n\n"
            f"実行履歴:\n{history}"
        )

    def _format_evidence_summary(self, evidence: list[Evidence]) -> str:
        if not evidence:
            return "（検索候補なし）"
        visible_evidence = evidence[:50]
        selectable_ids = ", ".join(item.id for item in visible_evidence)
        lines = [
            "重要:",
            "- select_evidence の evidence_ids には、下記の evidence_id の値だけを入れてください。",
            "- No. は表示順であり、evidence_id ではありません。",
            "- evidence_ids に No. を入れてはいけません。",
            f"選択可能な evidence_id 一覧: {selectable_ids}",
            "",
            "検索候補:",
        ]
        for index, item in enumerate(visible_evidence, start=1):
            lines.append(
                "\n".join(
                    [
                        f"No. {index}",
                        f"evidence_id: {item.id}",
                        f"file_name: {item.file_name}",
                        f"source_tool: {item.source_tool}",
                        f"source_query: {item.source_query}",
                        f"caption: {item.caption or '（キャプションなし）'}",
                    ]
                )
            )
        omitted = len(evidence) - len(visible_evidence)
        if omitted > 0:
            lines.append(f"... 省略 {omitted} 件")
        return "\n".join(lines)

    @staticmethod
    def _format_step_history(steps: list[ReactStep]) -> str:
        if not steps:
            return "（履歴なし）"
        return "\n".join(
            f"{index}. Thought: {step.thought}\n"
            f"   Action: {step.action}\n"
            f"   Action Input: {json.dumps(step.action_input, ensure_ascii=False)}\n"
            f"   Observation: {step.observation}"
            for index, step in enumerate(steps, start=1)
        )

    @staticmethod
    def _append_step_trace(trace: list[str], step_index: int, elapsed_ms: str):
        trace.append(f"Step {step_index} 完了 [{elapsed_ms}]")

    def default_answer(self, question: str, selected: list[Evidence], documents: str) -> str:
        if not selected:
            return "❌ 回答に使用できる検索結果が見つかりませんでした。"
        return f"以下の参照情報を元に回答してください。\n\n質問: {question}\n\n参照情報:\n{documents}"

    def _build_result(
        self,
        answer: str,
        selected_evidence: list[Evidence],
        trace: list[str],
        selection_reason: str,
        sufficiency: SufficiencyDecision,
        all_evidence: list[Evidence],
        steps: list[ReactStep],
    ) -> ReactAgenticRAGResult:
        return ReactAgenticRAGResult(
            answer=answer,
            selected_evidence=list(selected_evidence),
            trace="\n".join(f"- {line}" for line in trace),
            selection_reason=selection_reason,
            sufficiency=sufficiency,
            all_evidence=list(all_evidence),
            steps=list(steps),
        )

    @staticmethod
    def _parse_llm_json(response_text: str):
        response_text = str(response_text or "").strip()
        candidate = ReactAgenticRAGPipeline._extract_json_object_text(response_text)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            normalized = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                parsed = ast.literal_eval(normalized)
            except (SyntaxError, ValueError):
                raise
            if not isinstance(parsed, (dict, list)):
                raise ValueError("Controller出力はJSON互換のobjectまたはarrayである必要があります。")
            return parsed

    @staticmethod
    def _extract_json_object_text(response_text: str) -> str:
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1).strip()

        start = response_text.find("{")
        if start < 0:
            return response_text

        depth = 0
        in_string = False
        quote_char = ""
        escaped = False
        for index in range(start, len(response_text)):
            char = response_text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if in_string:
                if char == quote_char:
                    in_string = False
                continue
            if char in {'"', "'"}:
                in_string = True
                quote_char = char
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return response_text[start : index + 1].strip()
        return response_text[start:].strip()

    @staticmethod
    def _format_raw_controller_response(response_text: str, limit: int = 300) -> str:
        normalized = re.sub(r"\s+", " ", str(response_text or "")).strip()
        if not normalized:
            return "（空）"
        if len(normalized) > limit:
            return normalized[:limit] + "..."
        return normalized

    @staticmethod
    def _detect_generation_error(response_text: str) -> str | None:
        normalized = str(response_text or "").strip()
        error_markers = [
            "エラー:",
            "API エラー:",
            "OCI API エラー:",
            "テキスト生成中にエラーが発生しました:",
        ]
        for marker in error_markers:
            if normalized.startswith(marker) or marker in normalized[:80]:
                return normalized.splitlines()[0][:200]
        return None

    @staticmethod
    def _elapsed_ms(started_at: float) -> str:
        return f"{(time.perf_counter() - started_at) * 1000:.1f} ms"

    @staticmethod
    def _normalize_int(value, default: int, minimum: int, maximum: int) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = default
        return max(minimum, min(maximum, normalized))
