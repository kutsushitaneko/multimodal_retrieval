from __future__ import annotations

import re
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from PIL import Image


AnswerGenerator = Callable[[str, list["Evidence"], str], str]
LLMTextGenerator = Callable[[str], str]
MAX_EVIDENCE_FOR_LLM_PROMPT = 50


@dataclass
class Evidence:
    id: str
    image_id: Any = None
    file_name: str = ""
    caption: str = ""
    search_mode: str = ""
    source_query: str = ""
    source_tool: str = ""
    distance: Any = None
    image: Image.Image | None = None


@dataclass
class SufficiencyDecision:
    status: str
    reason: str
    missing_aspects: list[str] = field(default_factory=list)


@dataclass
class AgenticRAGResult:
    answer: str
    selected_evidence: list[Evidence]
    trace: str
    selection_reason: str
    sufficiency: SufficiencyDecision
    all_evidence: list[Evidence]


class EvidencePool:
    """検索方式ごとの候補を、IDを保ちながら重複排除して集約する。"""

    def __init__(self):
        self._evidence: list[Evidence] = []
        self._seen: set[str] = set()

    def add_many(self, results: Iterable[dict], source_query: str, source_tool: str):
        for result in results or []:
            evidence = self._from_result(result, source_query, source_tool)
            if evidence.id in self._seen:
                continue
            self._seen.add(evidence.id)
            self._evidence.append(evidence)

    def all(self) -> list[Evidence]:
        return list(self._evidence)

    @staticmethod
    def _from_result(result: dict, source_query: str, source_tool: str) -> Evidence:
        stable_id = str(result.get("image_id") or result.get("file_name") or f"{source_tool}:{len(str(result))}")
        image = result.get("image")
        return Evidence(
            id=stable_id,
            image_id=result.get("image_id"),
            file_name=result.get("file_name", stable_id),
            caption=result.get("caption", "") or "",
            search_mode=result.get("search_mode", ""),
            source_query=source_query,
            source_tool=source_tool,
            distance=result.get("distance"),
            image=image if isinstance(image, Image.Image) else None,
        )


class AgenticRAGPipeline:
    """既存SearchServiceをToolとして使う、アプリ内Agentic RAGパイプライン。"""

    def __init__(
        self,
        search_service,
        *,
        top_k: int = 8,
        max_iterations: int = 2,
        vector_threshold: float = 0.25,
        keyword_threshold: float = 0,
        max_selected_evidence: int = 6,
        llm_text_generator: LLMTextGenerator | None = None,
        decompose_llm_text_generator: LLMTextGenerator | None = None,
        sufficiency_llm_text_generator: LLMTextGenerator | None = None,
        followup_llm_text_generator: LLMTextGenerator | None = None,
    ):
        self.search_service = search_service
        self.top_k = self._normalize_int(top_k, 8, 1, 24)
        self.max_iterations = self._normalize_int(max_iterations, 2, 0, 3)
        self.vector_threshold = vector_threshold
        self.keyword_threshold = keyword_threshold
        self.max_selected_evidence = self._normalize_int(max_selected_evidence, 6, 1, 12)
        self.llm_text_generator = llm_text_generator
        self.decompose_llm_text_generator = decompose_llm_text_generator or llm_text_generator
        self.sufficiency_llm_text_generator = sufficiency_llm_text_generator or llm_text_generator
        self.followup_llm_text_generator = followup_llm_text_generator or llm_text_generator

    def run(
        self,
        question: str,
        *,
        uploaded_image=None,
        answer_generator: AnswerGenerator | None = None,
    ) -> AgenticRAGResult:
        total_started_at = time.perf_counter()
        question = (question or "").strip()
        trace: list[str] = []
        if not question and uploaded_image is None:
            decision = SufficiencyDecision("insufficient", "質問文または画像が必要です。", ["質問文"])
            return AgenticRAGResult("❌ 質問文を入力してください。", [], "", "", decision, [])

        pool = EvidencePool()
        started_at = time.perf_counter()
        subqueries = self.decompose_question(question)
        trace.append(
            f"質問分解 [{self._elapsed_ms(started_at)}]: {len(subqueries)} 件のサブクエリー"
            f"{self._format_numbered_items(subqueries)}"
        )

        started_at = time.perf_counter()
        self._run_searches(subqueries, pool, trace)
        trace.append(f"初回検索合計 [{self._elapsed_ms(started_at)}]: evidence {len(pool.all())} 件")
        if uploaded_image is not None:
            started_at = time.perf_counter()
            self._run_image_search(uploaded_image, pool, trace)
            trace.append(f"画像類似検索合計 [{self._elapsed_ms(started_at)}]: evidence {len(pool.all())} 件")

        started_at = time.perf_counter()
        decision = self.judge_evidence_sufficiency(question, pool.all())
        trace.append(f"十分性判定 [{self._elapsed_ms(started_at)}]: {decision.status} - {decision.reason}")
        trace.append(f"十分性判定入力: {self._format_llm_input_stats(question, pool.all(), 'sufficiency')}")

        iterations = 0
        while decision.status != "sufficient" and iterations < self.max_iterations:
            started_at = time.perf_counter()
            followup_queries = self.generate_followup_queries(question, decision, subqueries)
            followup_query_lines = self._format_numbered_items(followup_queries) if followup_queries else "\n  なし"
            trace.append(
                f"追加検索クエリー生成 [{self._elapsed_ms(started_at)}]: {len(followup_queries)} 件"
                f"{followup_query_lines}"
            )
            if not followup_queries:
                trace.append("追加検索クエリーが生成されなかったため再検索を停止。")
                break
            iterations += 1
            trace.append(
                f"追加検索 {iterations}/{self.max_iterations}: {len(followup_queries)} 件"
                f"{self._format_numbered_items(followup_queries)}"
            )
            started_at = time.perf_counter()
            self._run_searches(followup_queries, pool, trace, followup=True)
            trace.append(f"追加検索 {iterations} 合計 [{self._elapsed_ms(started_at)}]: evidence {len(pool.all())} 件")
            subqueries.extend(query for query in followup_queries if query not in subqueries)
            started_at = time.perf_counter()
            decision = self.judge_evidence_sufficiency(question, pool.all())
            trace.append(f"再判定 [{self._elapsed_ms(started_at)}]: {decision.status} - {decision.reason}")
            trace.append(f"再判定入力: {self._format_llm_input_stats(question, pool.all(), 'sufficiency')}")

        started_at = time.perf_counter()
        selected, selection_reason = self.filter_and_order_evidence(question, pool.all())
        trace.append(f"evidence選別・並べ替え [{self._elapsed_ms(started_at)}]: {selection_reason}")
        trace.append(f"evidence選別・並べ替え入力: {self._format_llm_input_stats(question, pool.all(), 'selection')}")
        started_at = time.perf_counter()
        documents = self.format_documents(selected)
        trace.append(f"回答用ドキュメント整形 [{self._elapsed_ms(started_at)}]: {len(selected)} 件")
        started_at = time.perf_counter()
        answer = answer_generator(question, selected, documents) if answer_generator else self.default_answer(question, selected, documents)
        trace.append(f"回答生成 [{self._elapsed_ms(started_at)}]")
        trace.append(f"Agentic RAG 全体 [{self._elapsed_ms(total_started_at)}]")

        return AgenticRAGResult(
            answer=answer,
            selected_evidence=selected,
            trace="\n".join(f"- {line}" for line in trace),
            selection_reason=selection_reason,
            sufficiency=decision,
            all_evidence=pool.all(),
        )

    def decompose_question(self, question: str) -> list[str]:
        question = (question or "").strip()
        if not question:
            return []

        llm_queries = self._decompose_question_with_llm(question)
        if llm_queries:
            return llm_queries

        return self._decompose_question_with_rules(question)

    def _decompose_question_with_llm(self, question: str) -> list[str]:
        if self.decompose_llm_text_generator is None:
            return []

        prompt = (
            "あなたはマルチモーダルRAGの検索計画を作るエージェントです。\n"
            "ユーザー質問を、回答に必要な観点ごとの検索サブクエリーへ分解してください。\n"
            "単一観点の質問は無理に分解せず、1件だけ返してください。\n"
            "画像ベクトル検索、キャプションベクトル検索、全文検索の全てに使いやすい自然文にしてください。\n"
            "重複したサブクエリーは禁止です。\n"
            "必ずJSONのみを返してください。\n"
            "形式: {\"subqueries\": [\"query1\", \"query2\"]}\n\n"
            f"ユーザー質問:\n{question}"
        )
        try:
            parsed = self._parse_llm_json(self.decompose_llm_text_generator(prompt))
        except Exception:
            return []

        raw_queries = []
        if isinstance(parsed, dict):
            raw_queries = parsed.get("subqueries") or parsed.get("queries") or []
        elif isinstance(parsed, list):
            raw_queries = parsed
        return self._dedupe_queries(raw_queries)[:5]

    def _decompose_question_with_rules(self, question: str) -> list[str]:
        question = (question or "").strip()
        if not question:
            return []

        parts = re.split(r"[。！？\n]|(?:\s+かつ\s+)|(?:\s+and\s+)|(?:、そして)|(?:そして)", question)
        queries = [part.strip(" 、,") for part in parts if part.strip(" 、,")]

        special_patterns = [
            r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+",
            r"(?<!\d)\d{4}\.\d{4,5}(?!\d)",
            r"\b[A-Z]{2,5}-\d{4,6}\b",
            r"\bORA-\d{5}\b",
        ]
        for pattern in special_patterns:
            for match in re.findall(pattern, question):
                queries.append(match)

        queries.insert(0, question)
        return self._dedupe_queries(queries)[:5]

    def judge_evidence_sufficiency(self, question: str, evidence: list[Evidence]) -> SufficiencyDecision:
        llm_decision = self._judge_evidence_sufficiency_with_llm(question, evidence)
        if llm_decision is not None:
            return llm_decision

        if not evidence:
            return SufficiencyDecision("insufficient", "検索候補がありません。", ["検索候補"])
        if len(evidence) >= 3:
            return SufficiencyDecision("sufficient", f"{len(evidence)}件の候補が見つかりました。")

        keywords = self._keywords(question)
        captions = " ".join(item.caption for item in evidence)
        matched = [keyword for keyword in keywords if keyword and keyword in captions]
        if matched:
            return SufficiencyDecision("sufficient", f"質問語に一致する候補があります: {', '.join(matched[:3])}")
        return SufficiencyDecision("uncertain", "候補数が少なく、質問語との一致も限定的です。", keywords[:3])

    def _judge_evidence_sufficiency_with_llm(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> SufficiencyDecision | None:
        if self.sufficiency_llm_text_generator is None:
            return None

        prompt = self._build_evidence_eval_prompt(question, self._format_evidence_for_prompt(evidence), "sufficiency")
        try:
            parsed = self._parse_llm_json(self.sufficiency_llm_text_generator(prompt))
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None

        status = str(parsed.get("status") or "").strip().lower()
        if status not in {"sufficient", "insufficient", "uncertain"}:
            return None
        missing_aspects = parsed.get("missing_aspects") or []
        if not isinstance(missing_aspects, list):
            missing_aspects = []
        return SufficiencyDecision(
            status=status,
            reason=str(parsed.get("reason") or "LLMが十分性を判定しました。"),
            missing_aspects=[str(item).strip() for item in missing_aspects if str(item).strip()][:5],
        )

    def generate_followup_queries(
        self,
        question: str,
        decision: SufficiencyDecision,
        existing_queries: list[str],
    ) -> list[str]:
        llm_queries = self._generate_followup_queries_with_llm(question, decision, existing_queries)
        if llm_queries:
            return llm_queries

        base_terms = decision.missing_aspects or self._keywords(question)[:2] or [question]
        candidates = []
        for term in base_terms[:2]:
            candidates.append(f"{term} 詳細")
            candidates.append(f"{term} 関連情報")
        unique = []
        for query in candidates:
            query = query.strip()
            if query and query not in existing_queries and query not in unique:
                unique.append(query)
        return unique[:3]

    def _generate_followup_queries_with_llm(
        self,
        question: str,
        decision: SufficiencyDecision,
        existing_queries: list[str],
    ) -> list[str]:
        if self.followup_llm_text_generator is None:
            return []

        prompt = (
            "あなたはマルチモーダルRAGの追加検索クエリーを作るエージェントです。\n"
            "不足観点を埋めるため、画像ベクトル検索、キャプションベクトル検索、全文検索に使える追加クエリーを最大3件作ってください。\n"
            "既存クエリーと重複しないようにしてください。\n"
            "必ずJSONのみを返してください。\n"
            "形式: {\"queries\": [\"query1\", \"query2\"]}\n\n"
            f"ユーザー質問:\n{question}\n\n"
            f"十分性判定: {decision.status}\n理由: {decision.reason}\n不足観点: {', '.join(decision.missing_aspects)}\n"
            f"既存クエリー: {', '.join(existing_queries)}"
        )
        try:
            parsed = self._parse_llm_json(self.followup_llm_text_generator(prompt))
        except Exception:
            return []
        if not isinstance(parsed, dict):
            return []
        return [
            query
            for query in self._dedupe_queries(parsed.get("queries") or [])
            if self._query_key(query) not in {self._query_key(existing) for existing in existing_queries}
        ][:3]

    def filter_and_order_evidence(self, question: str, evidence: list[Evidence]) -> tuple[list[Evidence], str]:
        llm_selection = self._filter_and_order_evidence_with_llm(question, evidence)
        if llm_selection is not None:
            return llm_selection

        keywords = self._keywords(question)

        def score(item: Evidence):
            caption_score = sum(1 for keyword in keywords if keyword and keyword in item.caption)
            source_score = {
                "caption_fulltext": 3,
                "caption_vector": 2,
                "image_vector_text": 2,
                "image_vector_image": 2,
            }.get(item.source_tool, 1)
            return caption_score * 10 + source_score

        selected = sorted(evidence, key=score, reverse=True)[: self.max_selected_evidence]
        reason = f"VLM選別ステップ相当: {len(evidence)}件から回答に使う候補を{len(selected)}件に絞り、質問語との一致と検索方式で並べ替えました。"
        return selected, reason

    def _filter_and_order_evidence_with_llm(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> tuple[list[Evidence], str] | None:
        if self.llm_text_generator is None or not evidence:
            return None

        prompt = self._build_evidence_eval_prompt(question, self._format_evidence_for_prompt(evidence), "selection")
        try:
            parsed = self._parse_llm_json(self.llm_text_generator(prompt))
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None

        selected_ids = parsed.get("selected_evidence_ids") or parsed.get("selected_image_ids") or []
        if not isinstance(selected_ids, list):
            return None
        evidence_by_id = {item.id: item for item in evidence}
        selected = []
        seen = set()
        for selected_id in selected_ids:
            selected_id = str(selected_id)
            if selected_id in seen or selected_id not in evidence_by_id:
                continue
            seen.add(selected_id)
            selected.append(evidence_by_id[selected_id])
        if not selected:
            return None
        reason = str(parsed.get("reason") or f"LLMが{len(evidence)}件から{len(selected)}件を選別・並べ替えました。")
        return selected[: self.max_selected_evidence], reason

    def format_documents(self, selected: list[Evidence]) -> str:
        return "\n\n".join(
            f"{index}. {item.file_name}\n{item.caption or '（キャプションなし）'}"
            for index, item in enumerate(selected, start=1)
        )

    def default_answer(self, question: str, selected: list[Evidence], documents: str) -> str:
        if not selected:
            return "❌ 回答に使用できる検索結果が見つかりませんでした。"
        return f"以下の参照情報を元に回答してください。\n\n質問: {question}\n\n参照情報:\n{documents}"

    def _run_searches(self, queries: list[str], pool: EvidencePool, trace: list[str], followup=False):
        for query in queries:
            self._search_caption_vector(query, pool, trace, followup)
            self._search_caption_fulltext(query, pool, trace, followup)
            self._search_image_by_text(query, pool, trace, followup)

    def _search_caption_vector(self, query: str, pool: EvidencePool, trace: list[str], followup: bool):
        started_at = time.perf_counter()
        try:
            results, _, _, _ = self.search_service.search_by_caption(
                query,
                "ベクトル検索",
                self.top_k,
                self.vector_threshold,
                self.keyword_threshold,
            )
            pool.add_many(results, query, "caption_vector")
            trace.append(f"{'再' if followup else '初回'}検索 caption_vector [{self._elapsed_ms(started_at)}]: {query} -> {len(results)}件")
        except Exception as exc:
            trace.append(f"caption_vector エラー [{self._elapsed_ms(started_at)}]: {query} -> {exc}")

    def _search_caption_fulltext(self, query: str, pool: EvidencePool, trace: list[str], followup: bool):
        started_at = time.perf_counter()
        try:
            results, _, _, _ = self.search_service.search_by_caption(
                query,
                "全文検索",
                self.top_k,
                self.vector_threshold,
                self.keyword_threshold,
            )
            pool.add_many(results, query, "caption_fulltext")
            trace.append(f"{'再' if followup else '初回'}検索 caption_fulltext [{self._elapsed_ms(started_at)}]: {query} -> {len(results)}件")
        except Exception as exc:
            trace.append(f"caption_fulltext エラー [{self._elapsed_ms(started_at)}]: {query} -> {exc}")

    def _search_image_by_text(self, query: str, pool: EvidencePool, trace: list[str], followup: bool):
        started_at = time.perf_counter()
        try:
            results, _, _, _ = self.search_service.search_by_image_text(
                query,
                self.top_k,
                self.vector_threshold,
            )
            pool.add_many(results, query, "image_vector_text")
            trace.append(f"{'再' if followup else '初回'}検索 image_vector_text [{self._elapsed_ms(started_at)}]: {query} -> {len(results)}件")
        except Exception as exc:
            trace.append(f"image_vector_text エラー [{self._elapsed_ms(started_at)}]: {query} -> {exc}")

    def _run_image_search(self, uploaded_image, pool: EvidencePool, trace: list[str]):
        started_at = time.perf_counter()
        try:
            results, _, _, _ = self.search_service.search_by_image_embedding(
                uploaded_image,
                self.top_k,
                self.vector_threshold,
            )
            pool.add_many(results, "アップロード画像", "image_vector_image")
            trace.append(f"画像類似検索 image_vector_image [{self._elapsed_ms(started_at)}] -> {len(results)}件")
        except Exception as exc:
            trace.append(f"image_vector_image エラー [{self._elapsed_ms(started_at)}]: {exc}")

    @staticmethod
    def _keywords(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_.:/-]+|[一-龥ぁ-んァ-ンー]{2,}", text or "")
        stopwords = {"について", "してください", "教えて", "ですか", "とは", "もの", "こと"}
        return [token for token in tokens if token not in stopwords][:8]

    def _format_evidence_for_prompt(self, evidence: list[Evidence]) -> str:
        if not evidence:
            return "（検索候補なし）"
        lines = []
        for index, item in enumerate(evidence[:MAX_EVIDENCE_FOR_LLM_PROMPT], start=1):
            lines.append(
                "\n".join(
                    [
                        f"{index}. id: {item.id}",
                        f"   file_name: {item.file_name}",
                        f"   source_tool: {item.source_tool}",
                        f"   source_query: {item.source_query}",
                        f"   search_mode: {item.search_mode or '不明'}",
                        f"   caption: {item.caption or '（キャプションなし）'}",
                    ]
                )
            )
        return "\n".join(lines)

    def _format_llm_input_stats(self, question: str, evidence: list[Evidence], step: str) -> str:
        evidence_prompt = self._format_evidence_for_prompt(evidence)
        prompt_text = self._build_evidence_eval_prompt(question, evidence_prompt, step)
        total_count = len(evidence)
        llm_input_count = min(total_count, MAX_EVIDENCE_FOR_LLM_PROMPT)
        image_count = sum(1 for item in evidence if isinstance(item.image, Image.Image))
        omitted_count = max(0, total_count - llm_input_count)
        approximate_tokens = self._estimate_tokens(prompt_text)
        return (
            f"evidence {total_count} 件, "
            f"LLM入力 {llm_input_count}/{total_count} 件（上限 {MAX_EVIDENCE_FOR_LLM_PROMPT}）, "
            f"画像付き evidence {image_count} 件, "
            "実画像入力 0 件, "
            f"概算入力 {approximate_tokens} tokens, "
            f"省略 {omitted_count} 件"
        )

    def _build_evidence_eval_prompt(self, question: str, evidence_prompt: str, step: str) -> str:
        if step == "selection":
            instruction = (
                "あなたはマルチモーダルRAGの検索候補を選別・並べ替えするエージェントです。\n"
                "ユーザー質問に回答するために役立つ evidence だけを選び、回答で参照すると自然な順序に並べてください。\n"
                "必ずJSONのみを返してください。\n"
                "形式: {\"selected_evidence_ids\": [\"id1\", \"id2\"], \"reason\": \"短い理由\"}\n\n"
            )
        else:
            instruction = (
                "あなたはマルチモーダルRAGの検索結果を評価するエージェントです。\n"
                "ユーザー質問に回答するために、検索候補が十分か判定してください。\n"
                "status は sufficient, insufficient, uncertain のいずれかです。\n"
                "不足している観点があれば missing_aspects に短い語句で入れてください。\n"
                "必ずJSONのみを返してください。\n"
                "形式: {\"status\": \"sufficient\", \"reason\": \"理由\", \"missing_aspects\": []}\n\n"
            )
        return f"{instruction}ユーザー質問:\n{question}\n\n検索候補:\n{evidence_prompt}"

    @staticmethod
    def _format_numbered_items(items: list[str]) -> str:
        if not items:
            return ""
        return "\n" + "\n".join(f"  {index}. {item}" for index, item in enumerate(items, start=1))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # 日本語を含む混在テキスト向けの粗い目安。実課金トークンとは一致しない。
        return max(1, len(text or "") // 4)

    def _parse_llm_json(self, response_text: str):
        response_text = str(response_text or "").strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", response_text, re.DOTALL)
        if fenced_match:
            response_text = fenced_match.group(1)
        else:
            json_match = re.search(r"\{.*\}|\[.*\]", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
        return json.loads(response_text)

    def _dedupe_queries(self, queries: Iterable[Any]) -> list[str]:
        unique = []
        seen = set()
        for query in queries or []:
            if isinstance(query, dict):
                query = query.get("query") or query.get("text") or query.get("subquery") or ""
            query = str(query).strip(" 、,\n\t")
            query = re.sub(r"[。！？?!]+$", "", query).strip()
            if not query:
                continue
            key = self._query_key(query)
            if key in seen:
                continue
            seen.add(key)
            unique.append(query)
        return unique

    @staticmethod
    def _query_key(query: str) -> str:
        return re.sub(r"\s+", "", str(query or "").strip().rstrip("。！？?!")).lower()

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
