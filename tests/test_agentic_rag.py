from unittest.mock import MagicMock, patch

from PIL import Image

from app.agentic_rag import MAX_EVIDENCE_FOR_LLM_PROMPT, AgenticRAGPipeline, EvidencePool
from app.ui.components import UIComponents
from app.ui.agentic_events import AgenticRAGEvents, REFERENCE_TYPE_ALL


def make_result(image_id, file_name, caption, search_mode="ベクトル検索"):
    return {
        "image_id": image_id,
        "file_name": file_name,
        "caption": caption,
        "search_mode": search_mode,
        "distance": 0.1,
        "image": Image.new("RGB", (4, 4), color="white"),
    }


class FakeSearchService:
    def __init__(self):
        self.caption_calls = []
        self.image_text_calls = []
        self.image_embedding_calls = []

    def search_by_caption(self, query, search_mode, top_k, vector_threshold, keyword_threshold):
        self.caption_calls.append((query, search_mode))
        if "missing" in query:
            return [], query, "", ""
        image_id = 1 if search_mode == "ベクトル検索" else 2
        return [make_result(image_id, f"{search_mode}.png", f"{query} のキャプション", search_mode)], query, "", ""

    def search_by_image_text(self, query, top_k, vector_threshold):
        self.image_text_calls.append(query)
        if "missing" in query:
            return [], query, "", ""
        return [make_result(3, "image-text.png", f"{query} の画像説明", "画像ベクトル")], query, "", ""

    def search_by_image_embedding(self, uploaded_image, top_k, vector_threshold):
        self.image_embedding_calls.append(uploaded_image)
        return [make_result(4, "uploaded.png", "アップロード画像に類似", "画像ベクトル")], "image", "", ""


class SixImageSearchService(FakeSearchService):
    def search_by_caption(self, query, search_mode, top_k, vector_threshold, keyword_threshold):
        self.caption_calls.append((query, search_mode))
        if search_mode == "ベクトル検索":
            return [
                make_result(1, "1.png", "caption 1", search_mode),
                make_result(2, "2.png", "caption 2", search_mode),
                make_result(3, "3.png", "caption 3", search_mode),
            ], query, "", ""
        return [
            make_result(4, "4.png", "caption 4", search_mode),
            make_result(5, "5.png", "caption 5", search_mode),
        ], query, "", ""

    def search_by_image_text(self, query, top_k, vector_threshold):
        self.image_text_calls.append(query)
        return [make_result(6, "6.png", "caption 6", "画像ベクトル")], query, "", ""


def test_decompose_question_extracts_compound_and_special_terms():
    pipeline = AgenticRAGPipeline(FakeSearchService())

    queries = pipeline.decompose_question("ORA-00923 とは何ですか。そして https://example.com の図を説明してください")

    assert queries[0].startswith("ORA-00923")
    assert "https://example.com" in queries
    assert any("図を説明" in query for query in queries)


def test_decompose_question_deduplicates_single_question_with_trailing_punctuation():
    pipeline = AgenticRAGPipeline(FakeSearchService())

    queries = pipeline.decompose_question("企業がコーディング・エージェントではなく独自のエージェントを開発する意義はどこにありますか？")

    assert queries == ["企業がコーディング・エージェントではなく独自のエージェントを開発する意義はどこにありますか"]


def test_llm_decompose_question_is_used_when_available():
    llm = MagicMock(return_value='{"subqueries": ["業務特化エージェントの意義", "コーディングエージェントとの違い"]}')
    pipeline = AgenticRAGPipeline(FakeSearchService(), llm_text_generator=llm)

    queries = pipeline.decompose_question("企業が独自のエージェントを開発する意義は？")

    assert queries == ["業務特化エージェントの意義", "コーディングエージェントとの違い"]
    assert llm.call_count == 1


def test_llm_judgement_and_selection_are_used_when_available():
    responses = [
        '{"subqueries": ["猫の特徴"]}',
        '{"status": "sufficient", "reason": "根拠あり", "missing_aspects": []}',
        '{"selected_evidence_ids": ["2"], "reason": "2番目が最適"}',
    ]
    llm = MagicMock(side_effect=responses)
    fake_search = FakeSearchService()
    pipeline = AgenticRAGPipeline(fake_search, top_k=8, max_iterations=1, llm_text_generator=llm)

    result = pipeline.run("猫", answer_generator=lambda q, selected, docs: docs)

    assert result.sufficiency.status == "sufficient"
    assert result.selection_reason == "2番目が最適"
    assert [evidence.id for evidence in result.selected_evidence] == ["2"]
    assert llm.call_count == 3


def test_step_specific_llm_generators_are_used():
    decompose_llm = MagicMock(return_value='{"subqueries": ["初回クエリー"]}')
    sufficiency_llm = MagicMock(return_value='{"status": "sufficient", "reason": "十分", "missing_aspects": []}')
    followup_llm = MagicMock(return_value='{"queries": ["追加クエリー"]}')
    selection_llm = MagicMock(return_value='{"selected_evidence_ids": ["1"], "reason": "選別"}')
    pipeline = AgenticRAGPipeline(
        FakeSearchService(),
        top_k=8,
        max_iterations=1,
        llm_text_generator=selection_llm,
        decompose_llm_text_generator=decompose_llm,
        sufficiency_llm_text_generator=sufficiency_llm,
        followup_llm_text_generator=followup_llm,
    )

    result = pipeline.run("猫", answer_generator=lambda q, selected, docs: docs)

    assert result.sufficiency.reason == "十分"
    assert result.selection_reason == "選別"
    assert decompose_llm.call_count == 1
    assert sufficiency_llm.call_count == 1
    assert followup_llm.call_count == 0
    assert selection_llm.call_count == 1


def test_agentic_model_default_resolves_env_model_name():
    vlm_models = {
        "cohere.command-r-08-2024(OCI)": {
            "model_name": "cohere.command-r-08-2024",
        },
        "google.gemini-2.5-flash-lite(OCI)": {
            "model_name": "google.gemini-2.5-flash-lite",
        },
        "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)": {
            "model_name": "meta.llama-4-maverick-17b-128e-instruct-fp8",
        },
    }

    default = UIComponents._resolve_agentic_model_default(
        vlm_models,
        "cohere.command-r-08-2024",
        "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)",
    )

    assert default == "cohere.command-r-08-2024(OCI)"


def test_agentic_model_choices_include_non_vision_models():
    all_models = {
        "cohere.command-r-08-2024(OCI)": {"vision": False},
        "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)": {"vision": True},
    }

    choices = UIComponents._get_agentic_model_choices(all_models, ["fallback"])

    assert choices == [
        "cohere.command-r-08-2024(OCI)",
        "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)",
    ]


def test_evidence_pool_deduplicates_by_image_id():
    pool = EvidencePool()

    pool.add_many([make_result(1, "a.png", "A")], "q1", "caption_vector")
    pool.add_many([make_result(1, "a-duplicate.png", "B")], "q2", "caption_fulltext")

    assert len(pool.all()) == 1
    assert pool.all()[0].file_name == "a.png"


def test_evidence_prompt_includes_up_to_configured_limit():
    pipeline = AgenticRAGPipeline(FakeSearchService())
    evidence = [
        EvidencePool._from_result(make_result(index, f"{index}.png", f"caption {index}"), "q", "caption_vector")
        for index in range(1, MAX_EVIDENCE_FOR_LLM_PROMPT + 2)
    ]

    prompt = pipeline._format_evidence_for_prompt(evidence)

    assert f"{MAX_EVIDENCE_FOR_LLM_PROMPT}. id: {MAX_EVIDENCE_FOR_LLM_PROMPT}" in prompt
    assert f"{MAX_EVIDENCE_FOR_LLM_PROMPT + 1}. id: {MAX_EVIDENCE_FOR_LLM_PROMPT + 1}" not in prompt


def test_pipeline_runs_multiple_search_modes_and_orders_evidence():
    fake_search = FakeSearchService()
    pipeline = AgenticRAGPipeline(fake_search, top_k=8, max_iterations=0)

    result = pipeline.run("富士山と寺院", answer_generator=lambda q, selected, docs: docs)

    assert fake_search.caption_calls
    assert fake_search.image_text_calls
    assert result.selected_evidence
    assert "caption_vector" in result.trace
    assert "caption_fulltext" in result.trace
    assert "image_vector_text" in result.trace


def test_pipeline_trace_includes_elapsed_times():
    fake_search = FakeSearchService()
    pipeline = AgenticRAGPipeline(fake_search, top_k=8, max_iterations=0)

    result = pipeline.run("ORA-00923 とは何ですか？", answer_generator=lambda q, selected, docs: "answer")

    assert "質問分解 [" in result.trace
    assert "caption_vector [" in result.trace
    assert "caption_fulltext [" in result.trace
    assert "image_vector_text [" in result.trace
    assert "十分性判定 [" in result.trace
    assert "evidence選別・並べ替え [" in result.trace
    assert "回答生成 [" in result.trace
    assert "Agentic RAG 全体 [" in result.trace
    assert " ms]" in result.trace


def test_pipeline_trace_formats_queries_and_llm_input_stats():
    llm = MagicMock(side_effect=[
        '{"subqueries": ["missing 概要", "missing 詳細"]}',
        '{"status": "insufficient", "reason": "不足", "missing_aspects": ["詳細"]}',
        '{"queries": ["猫 概要", "猫 詳細"]}',
        '{"status": "sufficient", "reason": "十分", "missing_aspects": []}',
        '{"selected_evidence_ids": ["1"], "reason": "選別"}',
    ])
    pipeline = AgenticRAGPipeline(FakeSearchService(), top_k=8, max_iterations=1, llm_text_generator=llm)

    result = pipeline.run("猫", answer_generator=lambda q, selected, docs: "answer")

    assert "質問分解 [" in result.trace
    assert "2 件のサブクエリー\n  1. missing 概要\n  2. missing 詳細" in result.trace
    assert "追加検索クエリー生成 [" in result.trace
    assert "2 件\n  1. 猫 概要\n  2. 猫 詳細" in result.trace
    assert "十分性判定入力: evidence" in result.trace
    assert "再判定入力: evidence" in result.trace
    assert "evidence選別・並べ替え入力: evidence" in result.trace
    assert "LLM入力" in result.trace
    assert "実画像入力 0 件" in result.trace
    assert "概算入力" in result.trace
    assert f"上限 {MAX_EVIDENCE_FOR_LLM_PROMPT}" in result.trace


def test_pipeline_default_thresholds_match_existing_search_tab_defaults():
    fake_search = FakeSearchService()
    pipeline = AgenticRAGPipeline(fake_search, top_k=8, max_iterations=0)

    pipeline.run("ORA-00923エラーの原因と意味", answer_generator=lambda q, selected, docs: docs)

    assert pipeline.vector_threshold == 0.25
    assert pipeline.keyword_threshold == 0
    assert all(call[0] == "ORA-00923エラーの原因と意味" for call in fake_search.caption_calls)


def test_pipeline_respects_followup_iteration_limit():
    fake_search = FakeSearchService()
    pipeline = AgenticRAGPipeline(fake_search, top_k=8, max_iterations=1)

    result = pipeline.run("missing", answer_generator=lambda q, selected, docs: "answer")

    assert "追加検索 1/1" in result.trace
    assert "追加検索 2/1" not in result.trace


def test_pipeline_adds_uploaded_image_search():
    fake_search = FakeSearchService()
    pipeline = AgenticRAGPipeline(fake_search, top_k=8, max_iterations=0)

    pipeline.run("猫", uploaded_image=Image.new("RGB", (4, 4)), answer_generator=lambda q, selected, docs: "answer")

    assert len(fake_search.image_embedding_calls) == 1


def test_agentic_rag_event_returns_expected_outputs_without_external_vlm():
    with patch("app.ui.agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = AgenticRAGEvents(FakeSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock(side_effect=[
        '{"subqueries": ["猫"]}',
        '{"status": "sufficient", "reason": "候補あり", "missing_aspects": []}',
    ])
    events._call_text_vlm = MagicMock(side_effect=[
        '{"selected_evidence_ids": ["1"], "reason": "猫に関係するため"}',
    ])

    answer, gallery, trace, reason = events.run_agentic_rag(
        "猫",
        None,
        REFERENCE_TYPE_ALL,
        8,
        0,
        "デフォルト（回答生成）",
        "model",
        0.0,
        1024,
        "Japan Central (Osaka)",
        "decompose-model",
        "sufficiency-model",
        "followup-model",
    )

    assert answer == "生成回答"
    assert gallery.visible is True
    assert len(gallery.value) > 0
    assert "質問分解 [" in trace
    assert reason == "猫に関係するため"
    assert events._call_text_model.call_count == 2
    assert events._call_text_vlm.call_count == 1


def test_agentic_rag_event_gallery_keeps_six_referenced_images_visible():
    with patch("app.ui.agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = AgenticRAGEvents(SixImageSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock(side_effect=[
        '{"subqueries": ["猫"]}',
        '{"status": "sufficient", "reason": "候補あり", "missing_aspects": []}',
    ])
    events._call_text_vlm = MagicMock(return_value='{"selected_evidence_ids": ["1", "2", "3", "4", "5", "6"], "reason": "6件を選別"}')

    answer, gallery, trace, reason = events.run_agentic_rag(
        "猫",
        None,
        REFERENCE_TYPE_ALL,
        8,
        0,
        "デフォルト（回答生成）",
        "model",
        0.0,
        1024,
        "Japan Central (Osaka)",
        "decompose-model",
        "sufficiency-model",
        "followup-model",
    )

    assert answer == "生成回答"
    assert reason == "6件を選別"
    assert len(gallery.value) == 6
    assert gallery.columns == 4
    assert gallery.rows == 2
    assert gallery.height == 480
    assert "参照画像ギャラリー: selected evidence 6 件, 画像表示 6 件" in trace


def test_agentic_rag_event_uses_step_specific_models():
    with patch("app.ui.agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = AgenticRAGEvents(FakeSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock(side_effect=[
        '{"subqueries": ["missing"]}',
        '{"status": "insufficient", "reason": "不足", "missing_aspects": ["詳細"]}',
        '{"queries": ["猫"]}',
        '{"status": "sufficient", "reason": "追加で十分", "missing_aspects": []}',
    ])
    events._call_text_vlm = MagicMock(side_effect=[
        '{"selected_evidence_ids": ["1"], "reason": "選別"}',
    ])

    events.run_agentic_rag(
        "猫",
        None,
        REFERENCE_TYPE_ALL,
        8,
        1,
        "デフォルト（回答生成）",
        "answer-model",
        0.0,
        1024,
        "Japan Central (Osaka)",
        "decompose-model",
        "sufficiency-model",
        "followup-model",
    )

    called_text_models = [call.args[1] for call in events._call_text_model.call_args_list]
    assert called_text_models == [
        "decompose-model",
        "sufficiency-model",
        "followup-model",
        "sufficiency-model",
    ]
    assert events._call_text_vlm.call_args_list[0].args[1] == "answer-model"
