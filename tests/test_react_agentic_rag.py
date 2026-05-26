from unittest.mock import MagicMock, patch

from PIL import Image

from app.react_agentic_rag import ReactAgenticRAGPipeline
from app.ui.react_agentic_events import ReactAgenticRAGEvents
from app.ui.workflow_agentic_events import REFERENCE_TYPE_ALL, REFERENCE_TYPE_CAPTION_ONLY


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
        self.caption_calls.append((query, search_mode, top_k, vector_threshold, keyword_threshold))
        image_id = 1 if search_mode == "ベクトル検索" else 2
        return [make_result(image_id, f"{search_mode}.png", f"{query} のキャプション", search_mode)], query, "", ""

    def search_by_image_text(self, query, top_k, vector_threshold):
        self.image_text_calls.append((query, top_k, vector_threshold))
        return [make_result(3, "image-text.png", f"{query} の画像説明", "画像ベクトル")], query, "", ""

    def search_by_image_embedding(self, uploaded_image, top_k, vector_threshold):
        self.image_embedding_calls.append((uploaded_image, top_k, vector_threshold))
        return [make_result(4, "uploaded.png", "アップロード画像に類似", "画像ベクトル")], "image", "", ""


def test_react_pipeline_executes_json_actions_and_generates_answer():
    controller = MagicMock(side_effect=[
        '{"thought": "猫の候補を探す", "action": "caption_vector_search", "action_input": {"query": "猫"}}',
        '{"thought": "回答に使う候補を選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1", "999", "1"], "reason": "猫に関係"}}',
        '{"thought": "十分なので回答する", "action": "generate_final_answer", "action_input": {}}',
    ])
    answer_generator = MagicMock(return_value="生成回答")
    fake_search = FakeSearchService()
    pipeline = ReactAgenticRAGPipeline(fake_search, top_k=8, max_steps=4, controller_llm_text_generator=controller)

    results = list(pipeline.run_stream("猫について", answer_generator=answer_generator))
    result = results[-1]

    assert fake_search.caption_calls[0][0] == "猫"
    assert result.answer == "生成回答"
    assert result.selection_reason == "猫に関係"
    assert [evidence.id for evidence in result.selected_evidence] == ["1"]
    assert "Thought: 猫の候補を探す" in result.trace
    assert "Action: caption_vector_search" in result.trace
    assert "Observation: selected evidence 1 件。 無効IDは除外しました: 999" in result.trace
    assert "ReAct Agentic RAG 全体 [" in result.trace
    answer_generator.assert_called_once()
    assert "ベクトル検索.png" in answer_generator.call_args.args[2]


def test_react_pipeline_multi_search_runs_query_variants_across_tools():
    controller = MagicMock(side_effect=[
        (
            '{"thought": "複数観点を多角的に探す", "action": "multi_search", '
            '"action_input": {"query_variants": ["猫", "子猫の特徴"], '
            '"tools": ["caption_vector_search", "caption_fulltext_search", "image_vector_text_search"]}}'
        ),
        '{"thought": "回答に使う候補を選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1", "2", "3"], "reason": "多角的に取得したため"}}',
        '{"thought": "回答する", "action": "generate_final_answer", "action_input": {}}',
    ])
    fake_search = FakeSearchService()
    pipeline = ReactAgenticRAGPipeline(fake_search, top_k=8, max_steps=4, controller_llm_text_generator=controller)

    result = pipeline.run("猫について", answer_generator=lambda q, selected, docs: "生成回答")

    assert [(call[0], call[1]) for call in fake_search.caption_calls] == [
        ("猫", "ベクトル検索"),
        ("子猫の特徴", "ベクトル検索"),
        ("猫", "全文検索"),
        ("子猫の特徴", "全文検索"),
    ]
    assert [call[0] for call in fake_search.image_text_calls] == ["猫", "子猫の特徴"]
    assert "Action: multi_search" in result.trace
    assert "queries 2 件, tools 3 種, calls 6 回, evidence 3 件" in result.trace
    assert result.selection_reason == "多角的に取得したため"


def test_react_pipeline_multi_search_reports_invalid_tools_and_empty_queries():
    controller = MagicMock(return_value=(
        '{"thought": "入力を補正しながら探す", "action": "multi_search", '
        '"action_input": {"query_variants": ["", "猫", "猫"], '
        '"tools": ["caption_vector_search", "unknown_tool"]}}'
    ))
    fake_search = FakeSearchService()
    pipeline = ReactAgenticRAGPipeline(fake_search, max_steps=1, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert [(call[0], call[1]) for call in fake_search.caption_calls] == [("猫", "ベクトル検索")]
    assert "無効Toolを無視: unknown_tool" in result.trace
    assert "空または重複クエリーを除外" in result.trace
    assert "最大ステップ到達: 1 step" in result.trace


def test_react_controller_prompt_guides_multi_search_and_tool_strengths():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())

    prompt = pipeline._build_controller_prompt("ORA-00923 とは何ですか？", [], [], [])

    assert "multi_search" in prompt
    assert "caption_vector_search と image_vector_text_search を必ず含める" in prompt
    assert "意味的類似、言い換え、抽象的質問" in prompt
    assert "URL、論文ID、エラーコード、固有名詞、文書内テキスト" in prompt
    assert "画像中のテキスト、スライド、スクリーンショット" in prompt
    assert "質問の分解、言い換え、専門語、固有語、エラーコード" in prompt


def test_react_pipeline_reports_invalid_json_unknown_action_and_max_steps():
    controller = MagicMock(side_effect=[
        "not json",
        '{"thought": "未定義を試す", "action": "unknown_tool", "action_input": {}}',
    ])
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=2, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert result.answer.startswith("❌ ReAct Agentic RAG が最大ステップ数")
    assert "Controller JSON解析エラー" in result.trace
    assert "未定義Actionです: unknown_tool" in result.trace
    assert "最大ステップ到達: 2 step" in result.trace


def test_react_pipeline_image_only_runs_image_vector_search_only():
    controller = MagicMock()
    fake_search = FakeSearchService()
    pipeline = ReactAgenticRAGPipeline(fake_search, max_steps=8, controller_llm_text_generator=controller)

    result = pipeline.run("", uploaded_image=Image.new("RGB", (4, 4)), answer_generator=MagicMock())

    assert len(fake_search.image_embedding_calls) == 1
    assert fake_search.caption_calls == []
    assert fake_search.image_text_calls == []
    assert result.answer.startswith("画像のみ入力として扱い")
    assert result.selected_evidence
    assert "画像のみ入力: 画像ベクトル検索のみ実行します。" in result.trace
    controller.assert_not_called()


def test_react_event_streams_outputs_and_uses_controller_model():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock(side_effect=[
        '{"thought": "検索する", "action": "caption_vector_search", "action_input": {"query": "猫"}}',
        '{"thought": "選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1"], "reason": "選別"}}',
        '{"thought": "回答する", "action": "generate_final_answer", "action_input": {}}',
    ])

    outputs = list(events.run_react_agentic_rag(
        "猫",
        None,
        REFERENCE_TYPE_ALL,
        8,
        4,
        "デフォルト（回答生成）",
        "answer-model",
        0.0,
        1024,
        "Japan Central (Osaka)",
        "controller-model",
    ))
    answer, gallery, trace, reason = outputs[-1]

    assert answer == "生成回答"
    assert gallery.visible is True
    assert len(gallery.value) == 1
    assert reason == "選別"
    assert "Action: caption_vector_search" in trace
    assert any("回答生成中..." in output[2] for output in outputs)
    assert [call.args[1] for call in events._call_text_model.call_args_list] == [
        "controller-model",
        "controller-model",
        "controller-model",
    ]


def test_react_event_streams_multi_search_observation():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock(side_effect=[
        (
            '{"thought": "多角的に検索する", "action": "multi_search", '
            '"action_input": {"query_variants": ["猫", "子猫"], '
            '"tools": ["caption_vector_search", "caption_fulltext_search", "image_vector_text_search"]}}'
        ),
        '{"thought": "選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1", "2", "3"], "reason": "選別"}}',
        '{"thought": "回答する", "action": "generate_final_answer", "action_input": {}}',
    ])

    outputs = list(events.run_react_agentic_rag(
        "猫",
        None,
        REFERENCE_TYPE_ALL,
        8,
        4,
        "デフォルト（回答生成）",
        "answer-model",
        0.0,
        1024,
        "Japan Central (Osaka)",
        "controller-model",
    ))
    answer, gallery, trace, reason = outputs[-1]

    assert answer == "生成回答"
    assert gallery.visible is True
    assert reason == "選別"
    assert "Action: multi_search" in trace
    assert "queries 2 件, tools 3 種, calls 6 回, evidence 3 件" in trace


def test_react_event_image_only_shows_gallery_without_llm_calls():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock()

    outputs = list(events.run_react_agentic_rag(
        "",
        Image.new("RGB", (4, 4)),
        REFERENCE_TYPE_CAPTION_ONLY,
        8,
        4,
        "デフォルト（回答生成）",
        "answer-model",
        0.0,
        1024,
        "Japan Central (Osaka)",
        "controller-model",
    ))
    answer, gallery, trace, reason = outputs[-1]

    assert answer.startswith("画像のみ入力として扱い")
    assert gallery.visible is True
    assert len(gallery.value) == 1
    assert reason == "画像のみ入力のため、画像ベクトル検索結果をそのまま表示しました。"
    assert "画像のみ入力: 画像ベクトル検索のみ実行します。" in trace
    events._generate_answer_with_vlm.assert_not_called()
    events._call_text_model.assert_not_called()
