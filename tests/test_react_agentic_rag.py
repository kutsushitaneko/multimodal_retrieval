from unittest.mock import MagicMock, patch

import gradio as gr
from PIL import Image

from app.agentic_rag_common import EvidencePool
from app.react_agentic_rag import ReactAgenticRAGPipeline
from app.ui.components import UIComponents
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
    assert result.trace.count("Thought: 猫の候補を探す") == 1
    assert "Action: caption_vector_search" in result.trace
    assert "Observation: selected evidence 1 件。 無効IDは除外しました: 999" in result.trace
    assert "Step 1: Controller応答 [" in result.trace
    assert "Step 1 完了 [" in result.trace
    assert "Step 1 [" not in result.trace
    assert "Step 3 Observation: 回答生成 LLM [" in result.trace
    assert "]: 最終回答を生成しました" in result.trace
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


def test_react_pipeline_yields_before_controller_after_controller_and_each_multi_search_call():
    controller = MagicMock(side_effect=[
        (
            '{"thought": "複数観点を多角的に探す", "action": "multi_search", '
            '"action_input": {"query_variants": ["猫", "子猫"], '
            '"tools": ["caption_vector_search", "image_vector_text_search"]}}'
        ),
        '{"thought": "回答に使う候補を選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1", "3"], "reason": "選別"}}',
        '{"thought": "回答する", "action": "generate_final_answer", "action_input": {}}',
    ])
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), top_k=8, max_steps=4, controller_llm_text_generator=controller)

    traces = [
        result.trace
        for result in pipeline.run_stream("猫について", answer_generator=lambda q, selected, docs: "生成回答")
    ]

    assert any("Step 1: Controller思考中..." in trace for trace in traces)
    assert any("Step 1: Controller応答 [" in trace and "Action: multi_search" in trace for trace in traces)
    assert any("multi_search 実行中: caption_vector_search / 猫" in trace for trace in traces)
    assert any("caption_vector_search [" in trace and "猫 -> 1件" in trace for trace in traces)
    assert any("multi_search 実行中: image_vector_text_search / 子猫" in trace for trace in traces)
    assert any("queries 2 件, tools 2 種, calls 4 回, evidence 2 件" in trace for trace in traces)
    assert any("select_evidence 実行中..." in trace for trace in traces)


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


def test_react_pipeline_skips_repeated_search_across_steps():
    controller = MagicMock(return_value=(
        '{"thought": "同じ観点で再検索する", "action": "multi_search", '
        '"action_input": {"query_variants": ["猫"], "tools": ["caption_vector_search"]}}'
    ))
    fake_search = FakeSearchService()
    pipeline = ReactAgenticRAGPipeline(
        fake_search, max_steps=5, max_stale_steps=2, controller_llm_text_generator=controller
    )

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert [(call[0], call[1]) for call in fake_search.caption_calls] == [("猫", "ベクトル検索")]
    assert "スキップ" in result.trace
    assert "既出クエリー" in result.trace


def test_react_pipeline_gives_up_when_no_new_evidence():
    controller = MagicMock(return_value=(
        '{"thought": "情報が足りないので再検索する", "action": "multi_search", '
        '"action_input": {"query_variants": ["猫"], "tools": ["caption_vector_search"]}}'
    ))
    pipeline = ReactAgenticRAGPipeline(
        FakeSearchService(), max_steps=8, max_stale_steps=2, controller_llm_text_generator=controller
    )

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert "情報が不足しており回答できません" in result.answer
    assert result.sufficiency.status == "insufficient"
    assert "最大ステップ到達" not in result.trace
    assert "新しい検索結果が得られない" in result.trace


def test_react_controller_prompt_lists_executed_searches_and_allows_giving_up():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())

    prompt_empty = pipeline._build_controller_prompt("質問", [], [], [])
    assert "実行済み検索" in prompt_empty
    assert "（まだ検索していません）" in prompt_empty
    assert "断念することは禁止" not in prompt_empty
    assert "情報が不足しており回答できません" in prompt_empty

    prompt_with_history = pipeline._build_controller_prompt(
        "質問", [], [], [], [("caption_vector_search", "猫")]
    )
    assert "- caption_vector_search: 猫" in prompt_with_history


def test_react_controller_prompt_guides_multi_hop_transitive_questions():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())

    prompt = pipeline._build_controller_prompt("看板の場所の緯度経度は？", [], [], [])

    assert "サブ質問" in prompt
    assert "推移的" in prompt
    assert "マルチホップ" in prompt
    assert "中間結果" in prompt
    assert "次ホップ" in prompt
    assert "中間結果が分かった時点で満足して終了してはいけない" in prompt
    assert "全サブ質問が選択済みevidenceでカバーされているか確認" in prompt


def test_react_controller_prompt_explains_select_evidence_role():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())

    prompt = pipeline._build_controller_prompt("質問", [], [], [])

    assert "最終回答に使う候補を選択して selected_evidence を更新するAction" in prompt
    assert "新しい情報は取得しない" in prompt
    assert "検索不足や未カバーのサブ質問を解消しない" in prompt
    assert "select_evidence ではなく検索Actionまたは multi_search を使う" in prompt
    assert "同じ evidence を再選択しても進捗にならない" in prompt


def test_react_controller_prompt_explains_image_vector_image_search_input():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())

    prompt = pipeline._build_controller_prompt("質問", [], [], [])

    assert "- image_vector_image_search: {}" in prompt
    assert "image_vector_image_search: {\"reason\"" not in prompt
    assert "アップロード画像がある場合のみ使用できる" in prompt
    assert "テキストクエリーや reason は検索条件に使わず" in prompt
    assert "アップロード画像そのものから視覚的に類似する画像を探す" in prompt
    assert "query_variants ごとではなくアップロード画像で1回だけ実行される" in prompt


def test_react_finalize_verifier_blocks_giveup_and_forces_research():
    controller = MagicMock(side_effect=[
        '{"thought": "初回検索", "action": "multi_search", '
        '"action_input": {"query_variants": ["猫"], "tools": ["caption_vector_search"]}}',
        '{"thought": "唯一の候補を選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1"], "reason": "唯一"}}',
        '{"thought": "情報不足だが断念", "action": "generate_final_answer", "action_input": {"answerable": false}}',
        '{"thought": "手がかりで再検索", "action": "caption_vector_search", "action_input": {"query": "子猫の特徴"}}',
        '{"thought": "再度断念", "action": "generate_final_answer", "action_input": {"answerable": false}}',
    ])
    verifier = MagicMock(side_effect=['["子猫の特徴"]', '["子猫の特徴"]'])
    pipeline = ReactAgenticRAGPipeline(
        FakeSearchService(),
        max_steps=6,
        controller_llm_text_generator=controller,
        finalize_verifier_llm_text_generator=verifier,
    )

    result = pipeline.run("猫について", answer_generator=lambda q, selected, docs: "生成回答")

    assert "確定保留" in result.trace
    assert result.answer == "生成回答"
    assert verifier.call_count == 2


def test_react_finalize_verifier_not_called_when_answerable_true():
    controller = MagicMock(side_effect=[
        '{"thought": "初回検索", "action": "multi_search", '
        '"action_input": {"query_variants": ["猫"], "tools": ["caption_vector_search"]}}',
        '{"thought": "候補を選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1"], "reason": "十分"}}',
        '{"thought": "十分なので回答", "action": "generate_final_answer", "action_input": {}}',
    ])
    verifier = MagicMock()
    pipeline = ReactAgenticRAGPipeline(
        FakeSearchService(),
        max_steps=4,
        controller_llm_text_generator=controller,
        finalize_verifier_llm_text_generator=verifier,
    )

    result = pipeline.run("猫について", answer_generator=lambda q, selected, docs: "生成回答")

    verifier.assert_not_called()
    assert result.answer == "生成回答"
    assert "確定保留" not in result.trace


def test_react_finalize_verifier_failopen_on_invalid_output():
    controller = MagicMock(side_effect=[
        '{"thought": "初回検索", "action": "multi_search", '
        '"action_input": {"query_variants": ["猫"], "tools": ["caption_vector_search"]}}',
        '{"thought": "候補を選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1"], "reason": "唯一"}}',
        '{"thought": "情報不足", "action": "generate_final_answer", "action_input": {"answerable": false}}',
    ])
    verifier = MagicMock(return_value="not json")
    pipeline = ReactAgenticRAGPipeline(
        FakeSearchService(),
        max_steps=4,
        controller_llm_text_generator=controller,
        finalize_verifier_llm_text_generator=verifier,
    )

    result = pipeline.run("猫について", answer_generator=lambda q, selected, docs: "生成回答")

    assert verifier.call_count == 1
    assert result.answer == "生成回答"
    assert "確定保留" not in result.trace


def test_react_finalize_verifier_disabled_when_generator_none():
    controller = MagicMock(side_effect=[
        '{"thought": "初回検索", "action": "multi_search", '
        '"action_input": {"query_variants": ["猫"], "tools": ["caption_vector_search"]}}',
        '{"thought": "候補を選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1"], "reason": "唯一"}}',
        '{"thought": "断念", "action": "generate_final_answer", "action_input": {"answerable": false}}',
    ])
    pipeline = ReactAgenticRAGPipeline(
        FakeSearchService(),
        max_steps=4,
        controller_llm_text_generator=controller,
    )

    result = pipeline.run("猫について", answer_generator=lambda q, selected, docs: "生成回答")

    assert result.answer == "生成回答"
    assert "確定保留" not in result.trace


def test_react_controller_prompt_guides_replanning_after_observation():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())

    prompt = pipeline._build_controller_prompt("ある識別子の説明は？", [], [], [])

    assert "検索結果は毎回見直して計画を更新する" in prompt
    assert "最初の検索結果を見て初めて判明することがある" in prompt
    assert "キャプション" in prompt
    assert "中間値" in prompt
    assert "識別子" in prompt
    assert "名称やタイトルだけを返すことが多い" in prompt
    assert "無関係と決めつけて断念してはいけない" in prompt
    assert "未使用の中間値" in prompt
    assert "2312.10997" not in prompt


def test_react_controller_prompt_guides_multi_search_and_tool_strengths():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())

    prompt = pipeline._build_controller_prompt("ORA-00923 とは何ですか？", [], [], [])

    assert "multi_search" in prompt
    assert "caption_vector_search と image_vector_text_search を必ず含める" in prompt
    assert "意味的類似、言い換え、抽象的質問" in prompt
    assert "URL、論文ID、エラーコード" in prompt
    assert "固有名詞" in prompt
    assert "文書内テキスト" in prompt
    assert "OR完全一致検索" in prompt
    assert "IPアドレス、製品名" in prompt
    assert "画像中のテキスト、スライド、スクリーンショット" in prompt
    assert "質問の分解、言い換え、専門語、固有語、エラーコード" in prompt


def test_react_controller_prompt_separates_display_number_from_evidence_id():
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), controller_llm_text_generator=MagicMock())
    evidence = [
        make_result(188, "article.png", "URLの記事タイトル"),
        make_result(544, "author.png", "著者情報"),
    ]
    # FakeSearchServiceのデフォルトIDではなく、実運用で混同しやすいIDを直接summaryに渡す。
    from app.agentic_rag_common import EvidencePool

    pool_evidence = [
        EvidencePool._from_result(evidence[0], "q1", "caption_fulltext_search"),
        EvidencePool._from_result(evidence[1], "q2", "caption_vector_search"),
    ]

    prompt = pipeline._build_controller_prompt("質問", pool_evidence, [], [])

    assert "選択可能な evidence_id 一覧: 188, 544" in prompt
    assert "No. 1" in prompt
    assert "evidence_id: 188" in prompt
    assert "No. は表示順" in prompt
    assert "evidence_id ではない" in prompt
    assert "悪い例" in prompt
    assert '{"evidence_ids": ["1", "2"]}' in prompt
    assert "id=188" not in prompt
    assert "1. id=188" not in prompt


def test_react_pipeline_reports_invalid_json_unknown_action_and_max_steps():
    controller = MagicMock(side_effect=[
        "not json",
        '{"thought": "未定義を試す", "action": "unknown_tool", "action_input": {}}',
    ])
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=2, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert result.answer == "❌ Controller の応答を解釈できなかったため停止しました。"
    assert "Step 1 Error [" in result.trace
    assert "]: Controller JSON解析エラー" in result.trace
    assert "Controller JSON解析エラー" in result.trace
    assert "Step 1 完了 [" in result.trace
    assert "Step 2 Error [" in result.trace
    assert "Controller応答不正: 未定義Actionです: unknown_tool" in result.trace
    assert "未定義Actionです: unknown_tool" in result.trace
    assert "Step 2 完了 [" in result.trace
    assert "Controller応答エラーが2回連続したため停止します。" in result.trace
    assert "最大ステップ到達: 2 step" not in result.trace


def test_react_pipeline_parses_python_dict_style_controller_response():
    controller = MagicMock(side_effect=[
        "{'thought': '猫の候補を探す', 'action': 'caption_vector_search', 'action_input': {'query': '猫',},}",
        "{'thought': '選ぶ', 'action': 'select_evidence', 'action_input': {'evidence_ids': ['1'], 'reason': '選別',},}",
        "{'thought': '回答する', 'action': 'generate_final_answer', 'action_input': {},}",
    ])
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=4, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=lambda q, selected, docs: "生成回答")

    assert result.answer == "生成回答"
    assert "Controller JSON解析エラー" not in result.trace
    assert "Action: caption_vector_search" in result.trace


def test_react_pipeline_stops_after_consecutive_controller_json_errors():
    controller = MagicMock(return_value="{'thought': 'broken', 'action': ")
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=8, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert result.answer == "❌ Controller の応答を解釈できなかったため停止しました。"
    assert controller.call_count == 2
    assert "Controller応答エラーが2回連続したため停止します。" in result.trace
    assert "最大ステップ到達: 8 step" not in result.trace


def test_react_pipeline_stops_after_empty_controller_json_object():
    controller = MagicMock(return_value="{}")
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=8, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert result.answer == "❌ Controller の応答を解釈できなかったため停止しました。"
    assert controller.call_count == 2
    assert "Controller応答不正: thought が空です。" in result.trace
    assert "raw={}" in result.trace
    assert "未定義Actionです:" not in result.trace
    assert "最大ステップ到達: 8 step" not in result.trace


def test_react_pipeline_stops_after_empty_controller_action():
    controller = MagicMock(return_value='{"thought": "考えた", "action": "", "action_input": {}}')
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=8, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert result.answer == "❌ Controller の応答を解釈できなかったため停止しました。"
    assert controller.call_count == 2
    assert "Controller応答不正: action が空です。" in result.trace
    assert "未定義Actionです:" not in result.trace
    assert "最大ステップ到達: 8 step" not in result.trace


def test_react_pipeline_treats_generation_error_text_as_controller_error():
    controller = MagicMock(return_value="OCI API エラー: {'status': 400, 'message': 'max_tokens is too large'}")
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=8, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert result.answer == "❌ Controller の応答を解釈できなかったため停止しました。"
    assert controller.call_count == 2
    assert "Controller呼び出しエラー: OCI API エラー:" in result.trace
    assert "Controller応答不正: thought が空です。" not in result.trace
    assert "最大ステップ到達: 8 step" not in result.trace


def test_react_pipeline_requires_query_for_search_action():
    controller = MagicMock(return_value='{"thought": "検索する", "action": "caption_vector_search", "action_input": {}}')
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=8, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert result.answer == "❌ Controller の応答を解釈できなかったため停止しました。"
    assert controller.call_count == 2
    assert "Controller応答不正: caption_vector_search には action_input.query が必要です。" in result.trace
    assert "caption_vector_search 実行中" not in result.trace
    assert "最大ステップ到達: 8 step" not in result.trace


def test_react_pipeline_reports_generate_answer_without_selected_evidence_as_observation():
    controller = MagicMock(return_value='{"thought": "すぐ回答する", "action": "generate_final_answer", "action_input": {}}')
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=1, controller_llm_text_generator=controller)

    result = pipeline.run("猫", answer_generator=MagicMock())

    assert "Step 1: Controller応答 [" in result.trace
    assert "Step 1 Observation: generate_final_answer の前に select_evidence" in result.trace
    assert "Step 1 完了 [" in result.trace
    assert "Step 1 [" not in result.trace
    assert "最大ステップ到達: 1 step" in result.trace


def test_react_pipeline_generates_answer_with_uploaded_image_without_selected_evidence():
    controller = MagicMock(return_value='{"thought": "画像で回答する", "action": "generate_final_answer", "action_input": {}}')
    answer_generator = MagicMock(return_value="生成回答")
    pipeline = ReactAgenticRAGPipeline(FakeSearchService(), max_steps=1, controller_llm_text_generator=controller)
    uploaded_image = Image.new("RGB", (4, 4), color="white")

    result = pipeline.run("画像について", uploaded_image=uploaded_image, answer_generator=answer_generator)

    assert result.answer == "生成回答"
    answer_generator.assert_called_once_with("画像について", [], "")
    assert "参照evidenceはありませんが、アップロード画像があるため回答生成を試みます。" in result.trace
    assert "最大ステップ到達" not in result.trace


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
    answer_vlm_service = MagicMock()
    answer_vlm_service.get_model_default_temperature.return_value = 1.0
    answer_vlm_service.get_model_default_tokens.return_value = 4096
    answer_vlm_service.get_model_default_region.return_value = "ap-osaka-1"
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=answer_vlm_service):
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
        0.7,
        131000,
        "US Midwest (Chicago)",
        "controller-model",
        0.4,
        2048,
        "US Midwest (Chicago)",
    ))
    answer, gallery, trace, reason, details = outputs[-1]

    assert answer == "生成回答"
    assert gallery.visible is True
    assert len(gallery.value) == 1
    assert reason == "選別"
    assert "Action: caption_vector_search" in trace
    assert any("回答生成中..." in output[2] for output in outputs)
    assert "Controllerモデル: controller-model" in trace
    assert [call.args[1] for call in events._call_text_model.call_args_list] == [
        "controller-model",
        "controller-model",
        "controller-model",
    ]
    assert [call.args[2] for call in events._call_text_model.call_args_list] == [0.4, 0.4, 0.4]
    assert [call.args[3] for call in events._call_text_model.call_args_list] == [2048, 2048, 2048]
    assert [call.args[4] for call in events._call_text_model.call_args_list] == [
        "US Midwest (Chicago)",
        "US Midwest (Chicago)",
        "US Midwest (Chicago)",
    ]


def test_react_vlm_default_resolves_env_model_name():
    all_models = {
        "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)": {
            "model_name": "meta.llama-4-maverick-17b-128e-instruct-fp8",
            "api_type": "oci.llama.chat",
            "vision": True,
            "max_tokens": 4000,
            "default_tokens": 4000,
        },
        "xai.grok-4.3(OCI)": {
            "model_name": "xai.grok-4.3",
            "api_type": "oci.xai.chat",
            "vision": True,
            "max_tokens": 131000,
            "default_tokens": 131000,
            "default_region": "us-chicago-1",
        },
        "google.gemini-2.5-flash-lite(OCI)": {
            "model_name": "google.gemini-2.5-flash-lite",
            "api_type": "oci.gemini.chat",
            "vision": True,
            "max_tokens": 65536,
            "default_tokens": 4096,
        },
    }
    vlm_models = {
        display_name: model_info
        for display_name, model_info in all_models.items()
        if model_info.get("vision") is True
    }
    fake_service = MagicMock()
    fake_service.model_settings = all_models
    fake_service.create_model_setting_components.side_effect = lambda selected_model, model_choices=None, model_label="VLMモデル": (
        gr.Dropdown(label=model_label, choices=model_choices, value=selected_model),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.3),
        gr.Slider(
            label="Max tokens",
            minimum=1,
            maximum=all_models[selected_model]["max_tokens"],
            value=all_models[selected_model]["default_tokens"],
        ),
        gr.Dropdown(
            label="OCIリージョン",
            choices=["Japan Central (Osaka)", "US Midwest (Chicago)"],
            value="US Midwest (Chicago)",
        ),
    )

    with (
        patch.dict("os.environ", {
            "REACT_AGENTIC_VLM_MODEL_ID": "xai.grok-4.3",
            "REACT_AGENTIC_CONTROLLER_MODEL_ID": "google.gemini-2.5-flash-lite",
        }),
        patch(
            "app.vlm_service.build_vlm_ui_initialization",
            return_value=(
                list(vlm_models.keys()),
                "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)",
                ["すべて", "OCI"],
                vlm_models,
                fake_service,
            ),
        ),
        patch("app.vlm_service.VLMService") as mock_vlm_service,
    ):
        mock_vlm_service.return_value.get_model_default_temperature.return_value = 0.0
        with gr.Blocks():
            (
                _,
                vlm_model,
                _,
                vlm_max_tokens,
                vlm_oci_region,
                controller_model,
                _controller_temperature,
                controller_max_tokens,
                controller_oci_region,
            ) = UIComponents().create_react_agentic_vlm_settings()

    assert vlm_model.value == "xai.grok-4.3(OCI)"
    assert vlm_max_tokens.value == 131000
    assert vlm_oci_region.value == "US Midwest (Chicago)"
    assert controller_model.value == "google.gemini-2.5-flash-lite(OCI)"
    assert controller_max_tokens.value == 4096
    assert controller_oci_region.value == "US Midwest (Chicago)"


def test_react_agentic_rag_answer_label_uses_react_name():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())
    selected_evidence = [EvidencePool._from_result(make_result(1, "slide.png", "猫"), "猫", "caption_vector_search")]

    with patch("app.ui.workflow_agentic_events.NLPService") as mock_nlp_service:
        mock_nlp_service.return_value.generate_caption_with_vlm.return_value = "生成回答"
        answer = events._generate_answer_with_vlm(
            "猫",
            selected_evidence,
            "参照情報",
            REFERENCE_TYPE_CAPTION_ONLY,
            "デフォルト（回答生成）",
            "model",
            0.0,
            1024,
            "Japan Central (Osaka)",
        )

    assert answer.startswith("（ReAct Agentic RAG が「slide.png」を参照して回答しました）")
    assert "Workflow Agentic RAG が" not in answer


def test_react_answer_label_includes_uploaded_image_with_selected_evidence():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())
    selected_evidence = [EvidencePool._from_result(make_result(1, "slide.png", "猫"), "猫", "caption_vector_search")]

    with patch("app.ui.workflow_agentic_events.NLPService") as mock_nlp_service:
        mock_nlp_service.return_value.generate_answer_with_vlm_images.return_value = "生成回答"
        answer = events._generate_answer_with_vlm(
            "猫",
            selected_evidence,
            "参照情報",
            REFERENCE_TYPE_ALL,
            "デフォルト（回答生成）",
            "model",
            0.0,
            1024,
            "Japan Central (Osaka)",
            uploaded_image=Image.new("RGB", (4, 4), color="white"),
        )

    assert answer.startswith("（ReAct Agentic RAG が「ユーザーがアップロードした画像」「slide.png」を参照して回答しました）")
    assert "Workflow Agentic RAG が" not in answer


def test_react_vlm_generates_with_uploaded_image_without_selected_evidence():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())

    with patch("app.ui.workflow_agentic_events.NLPService") as mock_nlp_service:
        mock_nlp_service.return_value.generate_answer_with_vlm_images.return_value = "生成回答"
        answer = events._generate_answer_with_vlm(
            "アップロード画像について説明してください",
            [],
            "",
            REFERENCE_TYPE_CAPTION_ONLY,
            "デフォルト（回答生成）",
            "model",
            0.0,
            1024,
            "Japan Central (Osaka)",
            uploaded_image=Image.new("RGB", (4, 4), color="white"),
        )

    assert answer.startswith("（ReAct Agentic RAG が「ユーザーがアップロードした画像」を参照して回答しました）")
    mock_nlp_service.return_value.generate_answer_with_vlm_images.assert_called_once()
    call_kwargs = mock_nlp_service.return_value.generate_answer_with_vlm_images.call_args.kwargs
    assert len(call_kwargs["image_paths"]) == 1
    assert "検索で見つかった参照画像がある場合は2枚目以降です" in call_kwargs["prompt_text"]


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
        0.4,
        2048,
        "US Midwest (Chicago)",
    ))
    answer, gallery, trace, reason, details = outputs[-1]

    assert answer == "生成回答"
    assert gallery.visible is True
    assert reason == "選別"
    assert "Action: multi_search" in trace
    assert "queries 2 件, tools 3 種, calls 6 回, evidence 3 件" in trace


def test_react_event_passes_uploaded_image_to_vlm_when_evidence_found():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock(side_effect=[
        '{"thought": "検索する", "action": "caption_vector_search", "action_input": {"query": "猫"}}',
        '{"thought": "選ぶ", "action": "select_evidence", "action_input": {"evidence_ids": ["1"], "reason": "選別"}}',
        '{"thought": "回答する", "action": "generate_final_answer", "action_input": {}}',
    ])
    uploaded_image = Image.new("RGB", (4, 4), color="white")

    outputs = list(events.run_react_agentic_rag(
        "猫",
        uploaded_image,
        REFERENCE_TYPE_ALL,
        8,
        4,
        "デフォルト（回答生成）",
        "answer-model",
        0.0,
        1024,
        "Japan Central (Osaka)",
        "controller-model",
        0.4,
        2048,
        "US Midwest (Chicago)",
    ))
    answer, _, _, _, _ = outputs[-1]

    assert answer == "生成回答"
    assert events._generate_answer_with_vlm.call_args.kwargs["uploaded_image"] is uploaded_image


def test_react_event_does_not_fallback_controller_to_answer_vlm():
    with patch("app.ui.workflow_agentic_events.VLMServiceFactory.create_answer_vlm_service", return_value=MagicMock()):
        events = ReactAgenticRAGEvents(FakeSearchService())
    events._generate_answer_with_vlm = MagicMock(return_value="生成回答")
    events._call_text_model = MagicMock()

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
        "",
        0.4,
        2048,
        "US Midwest (Chicago)",
    ))
    answer, gallery, trace, reason, details = outputs[-1]

    assert answer == "❌ ReAct Controllerモデルが設定されていません。"
    assert gallery.visible is False
    assert trace == ""
    assert reason == ""
    events._call_text_model.assert_not_called()
    events._generate_answer_with_vlm.assert_not_called()


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
        0.4,
        2048,
        "US Midwest (Chicago)",
    ))
    answer, gallery, trace, reason, details = outputs[-1]

    assert answer.startswith("画像のみ入力として扱い")
    assert gallery.visible is True
    assert len(gallery.value) == 1
    assert reason == "画像のみ入力のため、画像ベクトル検索結果をそのまま表示しました。"
    assert "画像のみ入力: 画像ベクトル検索のみ実行します。" in trace
    events._generate_answer_with_vlm.assert_not_called()
    events._call_text_model.assert_not_called()
