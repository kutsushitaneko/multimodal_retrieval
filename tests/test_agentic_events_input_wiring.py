"""Agentic RAG タブのイベント配線（register -> run の入力順）の回帰テスト。

回帰の背景:
register_*_agentic_rag_events のシグネチャ引数順が、呼び出し側
(multimodal_retriever.py)・inputs リスト・run_* 関数の引数順と食い違い、
UI の「最大ステップ数 / 再検索回数上限 / 検索件数 / 参照する情報の種類」が
別パラメータに流れ込んで反映されない不具合があった。

ここでは Gradio を起動せず、呼び出し側と同じ順序でコンポーネント
(役割名そのものを値に持つセンチネル) を register に渡し、run ボタンに
登録される inputs 配列が run 関数の引数名順と一致することを検証する。
特定の引数名にハードコードせず、inspect でシグネチャから期待値を導出する。
"""

from __future__ import annotations

import inspect
from unittest.mock import Mock

from app.ui.react_agentic_events import ReactAgenticRAGEvents
from app.ui.workflow_agentic_events import WorkflowAgenticRAGEvents


def _run_param_names(run_callable) -> list[str]:
    params = list(inspect.signature(run_callable).parameters)
    # 束縛済みメソッドなので self は含まれない
    return params


def _capture_registered_inputs(register_method, run_callable, extra_output_roles):
    """呼び出し側と同じ順序でセンチネルを渡し、run ボタンに登録される inputs を取得する。

    register の必須位置引数は「run 関数が消費する入力コンポーネント群」+
    「出力コンポーネント群」で構成される。前者は run の引数順と一致するはず、
    という不変条件を検証するため、入力ロールには run の引数名をそのまま値として渡す。
    """
    run_button = Mock()
    clear_button = Mock()

    input_roles = _run_param_names(run_callable)
    # 入力センチネル(run 引数名) + 出力センチネル を呼び出し側の順序で渡す
    positional = list(input_roles) + list(extra_output_roles)

    register_method(run_button, clear_button, *positional)

    assert run_button.click.call_count == 1
    captured_inputs = run_button.click.call_args.kwargs["inputs"]
    return captured_inputs, input_roles


def test_react_agentic_events_input_wiring_matches_run_signature():
    events = ReactAgenticRAGEvents(Mock())
    captured_inputs, input_roles = _capture_registered_inputs(
        events.register_react_agentic_rag_events,
        events.run_react_agentic_rag,
        extra_output_roles=[
            "answer_text",
            "referenced_images_gallery",
            "trace_text",
            "selection_reason_text",
        ],
    )

    # run ボタンの inputs は run 関数の引数順と完全一致しなければならない
    assert captured_inputs == input_roles
    # 回帰の核心: UI のスライダー類が正しいパラメータへ流れること
    assert captured_inputs.index("top_k") < captured_inputs.index("max_steps")
    assert "max_steps" in captured_inputs
    assert "reference_type" in captured_inputs


def test_workflow_agentic_events_input_wiring_matches_run_signature():
    events = WorkflowAgenticRAGEvents(Mock())
    captured_inputs, input_roles = _capture_registered_inputs(
        events.register_workflow_agentic_rag_events,
        events.run_workflow_agentic_rag,
        extra_output_roles=[
            "answer_text",
            "referenced_images_gallery",
            "trace_text",
            "selection_reason_text",
        ],
    )

    assert captured_inputs == input_roles
    assert captured_inputs.index("top_k") < captured_inputs.index("max_iterations")
    assert "max_iterations" in captured_inputs
    assert "reference_type" in captured_inputs
