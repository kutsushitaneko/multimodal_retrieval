# ReAct 初回検索戦略分離 TODO

## 修正箇所
- [x] `app/agentic_search_strategy.py` — 軽量ルール分類と hint_text
- [x] `prompt/agent/react/controller.txt` — 初回戦略章・{first_step_hint}
- [x] `app/react_agentic_rag.py` — ヒント注入・trace
- [x] `tests/test_agentic_search_strategy.py` — 分類テスト
- [x] `tests/test_react_agentic_rag.py` — プロンプト・trace テスト更新
- [x] `README.md` — ReAct 節追記
- [x] `uv run pytest` 実行（47 passed）
