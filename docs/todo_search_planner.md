# Search Planner 分離 実装 TODO

## 分析
- [x] ReAct: `multi_search` は tool×query 直積のためツール別クエリ不可
- [x] Workflow は `decompose.txt` あり、ReAct は Controller に戦略が集中
- [x] 推移的ヒントは属性 regex で誤検知しやすい → 削除し Planner LLM に委譲

## 実装
- [x] `config/entity_patterns.json` + `app/entity_patterns.py` + `SearchQueryGenerator` 連携
- [x] `app/search_planner.py` + `prompt/agent/react/search_plan.txt`
- [x] `plan_and_execute_search` in `react_agentic_rag.py`
- [x] `agentic_search_strategy.py` 推移的断定ヒント削除、regex entities ヘルパ
- [x] `controller.txt` 縮約
- [x] UI: Planner モデル（`components.py`, `react_agentic_events.py`, `multimodal_retriever.py`）
- [x] テスト + README + `.env_example`

## 確認
- [x] `uv run pytest` 関連テスト
