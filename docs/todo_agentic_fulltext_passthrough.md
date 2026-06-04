# Agentic fulltext 完全一致パススルー TODO

## 実装
- [x] `SearchQueryGenerator.resolve_agentic_fulltext_query` + `get_morphological_analysis_details` 分岐
- [x] `SearchService.search_by_caption` に `fulltext_query_mode` (legacy | agentic_exact)
- [x] `workflow_agentic_rag._search_caption_fulltext` で `agentic_exact`
- [x] `react_agentic_rag` caption_fulltext で `agentic_exact`
- [x] テスト + README

## 確認
- [x] `uv run pytest` 関連テスト
