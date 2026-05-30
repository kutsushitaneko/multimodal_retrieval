# Workflow 十分性・根拠選定 改善 TODO

## 1. プロンプト
- [x] `decompose.txt` 初回ホップ限定・dependent_aspects
- [x] `followup.txt` evidence_summary・chained query
- [x] `evidence_sufficiency.txt` supporting_evidence_ids・binding
- [x] `evidence_selection.txt` 観点カバレッジ

## 2. パイプライン
- [x] `workflow_agentic_rag.py` 逐次フロー・ガード・バリデーション
- [x] `workflow_agentic_events.py` 選別 LLM テキスト化

## 3. テスト
- [x] 新規テスト追加
- [x] `uv run pytest` 実行
