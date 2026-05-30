# プロンプト・設定ファイル集中管理 TODO

## 1. 基盤
- [x] `app/paths.py` 作成（パス定数）
- [x] `app/prompt_loader.py` 作成（テンプレート読込・置換）

## 2. ディレクトリ・ファイル移動
- [x] `prompt/caption/` へ既存 caption テンプレート移動
- [x] `prompt/answer/` へ既存 answer テンプレート移動
- [x] Agent/Retrieval/Snippets テンプレート新規作成
- [x] `config/` へ JSON 移動

## 3. PromptService 拡張
- [x] caption/answer カテゴリ対応
- [x] `render_answer_prompt` 追加
- [x] UI の answer_prompt 直接 I/O を PromptService に統合

## 4. コード変更
- [x] `react_agentic_rag.py` プロンプト外だし
- [x] `workflow_agentic_rag.py` プロンプト外だし
- [x] `events.py` listwise + answer CRUD
- [x] `fulltext_entity_extractor.py` プロンプト外だし
- [x] `workflow_agentic_events.py` answer 読込・snippet
- [x] `database_service.py` / `batch_injestion.py` キャプション統一
- [x] `vlm_service.py` / `components.py` config パス更新

## 5. クリーンアップ
- [x] 旧 `answer_prompt/` 削除
- [x] ルート JSON 削除

## 6. ドキュメント・テスト
- [x] README / docs 更新
- [x] テスト修正（既存テスト 137 passed、1 failed は変更無関係）
- [x] `uv run pytest` 実行
