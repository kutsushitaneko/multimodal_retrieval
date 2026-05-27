# Agentic RAG UI レイアウト変更 todo

## 修正対象

- [x] `app/ui/components.py` の `_create_agentic_rag_section_variant()` を修正する。
- [x] `参照する情報の種類`、`検索件数`、`再検索回数上限` または `最大ステップ数` を `Agentic RAG設定` アコーディオンに入れる。
- [x] `実行トレース` を `進捗状況` に変更し、回答欄の上の閉じたアコーディオンに配置する。
- [x] `参照した画像` を回答欄の上の閉じたアコーディオンに配置する。
- [x] `選別・並べ替え理由` を回答欄の上の閉じたアコーディオンに配置する。
- [x] `回答` 欄を進捗、参照画像、選別理由の下に配置する。
- [x] `選別・並べ替え理由` を `参照した画像` アコーディオン内のギャラリー下に移動する。
- [x] タブ順を `ReAct Agentic RAG`、`Workflow Agentic RAG`、`検索と回答生成`、`イメージ管理` に変更する。

## 確認対象

- [x] `create_workflow_agentic_rag_section()` と `create_react_agentic_rag_section()` の返却順序を維持する。
- [x] `multimodal_retriever.py` 側の受け取り順序に影響がないことを確認する。
- [x] `app/ui/workflow_agentic_events.py` と `app/ui/react_agentic_events.py` の outputs 数に影響がないことを確認する。
- [x] 既存テストの UI 文字列検査に影響があれば更新する。
- [x] linter 診断と必要なテストを確認する。
- [x] 追加変更後の linter 診断と必要なテストを確認する。
