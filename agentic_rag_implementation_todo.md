# Agentic RAG 実装 ToDo

## 修正箇所

- [x] `app/agentic_rag.py` を追加し、UI 非依存の Agentic RAG パイプラインを実装する。
  - 質問分解
  - 画像ベクトル検索、キャプションベクトル検索、全文検索の併用
  - EvidencePool の統合と重複排除
  - 十分性判定
  - 不足観点に基づく追加クエリー生成と再検索
  - evidence の選別・並べ替え
  - 回答生成用ドキュメントと実行トレースの生成
- [x] `app/ui/agentic_events.py` を追加し、Agentic RAG タブ用イベントを既存 `UIEvents` から分離して実装する。
- [x] `app/ui/components.py` に Agentic RAG タブの UI コンポーネント作成メソッドを追加する。
- [x] `multimodal_retriever.py` に第3タブ `Agentic RAG` を追加し、既存2タブはそのまま残す。
- [x] `tests/test_agentic_rag.py` を追加し、質問分解、EvidencePool、再検索上限、選別・並べ替え、イベント出力を unit テストする。

## 確認観点

- [x] 既存の `検索と回答生成` タブと `イメージ管理` タブのイベント登録を変更しすぎていないこと。
- [x] Gradio イベントの inputs / outputs 数が一致していること。
- [x] unit テストでは Oracle DB、外部 VLM、埋め込み API を呼ばないこと。
- [x] `uv run pytest` で追加テストが通ること。

## 追加修正

- [x] Agentic RAG の既定検索閾値を既存タブと揃える。
  - ベクトル検索閾値: `0.25`
  - 全文検索閾値: `0`
- [x] 閾値差で既存タブではヒットする結果が Agentic RAG で落ちないよう unit テストを追加する。
- [x] Agentic RAG の実行トレースに各処理の所要時間を出力する。
- [x] 処理時間トレースの unit テストを追加する。

## 追加修正: 判定系モデルの個別指定

- [x] `.env_example` に Agentic RAG の質問分解、十分性判定、追加検索クエリー生成のデフォルトモデル設定を追加する。
- [x] `app/agentic_rag.py` で単一の `llm_text_generator` に加えて、質問分解、十分性判定、追加検索クエリー生成ごとの LLM 関数を注入できるようにする。
- [x] `app/ui/components.py` で Agentic RAG タブに判定系モデルの個別選択 UI を追加し、`.env` の値を初期値にする。
- [x] `app/ui/agentic_events.py` で UI から受け取ったモデルを処理別に使い分ける。
- [x] `multimodal_retriever.py` で Agentic RAG イベント登録の inputs 数を更新する。
- [x] `tests/test_agentic_rag.py` に処理別モデル呼び分けと `.env` デフォルト解決の unit テストを追加する。
- [x] `uv run pytest tests/test_agentic_rag.py` で回帰確認する。

## 追加修正: LLM入力 evidence 上限

- [x] 十分性判定と evidence 選別・並べ替えへ渡す evidence 上限を 20 件から 50 件に変更する。
- [x] evidence プロンプト上限を unit テストで確認する。
- [x] `uv run pytest tests/test_agentic_rag.py` で回帰確認する。

## 追加修正: トレース出力形式

- [x] 質問分解と追加検索クエリー生成のトレースを複数行表示にする。
- [x] 十分性判定と evidence 選別・並べ替えのトレースに LLM 入力 evidence 件数、画像件数、概算トークン数、省略件数を出力する。
- [x] トレース出力形式の unit テストを追加する。
- [x] `uv run pytest tests/test_agentic_rag.py` で回帰確認する。

## 追加修正: 判定系モデルの全モデル対応

- [x] 質問分解、十分性判定、追加検索クエリー生成の Dropdown で `model_settings.json` の全モデルを選択できるようにする。
- [x] 判定系モデル呼び出しを、非Visionモデルでも動くテキスト生成経路に切り替える。
- [x] 最終回答生成用 VLM Dropdown は Vision 対応モデル限定のまま維持する。
- [x] 全モデル選択肢とテキスト生成経路の unit テストを追加・更新する。
- [x] `uv run pytest tests/test_agentic_rag.py` で回帰確認する。

## 追加修正: 参照画像ギャラリー表示

- [x] Agentic RAG の参照画像ギャラリーを複数行表示にして、6件選別時にも見えるようにする。
- [x] イベント返却時に Gallery の columns / rows / height 設定を維持する。
- [x] トレースに selected evidence 件数とギャラリー表示画像件数を出力する。
- [x] 6件の参照画像がギャラリーへ渡る unit テストを追加する。
- [x] `uv run pytest tests/test_agentic_rag.py` で回帰確認する。
