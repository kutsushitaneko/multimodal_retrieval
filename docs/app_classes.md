## app ディレクトリの Python スクリプトとクラス一覧

このドキュメントは `app` 配下の主要な Python スクリプトと、それぞれに定義されているクラスの役割・説明をまとめたものです。Gradio UI、検索、埋め込み、DB、VLM、NLP など本アプリの中核サービスを俯瞰できます。

### app/ui/components.py
- **UIComponents**: UI コンポーネントを構築するヘルパークラス。
  - create_search_section: 検索対象・検索方法、クエリ入力、アップロード画像エリア、各種ボタンを含む検索セクションを構築。
  - create_search_vlm_settings: 検索タブ専用の VLM 設定（モデル、プロバイダー、温度、max tokens、OCI リージョン）を生成し、VLMService と連携。
  - create_upload_edit_section: 画像アップロード、ファイル名入力、キャプション生成/編集/登録の UI、プロンプト編集と削除、VLM 設定、削除アコーディオンを構築。
  - create_results_section: ベクトル検索・全文検索の結果ギャラリーを構築。
  - create_pagination_section: 前後ページボタン・ページ情報の UI を構築。
  - create_detail_section: 検索結果のファイル名・スコア・キャプション表示を構築。
  - create_query_detail_section: 実行クエリ、SQL、形態素解析結果の表示エリアを構築。
  - create_advanced_settings_section: 閾値や表示件数など高度な設定 UI を構築。
  - create_answer_generation_section: 参照画像・参照タイプ・質問入力・回答表示の UI を構築。
  - create_answer_prompt_settings_section: 回答生成プロンプトの選択・編集・保存・削除 UI を構築。

### app/ui/events.py
- **UIEvents**: Gradio のイベントハンドリングを統括するクラス（詳細はファイル参照）。UIComponents で生成したコンポーネント群に対して、検索、画像アップロード、キャプション生成/更新、プロンプト保存/削除、VLM 設定変更などのイベントを接続する責務を担う。

### app/search_service.py
- **SearchService**: 検索ロジックの集約クラス。
  - search_by_caption: テキストクエリでキャプションを検索（ベクトル検索 or 全文検索）。形態素解析結果も返却。
  - search_by_image_text: テキストから画像ベクトル領域を検索。
  - search_by_image_embedding: 画像アップロードから画像ベクトル検索。
  - hybrid_search: ベクトル検索と全文検索の融合結果を統合。
  - search_images: UI からの入力（検索対象/方法、画像）に応じた統合検索エントリポイント。
  - load_recent_images: アプリ起動時の最近画像読み込み。

### app/search_query_generator.py
- **SearchQueryGenerator**: 全文検索向けのクエリー生成器。
  - ginza/ja_ginza による形態素解析を利用し、停止語・助数詞・色形容詞の名詞化、URL/ID/パス/メール/バージョン等の特殊パターンの中カッコ完全一致化などを施した AND クエリーを生成。
  - get_morphological_analysis_details: 生成過程を Markdown テーブルで可視化。

### app/global_nlp_service.py
- **GlobalNLPService**: spaCy ja_ginza のグローバル・シングルトンサービス。
  - スレッドセーフに単一インスタンスを共有し、初期化コストとメモリ使用量を最小化。
  - get_global_nlp_service 関数でアプリ全体から取得可能。

### app/nlp_service.py
- **NLPService**: タブ毎に独立構成が可能な NLP + 画像キャプション生成支援サービス。
  - get_nlp: スレッドセーフに ja_ginza を遅延ロード。
  - generate_caption_with_vlm: VLMService を介して Anthropic / OCI / OpenAI / Bedrock / Vertex を切替し画像キャプションを生成（画像の Data URL 化と形式変換を内包）。

### app/vlm_service.py
- **VLMService**: Vision 対応モデル設定の参照と UI 更新ヘルパー。
  - model_settings.json を `config/model_settings.json` から読み込み、Vision 対応モデルのみを提供。
  - サービスプロバイダー（OCI/AWS/Anthropic/OpenAI/Cohere）での絞り込み、モデル変更時の関連 UI（max tokens/temperature/OCI リージョン）更新用オブジェクトを返すユーティリティを提供。

### app/vlm_service_factory.py
- **VLMServiceFactory**: タブ毎に独立した VLMService を生成するファクトリー。
  - create_search_vlm_service / create_upload_vlm_service / create_answer_vlm_service: 用途に応じた初期設定（temperature, max_tokens, oci_region）を付与。

### app/config.py
- **Config**: 環境変数・OCI/DB/埋め込み・起動設定の単一点管理。
  - _validate_env_vars: 必須環境変数の検証。
  - get_db_pool / close_db_pool: Oracle DB コネクションプール（`pool_alias` 付き単一プール、`ping_interval` による死活管理）。
  - get_cohere_client / get_oci_generative_ai_client: API クライアントの初期化。
  - get_launch_config: Gradio の起動パラメータ（remote/local）を返却。

### app/database_service.py
- **DatabaseService**: 画像とキャプションの保存・検索・更新・削除を担う DB アクセス層。
  - `_execute_with_retry`: 接続エラー時に `pool.drop(conn)` 後リトライ（プール全体の再作成は行わない）。
  - get_image_caption: OCI Generative AI で画像キャプションを生成（DataURL 化、プロンプト、レスポンス抽出、UTF-8 バイト長安全切詰め、整形を内包）。
  - insert_image_to_db / update_image_caption / delete_image: 画像の登録・更新・削除。
  - search_by_caption_vector / search_by_fulltext / search_by_image_vector / get_recent_images: ベクトル/Oracle Text/画像ベクトル/最近一覧の検索 API。
  - 内部で BLOB→PIL 変換、距離や検索モード付の結果整形を実施。

### app/embedding_service.py
- **EmbeddingService**: CohereAI / OCI のテキスト・画像埋め込み生成。
  - get_text_embedding / get_image_embedding: プロバイダーに応じた実装を選択し、必要に応じて画像の JPEG 変換→DataURL 化を実施。

### app/prompt_service.py
- **PromptService**: `prompt/caption/` と `prompt/answer/` のテンプレート列挙・読込・保存・削除を行う。
  - category 引数（`caption` / `answer`）でサブフォルダーを切り替え。
  - render_answer_prompt: 回答生成用プレースホルダ `{query_text}` / `{documents}` の置換。

### app/prompt_loader.py
- **load_prompt**: Agent / Retrieval 用の読み取り専用テンプレートを `prompt/agent/`、`prompt/retrieval/`、`prompt/snippets/` から読み込み、プレースホルダを置換。

### app/paths.py
- プロンプト・設定ファイルのパス定数（`PROMPT_*`, `CONFIG_*`）を集約。

### app/cleanup_service.py
- **CleanupService**: `temp/gradio` 配下の Gradio 一時ファイルを定期クリーンアップ。
  - バックグラウンドスレッドで一定間隔実行、最大保持時間を超えた項目を削除。


