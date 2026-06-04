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

## Agentic RAG サブシステム

Workflow / ReAct の 2 つの Agentic RAG は、検索ロジック（`SearchService`）の上に構築され、`EvidencePool` による evidence 集約・重複排除を共有します。判定系・制御系には画像を渡さないテキスト LLM を、最終回答生成にだけ VLM を使う構成です。

### app/agentic_rag_common.py（両 Agentic 共通の土台）
- **Evidence** (dataclass): 1 検索ヒットを表す。`id`（安定キー）、`image_id`、`file_name`、`caption`、`search_mode`、`source_query`、`source_tool`、`distance`、`image`(PIL)。
- **QuestionAspect** (dataclass): 質問観点。`kind` は `material`（コーパスから取得すべき事実）/ `operation`（要約・翻訳など回答モデルが材料から導出する処理）、`depends_on`、`satisfied`、`evidence_ids`。
- **DecomposeResult / SufficiencyDecision** (dataclass): 質問分解結果（subqueries + aspects）と十分性判定結果（status / reason / missing_aspects / aspects）。
- **EvidencePool**: 検索方式・サブクエリー・再検索を跨いで結果を集約。`add_many` が `Evidence.id`（`image_id`→`file_name`→生成キーの優先で決定）で重複排除するため、同一画像は 1 件に統合される。
- **format_documents**: 選別済み evidence を回答プロンプト用の番号付きテキストへ整形。

### app/workflow_aspects.py（material / operation 観点の解析）
- **parse_aspect_items / _normalize_kind**: 分解 LLM が返す aspects を `QuestionAspect` 群へ正規化（不明な kind は `material` 既定）。
- **merge_dependent_aspects**: マルチホップの未確定 material（`dependent_aspects`）を観点として登録しつつ、初回 subqueries には含めない。
- **material_aspects / operation_aspects / material_missing_labels**: 観点の絞り込みと、material かつ未充足のラベル抽出（追加検索の対象は material 不足のみ）。
- **reconcile_status / apply_sufficiency_normalization**: operation 観点は依存 material が揃えば充足可とみなし、十分性ステータスを決定論的に補正。
- **format_aspects_for_prompt / format_operation_aspects_for_followup / aspect_counts_summary**: プロンプト・進捗表示向けの整形。

### app/workflow_agentic_rag.py（固定ワークフロー）
- **WorkflowAgenticRAGResult** (dataclass): 回答・選別 evidence・全 evidence・trace 等の出力。
- **WorkflowAgenticRAGPipeline**: 質問分解→初回検索→十分性判定→追加検索→evidence 選別→回答生成を固定順で実行。
  - `run` / `run_stream`: 一括／逐次（進捗 yield）実行のエントリポイント。
  - `decompose_question`（`_decompose_question_with_llm` / `_decompose_question_with_rules`）: 最大 5 件のサブクエリー＋観点へ分解。
  - `_run_searches` → `_search_caption_vector` / `_search_caption_fulltext` / `_search_image_by_text`: 各検索後に `EvidencePool.add_many` で都度マージ。全文は `agentic_exact` モード。
  - `_run_image_search`: アップロード画像による類似画像検索。
  - `judge_evidence_sufficiency` / `generate_followup_queries`: material 不足のみを対象に再判定・追加クエリー生成（`_dedupe_queries` で重複排除、中間値＋不足属性の chained query 優先）。
  - `filter_and_order_evidence`: テキスト LLM（既定は十分性判定モデル、キャプションのみ参照）で回答用 evidence を選別・並べ替え。
  - **fail closed**: `sufficient` 以外、再検索上限到達、追加クエリー生成失敗、選別失敗のいずれでも回答生成を行わない。

### app/react_agentic_rag.py（ReAct ループ）
- **ReactStep / ReactAgenticRAGResult** (dataclass): 1 ステップの thought/action/observation と最終出力。
- **ReactToolRegistry**: Controller が選んだ Action を実行。`ALLOWED_ACTIONS` = `plan_and_execute_search`（推奨）/ `multi_search`（レガシー・直積）/ `caption_vector_search` / `caption_fulltext_search` / `image_vector_text_search` / `image_vector_image_search` / `select_evidence` / `generate_final_answer`。
  - `iter_execute` / `_execute_search`: 同一 `(tool, query)` の再検索を `_search_key` でスキップ。各検索後に `EvidencePool.add_many`。
  - `_iter_execute_plan_and_search`: `SearchPlanner` の計画どおりにツール別クエリーを実行。
  - `_select_evidence`: 取得済み evidence から回答候補を選別（`参照ドキュメント数` 上限）。
- **ReactAgenticRAGPipeline**: ループ制御。
  - `_call_controller` / `_validate_controller_response` / `_build_controller_prompt`: Controller LLM 呼び出しと応答検証。
  - 自動打切: `max_stale_steps`（既定 2）回連続で新規 evidence が増えなければ情報不足終了。
  - **Finalize Verifier Gate**: `_run_finalize_verifier` / `_find_unsearched_finalize_leads` が `answerable: false` 時に evidence キャプションから未検索 lead を抽出し、残っていれば確定保留として再検索を強制。

### app/search_planner.py
- **SearchTask / SearchPlan** (dataclass) と **SearchPlanner**: `plan_and_execute_search` 用に、質問分解・ツール選択・ツール別クエリー（`exact_terms` 含む）を専用プロンプト（`prompt/agent/react/search_plan.txt`）で計画。`config/entity_patterns.json` の正規表現検出語を**参考ヒント**として渡す（推移性は LLM に委譲）。

### app/agentic_search_strategy.py
- ReAct 初回検索向けの軽量ルール（LLM 呼び出しなし）。`classify_question_strategy` は `identifier` / `none` のみ判定し、識別子検出時だけ Controller プロンプトへ弱いヒントを注入。`fulltext_natural_language_warning` / `fulltext_mixed_query_warning` / `format_lead_tool_recommendation` で全文 vs ベクトルの使い分けをソフトガード。

### app/entity_patterns.py
- **extract_entities_from_text**: `config/entity_patterns.json` の正規表現で URL・論文ID・IPアドレス・エラーコード・バージョン・ファイル名・API名などのルールベース固有表現を抽出。`format_entities_for_planner_prompt` で Planner プロンプト用に整形。

### app/fulltext_entity_extractor.py
- **FulltextEntityExtractor**: `FULLTEXT_ENTITY_EXTRACTION_ENABLED=true` のとき、LLM で製品名・サービス名・組織名などの固有表現を抽出（`normalize_entities` で正規化・包含除去・件数上限）。ルールベース抽出を補完する。

### app/ui/workflow_agentic_events.py
- **WorkflowAgenticRAGEvents**: Workflow Agentic RAG タブのイベント統括。`run_workflow_agentic_rag` がパイプラインを `run_stream` で駆動し、`_generate_answer_with_vlm` が最終回答 VLM 呼び出し（アップロード画像を 1 枚目、参照画像を 2 枚目以降）を担う。VLM 設定／モデル設定の変更イベント、参照画像ギャラリー詳細表示も登録。

### app/ui/react_agentic_events.py
- **ReactAgenticRAGEvents**（`WorkflowAgenticRAGEvents` を継承）: ReAct Agentic RAG タブのイベント統括。回答生成 VLM・Controller・Search Planner の 3 系統のモデル設定を登録し、`run_react_agentic_rag` がループを駆動。回答生成・参照画像表示は親クラスのロジックを再利用。


