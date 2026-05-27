# マルチモーダル・レトリバー
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kutsushitaneko/multimodal_retrieval)

## 機能
- OCI Generative AI もしくは Cohere AI のマルチモーダル埋め込みモデルと Oracle Database AI Vector Search を使ったマルチモーダル検索
    - テキストによる画像ベクトル検索
    - 画像による類似画像ベクトル検索
    - キャプションのテキストベクトル検索
- Oracle Text を使ったキャプション全文検索
    - GiNZA / spaCy による日本語クエリー生成
    - エラーコード、URL、IPアドレス、論文ID、ファイル名、API名などのルールベース固有表現抽出
    - 任意の LLM による製品名、サービス名、組織名などの固有表現抽出
    - 固有表現を Oracle Text の中カッコ完全一致 OR 検索へ変換
    - 実行クエリー、SQL、形態素解析・固有表現抽出結果の表示と再実行
- Agentic RAG
    - `Workflow Agentic RAG`: 質問分解、複数検索、十分性判定、追加検索、evidence 選別・並べ替え、回答生成を固定ワークフローで自動実行
    - `ReAct Agentic RAG`: LLM が Thought / Action / Observation を繰り返し、必要な検索 Tool を選びながら回答生成
    - キャプションベクトル検索、キャプション全文検索、画像ベクトル検索を統合して evidence を重複排除
    - 処理ステップ、所要時間、LLM 入力規模を実行トレースとして逐次表示
- 通常 RAG
    - 検索結果の先頭画像、または選択した1画像を元にした回答生成
    - `VLMによるフィルタリングと並べ替え` により、検索結果全体から VLM が回答に有用な画像を選別し、自然な順序へ並べ替えてから複数画像を元に回答生成
    - 回答生成時に参照する情報は `すべて`、`キャプションのみ`、`画像のみ` から選択可能
- 画像管理
    - 画像アップロード、キャプション生成・再生成、キャプション編集、データベース登録・更新、削除
    - キャプション生成プロンプトと回答生成プロンプトのテンプレート編集

## デモ画像
#### マルチモーダルRAGのデモ（PowerPointスライドの自然言語検索と回答生成及び画像による類似画像の検索デモ）
[![マルチモーダルRAGのデモ（PowerPointスライドの自然言語検索と回答生成及び画像による類似画像の検索デモ）](https://img.youtube.com/vi/J6AwW_afEAc/0.jpg)](https://youtu.be/J6AwW_afEAc)
#### マルチモーダルRAGのデモ（ゴミ収集日の確認）
[![マルチモーダルRAGのデモ（ゴミ収集日の確認）](https://img.youtube.com/vi/roF3rKRjhpM/0.jpg)](https://youtu.be/roF3rKRjhpM)
#### マルチモーダルRAGのデモ（写真に対するRAG）
[![マルチモーダルRAGのデモ（写真に対するRAG）](https://img.youtube.com/vi/vExpf6bL9rQ/0.jpg)](https://youtu.be/vExpf6bL9rQ)

## 対応マルチモーダル埋め込みモデル
- OCI Generative AI Service のマルチモーダル埋め込みモデル
    - 例: `cohere.embed-multilingual-image-v3.0`
- Cohere AI のマルチモーダル埋め込みモデル
    - 例: `embed-v4.0`

## 対応マルチモーダルLLM（VLM）プロバイダー
- OCI Generative AI Service
- OpenAI
- Anthropic
- AWS / Amazon Bedrock

VLM とテキスト生成モデルの選択肢は `model_settings.json` で定義します。最終回答生成やキャプション生成で画像を渡す VLM Dropdown には、`vision: true` が設定されたモデルのみが表示されます。Workflow Agentic RAG の判定系モデルと ReAct Controller モデルは画像を渡さないため、`model_settings.json` の全モデルを選択できます。OCI のモデルでは、モデルごとの `default_region` が設定されている場合は初期リージョンとして使用されます。

## アプリケーションのファイル説明
- アプリ本体: `multimodal_retriever.py`
- 環境変数: `.env`（例: `.env_example`）
- VLM / テキスト生成モデル定義: `model_settings.json`
- 画像のバッチ登録スクリプト: `batch_injestion.py`
- データベース環境構築 SQL: `sql` フォルダ
- Python ライブラリの依存関係: `requirements.txt`
- キャプション生成用プロンプト: `prompt` フォルダ
- 回答生成用プロンプト: `answer_prompt` フォルダ
- 質問例: `question_examples.json`
- アプリ起動用バッチスクリプト: `run.sh`

## データベースの準備
### データベースユーザーの作成
下記 SQL スクリプトを sqlplus などで実行します。

- `sql/create_user_user.sql`
- `sql/grant_to_user.sql`

### データベース・テーブルの作成
- `sql/create_tables.sql`

### Oracle Text ストップリストの作成
- `sql/create_stoplist.sql`

## 環境設定

### Python バージョン
3.12.x でのみ動作確認済み

### 依存パッケージのインストール

```bash
uv pip install -r requirements.txt
```

### .envファイルの設定

アプリケーションの設定は `.env` ファイルで管理されます。`.env_example` を参考に `.env` ファイルを作成して以下の項目を設定してください。

#### データベース設定

```bash
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_DSN=your_db_dsn
TNS_ADMIN=/path/to/tns_admin
```

#### OCI設定

```bash
OCI_CONFIG_PROFILE=default
OCI_REGION=us-ashburn-1
OCI_COMPARTMENT_ID=your_compartment_id
```

#### OCI 以外のサービスプロバイダーのクレデンシャル設定

```bash
ANTHROPIC_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_DEFAULT_REGION=us-west-2
```

現行の起動時チェックでは `COHERE_API_KEY` も必須環境変数として確認されます。Cohere AI の埋め込みモデルを使わない場合も、`.env_example` を参考に設定してください。

#### 埋め込みモデルの設定

```bash
EMBED_MODEL_PROVIDER=OCI
EMBED_MODEL_ID=cohere.embed-multilingual-image-v3.0
```

`EMBED_MODEL_PROVIDER` は `OCI` または `CohereAI` を指定します。Cohere AI を直接利用する場合は、例として以下のように設定します。

```bash
EMBED_MODEL_PROVIDER=CohereAI
EMBED_MODEL_ID=embed-v4.0
```

#### マルチモーダルLLM（VLM）のデフォルト設定

```bash
MLLM_MODEL_PROVIDER=OCI
MLLM_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct-fp8
```

`MLLM_MODEL_ID` は Gradio UI 起動時の回答生成・キャプション生成 VLM の初期選択と、バッチ登録時のキャプション生成に使用されます。`model_settings.json` の表示名、または `model_name` と一致する値を設定してください。UI では Workflow Agentic RAG、ReAct Agentic RAG、検索・回答生成、イメージ管理の VLM 設定をそれぞれ独立して変更できます。

#### Workflow Agentic RAG 判定系モデルのデフォルト設定

```bash
WORKFLOW_AGENTIC_DECOMPOSE_MODEL_ID=google.gemini-2.5-flash-lite
WORKFLOW_AGENTIC_SUFFICIENCY_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct-fp8
WORKFLOW_AGENTIC_FOLLOWUP_QUERY_MODEL_ID=google.gemini-2.5-flash-lite
```

`WORKFLOW_AGENTIC_DECOMPOSE_MODEL_ID` は質問分解、`WORKFLOW_AGENTIC_SUFFICIENCY_MODEL_ID` は十分性判定、`WORKFLOW_AGENTIC_FOLLOWUP_QUERY_MODEL_ID` は追加検索クエリー生成の初期モデルです。`model_settings.json` の表示名、または `model_name` と一致する値を設定してください。後方互換として `AGENTIC_DECOMPOSE_MODEL_ID`、`AGENTIC_SUFFICIENCY_MODEL_ID`、`AGENTIC_FOLLOWUP_QUERY_MODEL_ID` も参照されます。

#### ReAct Agentic RAG のデフォルト設定

```bash
REACT_AGENTIC_VLM_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct-fp8
REACT_AGENTIC_CONTROLLER_MODEL_ID=google.gemini-2.5-flash-lite
```

`REACT_AGENTIC_VLM_MODEL_ID` は ReAct Agentic RAG タブの回答生成 VLM 初期値です。未設定の場合は `MLLM_MODEL_ID` 由来の通常 VLM 初期値を使用します。`REACT_AGENTIC_CONTROLLER_MODEL_ID` は Thought / Action / Observation の次の Action を決める Controller 初期値です。どちらも `model_settings.json` の表示名、または `model_name` と一致する値を設定してください。

#### 全文検索の LLM 固有表現抽出（オプション）

```bash
FULLTEXT_ENTITY_EXTRACTION_ENABLED=false
FULLTEXT_ENTITY_EXTRACTION_MODEL_ID=google.gemini-2.5-flash-lite
```

`FULLTEXT_ENTITY_EXTRACTION_ENABLED=true` にすると、ルールベース抽出に加えて LLM が製品名、サービス名、組織名、人名、地名などの固有表現を抽出します。抽出された語は Oracle Text の中カッコ完全一致 OR 検索に変換されます。ルールベース抽出は常に有効で、URL、論文ID、IPアドレス、エラーコード、バージョン、ファイル名、API名、識別子などを補完します。

#### MLLM専用リージョン（オプション）
#### イメージのバッチ登録で、MLLM_MODEL_PROVIDER=OCIでかつMLLMで異なるリージョンを使用したい場合のみ設定（Gradio UI では使用していない）

```bash
#OCI_REGION_OVERRIDE_FOR_MLLM=us-chicago-1
```

#### デバッグログ設定（オプション）

```bash
VERBOSE=False
```

`VERBOSE=True` にすると、一部の VLM 設定変更や回答生成処理でデバッグログを出力します。

#### アプリケーション実行環境設定

**基本設定**

```bash
# アプリケーション実行モード (local または remote)
APP_MODE=local
```

**ローカル実行モード (APP_MODE=local)**
本アプリを実行するホストとブラウザを起動するホストが同じ場合（手元のノートPCなど）

```bash
# ブラウザ共有機能（通常はfalse）
LOCAL_SHARE=false
# 自動でブラウザを開く（通常はtrue）
LOCAL_INBROWSER=true
```

**リモート実行モード (APP_MODE=remote)**
本アプリを実行するホストとブラウザを起動するホストが異なる場合（本アプリをOCI Compute で起動するなど）

```bash
# 外部共有機能（通常はtrue）
REMOTE_SHARE=true
# サーバーホスト名
REMOTE_SERVER_NAME=0.0.0.0
# ポート番号
REMOTE_SERVER_PORT=8899
```

##### アプリケーション実行環境設定例

**ローカル開発環境での実行:**

```bash
APP_MODE=local
LOCAL_SHARE=false
LOCAL_INBROWSER=true
```

**リモートサーバーでの実行（ポート8080）:**

```bash
APP_MODE=remote
REMOTE_SHARE=true
REMOTE_SERVER_NAME=0.0.0.0
REMOTE_SERVER_PORT=8080
```

**リモートサーバーでの実行（デフォルトポート8899）:**

```bash
APP_MODE=remote
REMOTE_SHARE=true
REMOTE_SERVER_NAME=0.0.0.0
REMOTE_SERVER_PORT=8899
```

## UI実行方法

```bash
uv run multimodal_retriever.py
```

or

```bash
./run.sh
```

アプリケーションは`.env`ファイルの設定に従って、ローカルまたはリモートモードで起動します。Gradio の一時ディレクトリはカレントディレクトリ配下の `temp/gradio` に作成されます。

`run.sh` は `uv run multimodal_retriever.py` をバックグラウンドで起動し、標準出力と標準エラーを `output.log` に出力します。

## UIの主な使い方

アプリのタブは、`Workflow Agentic RAG`、`ReAct Agentic RAG`、`検索と回答生成`、`イメージ管理` の順に並びます。

### Workflow Agentic RAG

`Workflow Agentic RAG` タブは、自然文の質問から回答生成までを固定ワークフローで実行するタブです。質問と任意の入力画像を指定し、`参照する情報の種類`、`検索件数`、`再検索回数上限` を設定して `Workflow Agentic RAG 実行` を押します。

主な処理の流れは以下です。

- 質問を最大5件のサブクエリーへ分解します。
- 各サブクエリーに対して、キャプションベクトル検索、キャプション全文検索、テキストによる画像ベクトル検索を実行します。
- 入力画像がある場合は、画像による類似画像ベクトル検索も実行します。
- 検索結果を evidence として重複排除し、十分性判定を行います。
- evidence が不足している場合は、追加検索クエリーを生成し、上限回数まで再検索します。
- 回答に使う evidence を選別・並べ替え、選択された画像とキャプションを元に回答を生成します。

質問なしで画像だけを入力した場合は、画像ベクトル検索のみを実行し、類似画像を `参照した画像` に表示します。

`VLM設定（Workflow Agentic RAG用）` では、回答生成 VLM と、質問分解モデル、十分性判定モデル、追加検索クエリー生成モデルを個別に選択できます。`実行トレース` には質問分解、各検索、十分性判定、追加検索、evidence 選別・並べ替え、回答生成の進行状況と所要時間が逐次表示されます。

### ReAct Agentic RAG

`ReAct Agentic RAG` タブは、Controller モデルが Thought / Action / Observation を繰り返しながら、必要な検索 Tool を選択して回答生成まで進めるタブです。質問と任意の入力画像を指定し、`参照する情報の種類`、`検索件数`、`最大ステップ数` を設定して `ReAct Agentic RAG 実行` を押します。

Controller が使用できる主な Action は以下です。

- `multi_search`: 複数のクエリー候補と複数の検索 Tool を組み合わせて一括検索します。
- `caption_vector_search`: キャプションのテキストベクトル検索を実行します。
- `caption_fulltext_search`: キャプションの全文検索を実行します。固有表現や完全一致が重要な質問を補完します。
- `image_vector_text_search`: テキストクエリーから画像ベクトル検索を実行します。
- `image_vector_image_search`: 入力画像から類似画像ベクトル検索を実行します。
- `select_evidence`: 回答に使う evidence を選別します。
- `generate_final_answer`: 選別済み evidence で最終回答を生成します。

ReAct Controller は、初回検索では原則 `multi_search` を使い、キャプションベクトル検索、画像ベクトル検索、必要に応じて全文検索を組み合わせます。`VLM設定（ReAct Agentic RAG用）` では、回答生成 VLM と ReAct Controller モデルを個別に選択できます。`実行トレース` には Controller 応答、Action、Observation、検索結果件数、最終回答生成までの流れが逐次表示されます。

### 検索と回答生成
`検索と回答生成` タブでは、手動で検索条件を指定し、検索結果を確認してから回答生成できます。アプリ起動時は `検索件数` に応じて最近アップロードされた画像を表示します。

`検索設定` では、検索対象、クエリーの種類、検索件数を指定できます。

- `検索対象`
    - `画像ベクトル`: テキストまたは画像から画像ベクトルを検索
    - `キャプション（テキストベクトルと全文）`: キャプションのテキストベクトル検索と Oracle Text 全文検索を統合
- `クエリーの種類`
    - `テキスト`: 検索クエリ欄に入力した自然言語で検索
    - `画像`: アップロードした画像で類似画像を検索
- `検索件数`: 1〜24件の範囲で指定できます。初期値は8件です。通常検索、`検索と回答生成`、`全件表示` の表示件数に使われます。

`高度な設定` では、ベクトル検索閾値と全文検索閾値を指定できます。Agentic RAG 側の既定値は、ベクトル検索閾値 `0.25`、全文検索閾値 `0` です。

`検索結果の選別・並べ替え設定` では、回答生成時に検索結果をどう参照するかを指定します。

- `回答生成モード` の初期値は `VLMによるフィルタリングと並べ替え` です。
- `先頭画像あるいは選択した１つの画像`: 検索結果の先頭画像、またはギャラリーで選択した1画像を参照します。
- `VLMによるフィルタリングと並べ替え`: 検索結果全体から VLM が回答に有用な画像を選別し、回答生成に自然な順序へ並べ替えて参照します。
- 画像クエリー時は `先頭画像あるいは選択した１つの画像` に戻ります。

検索操作には以下のボタンを使います。

- `検索`: 検索だけを実行します。
- `検索と回答生成`: 検索後に、現在の回答生成モードで回答生成まで実行します。
- `クリア`: 検索入力、検索結果、回答表示などをクリアします。
- `全件表示`: 登録済み画像をアップロード日時が新しい順に表示します。

検索入力では、テキストクエリ欄と画像アップロード欄がクエリーの種類に応じて切り替わります。テキストクエリには `質問の例` からサンプルを選択できます。質問例は `question_examples.json` で管理します。

検索結果と詳細情報は以下の領域に表示されます。

- `検索結果`: 画像ベクトル系の結果ギャラリーと、全文検索結果ギャラリーを表示します。`キャプション（テキストベクトルと全文）` では、ベクトル検索と全文検索の結果をそれぞれ別ギャラリーに表示します。
- `ページング`: 検索結果を8件単位で `前へ` / `次へ` により切り替えます。
- `画像詳細`: 選択画像のファイル名、コサイン類似度または全文検索スコア、キャプションを表示します。
- `クエリ詳細`: 実行されたクエリ、実行SQL、全文検索時の形態素解析・固有表現抽出結果を表示します。全文検索クエリは `このクエリを実行` で再実行できます。

回答生成では、`参照する情報の種類` を `すべて`、`キャプションのみ`、`画像のみ` から選択できます。Listwiseモードでは、参照した画像が `参照した画像` に表示され、VLM の選別理由が `reason` に表示されます。

回答生成用 VLM は、検索タブの `VLM設定（検索・回答生成用）` で、サービスプロバイダ、VLMモデル、Temperature、Max tokens、OCIリージョンを変更できます。回答生成プロンプトは `回答生成プロンプトの設定と編集` で保存・編集・削除できます。

### 全文検索の動作

全文検索は画像キャプションを対象にします。通常の自然文では GiNZA / spaCy による形態素解析と停止語除去により Oracle Text クエリーを生成します。URL、論文ID、IPアドレス、エラーコードなどの固有表現が含まれる場合は、形態素解析より優先して `{ORA-00923} OR {https://example.com}` のような中カッコ完全一致 OR 検索を生成します。

`FULLTEXT_ENTITY_EXTRACTION_ENABLED=true` の場合は、LLM による固有表現抽出も追加されます。これにより、製品名やサービス名など、ルールだけでは拾いにくい語を全文検索で補完できます。

## 画像の登録
### バッチ登録
`images` フォルダーに画像ファイルを配置してバッチスクリプトで登録できます。`images` フォルダーが存在しない場合は作成してください。対象拡張子は `jpg`、`jpeg`、`webp` です。既に同じファイル名で登録済みの画像はスキップされます。

```bash
uv run batch_injestion.py
```

### UI で1枚ずつ登録・編集
アプリを起動して、「イメージ管理」タブから画像を登録・編集します。

- 画像アップロードとファイル名入力
- VLM によるキャプション生成・再生成
- 生成キャプションの編集
- データベースへの登録・更新
- 既存画像の検索
- 検索結果からイメージ管理タブへのコピー
- 既存画像の削除
- キャプション生成プロンプトの保存・編集・削除

イメージ管理タブの `VLM設定` は、検索タブの VLM 設定とは独立しています。キャプション生成用に、サービスプロバイダ、VLMモデル、Temperature、Max tokens、OCIリージョンを変更できます。

## その他
### util_compress_image.py
アプリを OCI などで実行しているとブラウザへの画像転送に時間がかかります。画面表示が遅い場合は、DB が遅いのではなく画像転送である確率が高いです。その場合は、アップロードする画像のサイズを小さくしてください。このユーティリティは、`images_original` フォルダにある画像を一辺が最大1024ピクセルに縮小して `images` フォルダに配置します。