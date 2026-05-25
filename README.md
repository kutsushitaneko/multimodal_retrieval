# 🐕マルチモーダル・レトリバー🐕
## 機能
- OCI Generative AI もしくは Cohere AI のマルチモーダル埋め込みモデルと Oracle Database AI Vector Search を使ったマルチモーダルセマンティック検索
    - テキストによる画像検索（クエリーテキスト⇒テキストベクトル⇒画像ベクトル⇒画像）
    - 画像による画像検索（クエリー画像⇒画像ベクトル⇒画像ベクトル⇒画像）
- Oracle Text を使った全文検索（クエリーテキスト⇒形態素解析⇒画像キャプションの全文検索⇒画像）
    - 検索対象は画像のキャプション
- RAG
    - 検索結果の先頭画像、または選択した1画像を元にした回答生成
    - `VLMによるフィルタリングと並べ替え` により、検索結果全体からVLMが回答に有用な画像を選別し、自然な順序へ並べ替えてから複数画像を元に回答生成
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

VLM の選択肢は `model_settings.json` で定義します。UI に表示されるのは `vision: true` が設定されたモデルのみです。OCI の VLM では、モデルごとの `default_region` が設定されている場合は初期リージョンとして使用されます。

## アプリケーションのファイル説明
- アプリ本体： multimodal_retriever.py
- 環境変数：.env （例：.env_example）
- MLLM の定義：model_settings.json
- 画像のバッチ登録スクリプト：batch_injestion.py
- データベース環境構築SQL： sql フォルダに配置
- app 配下の class は、docs/app_classes.md　に説明あり
- Python ライブラリの依存関係：requirements.txt
- キャプション生成用プロンプト： prompt フォルダ
- 回答生成用プロンプト： answer_prompt フォルダ
- アプリ起動用バッチスクリプト： run.sh

## データベースの準備
### データベースユーザーの作成
下記SQLスクリプトを sqlplus などで実行
- sql/create_user_user.sql
- sql/grant_to_user.sql
### データベース・テーブルの作成
- sql/create_tables.sql
### Oracle Text ストップリストの作成
- sql/create_stoplist.sql

## 環境設定

### Python バージョン
3.12.x でのみ動作確認済み
### 依存パッケージのインストール
```
uv pip install -r requirements.txt
```

### .envファイルの設定

アプリケーションの設定は`.env`ファイルで管理されます。.env_example を参考に .env ファイルを作成して以下の項目を設定してください

#### データベース設定
```
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_DSN=your_db_dsn
TNS_ADMIN=/path/to/tns_admin
```

#### OCI設定
```
OCI_CONFIG_PROFILE=default
OCI_REGION=us-ashburn-1
OCI_COMPARTMENT_ID=your_compartment_id
```

#### OCI 以外のサービスプロバイダーのクレデンシャル設定
```
ANTHROPIC_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_DEFAULT_REGION=us-west-2
```

現行の起動時チェックでは `COHERE_API_KEY` も必須環境変数として確認されます。Cohere AI の埋め込みモデルを使わない場合も、`.env_example` を参考に設定してください。

#### 埋め込みモデルの設定
```
EMBED_MODEL_PROVIDER=OCI
EMBED_MODEL_ID=cohere.embed-multilingual-image-v3.0
```

`EMBED_MODEL_PROVIDER` は `OCI` または `CohereAI` を指定します。Cohere AI を直接利用する場合は、例として以下のように設定します。

```
EMBED_MODEL_PROVIDER=CohereAI
EMBED_MODEL_ID=embed-v4.0
```

#### マルチモーダルLLM（VLM）のデフォルト設定
```
MLLM_MODEL_PROVIDER=OCI
MLLM_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct-fp8
```

`MLLM_MODEL_ID` は Gradio UI 起動時の VLM 初期選択と、バッチ登録時のキャプション生成に使用されます。`model_settings.json` の表示名、または `model_name` と一致する値を設定してください。UI では検索・回答生成用とイメージ管理用の VLM 設定をそれぞれ独立して変更できます。

#### MLLM専用リージョン（オプション）
#### イメージのバッチ登録で、MLLM_MODEL_PROVIDER=OCIでかつMLLMで異なるリージョンを使用したい場合のみ設定（Gradio UI では使用していない）
```
#OCI_REGION_OVERRIDE_FOR_MLLM=us-chicago-1
```

#### デバッグログ設定（オプション）
```
VERBOSE=False
```

`VERBOSE=True` にすると、一部の VLM 設定変更や回答生成処理でデバッグログを出力します。

#### アプリケーション実行環境設定

**基本設定**
```
# アプリケーション実行モード (local または remote)
APP_MODE=local
```

**ローカル実行モード (APP_MODE=local)**
本アプリを実行するホストとブラウザを起動するホストが同じ場合（手元のノートPCなど）
```
# ブラウザ共有機能（通常はfalse）
LOCAL_SHARE=false
# 自動でブラウザを開く（通常はtrue）
LOCAL_INBROWSER=true
```

**リモート実行モード (APP_MODE=remote)**
本アプリを実行するホストとブラウザを起動するホストが異なる場合（本アプリをOCI Compute で起動するなど）
```
# 外部共有機能（通常はtrue）
REMOTE_SHARE=true
# サーバーホスト名
REMOTE_SERVER_NAME=0.0.0.0
# ポート番号
REMOTE_SERVER_PORT=8899
```

##### アプリケーション実行環境設定例

**ローカル開発環境での実行:**
```
APP_MODE=local
LOCAL_SHARE=false
LOCAL_INBROWSER=true
```

**リモートサーバーでの実行（ポート8080）:**
```
APP_MODE=remote
REMOTE_SHARE=true
REMOTE_SERVER_NAME=0.0.0.0
REMOTE_SERVER_PORT=8080
```

**リモートサーバーでの実行（デフォルトポート8899）:**
```
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

### 検索と回答生成
検索タブでは、検索対象とクエリーの種類を切り替えて検索できます。

- `検索対象`
    - `画像ベクトル`: テキストまたは画像から画像ベクトルを検索
    - `キャプション（テキストベクトルと全文）`: キャプションのテキストベクトル検索と Oracle Text 全文検索を統合
- `クエリーの種類`
    - `テキスト`
    - `画像`
- `検索件数`: 1〜24件
- `高度な設定`: ベクトル検索の閾値、全文検索の閾値
- `クエリ詳細`: 実行されたクエリ、実行SQL、全文検索時の形態素解析結果
- `全件表示`: 登録済み画像をページング表示

回答生成では、検索結果を元に以下のモードを選択できます。

- `先頭画像あるいは選択した１つの画像`: 検索結果の先頭画像、またはギャラリーで選択した1画像を参照
- `VLMによるフィルタリングと並べ替え`: 検索結果全体から VLM が回答に有用な画像を選別・並べ替えて参照

回答生成用 VLM は、検索タブの `VLM設定（検索・回答生成用）` で、サービスプロバイダ、VLMモデル、Temperature、Max tokens、OCIリージョンを変更できます。回答生成プロンプトは `回答生成プロンプトの設定と編集` で保存・編集・削除できます。

## 回答生成モードの手動テスト

- `回答生成モード` の初期値が `VLMによるフィルタリングと並べ替え` であることを確認します。
- テキストクエリーでは `VLMによるフィルタリングと並べ替え` を選択でき、画像クエリーでは `先頭画像あるいは選択した１つの画像` に戻ることを確認します。
- 単一画像モードで、検索結果ギャラリーで選択した画像、または未選択時の先頭画像を元に回答が生成されることを確認します。
- Listwiseモードで、検索結果全体からVLMが選別した画像が `参照した画像` に順序どおり表示され、選別理由が `reason` に表示されることを確認します。
- `参照する情報の種類` の `すべて`、`キャプションのみ`、`画像のみ` を切り替え、回答生成がエラーなく完了することを確認します。

## 画像の登録
### バッチ登録
`images` フォルダーに画像ファイルを配置してバッチスクリプトで登録することができます。`images` フォルダーが存在しない場合は作成してください。対象拡張子は `jpg`、`jpeg`、`webp` です。既に同じファイル名で登録済みの画像はスキップされます。

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
アプリを OCI などで実行しているとブラウザへの画像転送に時間がかかります。画面表示が遅い場合は、DBが遅いのではなく画像転送である確率が高いです。その場合は、アップロードする画像のサイズを小さくしてください。このユーティリティは、images_original フォルダにある画像を一辺が最大1024ピクセルに縮小して images フォルダに配置します。