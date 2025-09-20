# 🐕マルチモーダル・レトリバー🐕 by cohere.embed-v4.0
## 機能
- OCI Generative AI もしくは Cohre AI の埋め込みモデル Cohere Embed 4 と Oracle Database AI Vector Search を使ったマルチモーダルセマンティック検索
    - テキストによる画像検索（クエリーテキスト⇒テキストベクトル⇒画像ベクトル⇒画像）
    - 画像による画像検索（クエリー画像⇒画像ベクトル⇒画像ベクトル⇒画像）
- Oracle Text を使った全文検索（クエリーテキスト⇒形態素解析⇒画像キャプションの全文検索⇒画像）
    - 検索対象は画像のキャプション
- RAG
    - セマンティック検索、もしくは、全文検索の検索結果の１画像を元にした回答生成（複数画像は未対応）

## デモ画像
#### マルチモーダルRAGのデモ（PowerPointスライドの自然言語検索と回答生成及び画像による類似画像の検索デモ）
[![マルチモーダルRAGのデモ（PowerPointスライドの自然言語検索と回答生成及び画像による類似画像の検索デモ）](https://img.youtube.com/vi/J6AwW_afEAc/0.jpg)](https://youtu.be/J6AwW_afEAc)
#### マルチモーダルRAGのデモ（ゴミ収集日の確認）
[![マルチモーダルRAGのデモ（ゴミ収集日の確認）](https://img.youtube.com/vi/roF3rKRjhpM/0.jpg)](https://youtu.be/roF3rKRjhpM)
#### マルチモーダルRAGのデモ（写真に対するRAG）
[![マルチモーダルRAGのデモ（写真に対するRAG）](https://img.youtube.com/vi/vExpf6bL9rQ/0.jpg)](https://youtu.be/vExpf6bL9rQ)

## 対応マルチモーダル埋め込みモデル
- OCI Generative AI Service の Cohere Embed 4
- Cohere AI の Cohere Embed 4

## 対応マルチモーダルLLM（VLM）プロバイダー
- OCI Generative AI Service
- OpenAI
- Anthropic
- Amazon Bedrock

## アプリケーションのファイル説明
- アプリ本体： multimodal_retriever.py
- 環境変数：.env （例：.env_example）
- MLLM の定義：model_settings.json
- 画像のバッチ登録スクリプト：batch_injestion.py
- データベース環境構築SQL： sql フォルダに配置
- app 配下の class は、docs/app_classes.md　に説明あり
- Python ライブライの依存関係：requirements.txt
- キャプション生成用プロンプト： prompt フォルダ
- アプリ起動用バッチスクリプト： run.sh

## データベースの準備
### データベースユーザーの作成
下記SQLスクリプトを sqlplus などで実行
- sql/create_user_user.sql
- sql/grant_to_user.sql
- sql/create_tables.sql
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
OCI_GENAI_MLLM_MODEL_ID=your_model_id
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

#### 埋め込みモデルの設定
```
EMBED_MODEL_PROVIDER=OCI or CohereAI
EMBED_MODEL_ID=embed-v4.0
```

#### バッチ投入時のマルチモーダルLLM（VLM）の設定（UIでは、画面上で選択）
```
MLLM_MODEL_PROVIDER=OCI
MLLM_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct-fp8
```

#### MLLM専用リージョン（オプション）
#### イメージのバッチ登録で、MLLM_MODEL_PROVIDER=OCIでかつMLLMで異なるリージョンを使用したい場合のみ設定（Gradio UI では使用していない）
```
#OCI_REGION_OVERRIDE_FOR_MLLM=us-chicago-1
```

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

アプリケーションは`.env`ファイルの設定に従って、ローカルまたはリモートモードで起動します。 

## 画像の登録
### バッチ登録
`images` フォルダーに画像ファイルを配置してバッチスクリプトで登録することができます。
```bash
batch_injestion.py
```
### UI で1枚づつ登録
アプリを起動して、「イメージ管理」タブの画像アップロードから登録します。

## その他
### util_compress_image.py
アプリを OCI などで実行しているとブラウザへの画像転送に時間がかかります。画面表示が遅い場合は、DBが遅いのではなく画像転送である確率が高いです。その場合は、アップロードする画像のサイズを小さくしてください。このユーティリティは、images_original フォルダにある画像を一辺が最大1024ピクセルに縮小して images フォルダに配置します。