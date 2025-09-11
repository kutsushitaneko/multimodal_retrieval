# マルチモーダル画像検索アプリケーション

## 環境設定

### .envファイルの設定

アプリケーションの設定は`.env`ファイルで管理されます。以下の項目を設定してください：

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

# 埋め込みモデルの設定
EMBED_MODEL_PROVIDER=OCI or CohereAI
EMBED_MODEL_ID=embed-v4.0

# バッチ投入時のマルチモーダルLLM（VLM）の設定（UIでは、画面上で選択）
MLLM_MODEL_PROVIDER=OCI
MLLM_MODEL_ID=meta.llama-4-maverick-17b-128e-instruct-fp8

# MLLM専用リージョン（オプション）
# イメージのバッチ登録で、MLLM_MODEL_PROVIDER=OCIでかつMLLMで異なるリージョンを使用したい場合のみ設定（Gradio UI では使用していない）
#OCI_REGION_OVERRIDE_FOR_MLLM=us-chicago-1

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

### 使用例

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