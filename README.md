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

#### Cohere API設定
```
COHERE_API_KEY=your_cohere_api_key
```

#### OCI設定
```
OCI_CONFIG_PROFILE=default
OCI_REGION=us-ashburn-1
OCI_COMPARTMENT_ID=your_compartment_id
OCI_GENAI_MLLM_MODEL_ID=your_model_id
```

#### アプリケーション実行環境設定

**基本設定**
```
# アプリケーション実行モード (local または remote)
APP_MODE=local
```

**ローカル実行モード (APP_MODE=local)**
```
# ブラウザ共有機能（通常はfalse）
LOCAL_SHARE=false
# 自動でブラウザを開く（通常はtrue）
LOCAL_INBROWSER=true
```

**リモート実行モード (APP_MODE=remote)**
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

## 実行方法

```bash
uv run main.py
```

アプリケーションは`.env`ファイルの設定に従って、ローカルまたはリモートモードで起動します。 