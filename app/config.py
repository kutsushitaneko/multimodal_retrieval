import os
import oci
import cohere
import oracledb
from dotenv import load_dotenv, find_dotenv

POOL_ALIAS = "multimodal_retriever"
DEFAULT_POOL_MIN = 2
DEFAULT_POOL_MAX = 10
DEFAULT_POOL_INCREMENT = 1
DEFAULT_POOL_TIMEOUT = 60
DEFAULT_PING_INTERVAL = 30
DEFAULT_PING_TIMEOUT = 5000


class Config:
    def __init__(self):
        # 環境変数を読み込む
        load_dotenv(find_dotenv())
        self._validate_env_vars()
        self._init_config()
        
    def _validate_env_vars(self):
        # 必要な環境変数のリスト
        required_env_vars = [
            "COHERE_API_KEY",
            "TNS_ADMIN",
            "DB_USER",
            "DB_PASSWORD",
            "DB_DSN",
            "OCI_CONFIG_PROFILE",
            "OCI_REGION",
            "OCI_COMPARTMENT_ID",
            "MLLM_MODEL_ID",
            "EMBED_MODEL_PROVIDER",
            "EMBED_MODEL_ID"
        ]
        
        # 環境変数の存在確認
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        # CohereAI APIキーをプロバイダーに応じて必須にする
        embed_provider = os.getenv("EMBED_MODEL_PROVIDER")
        if embed_provider == "CohereAI" and not os.getenv("COHERE_API_KEY"):
            missing_vars.append("COHERE_API_KEY")
        
        if missing_vars:
            print("エラー: 以下の環境変数が設定されていません:")
            for var in missing_vars:
                print(f"  - {var}")
            exit(1)
    
    def _init_config(self):
        # 環境変数から設定を読み込む
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_dsn = os.getenv("DB_DSN")
        
        # OCI Config の設定
        self.config_profile = os.getenv("OCI_CONFIG_PROFILE")
        self.oci_config = oci.config.from_file(
            file_location='~/.oci/config', 
            profile_name=self.config_profile
        )
        self.oci_config["region"] = os.getenv("OCI_REGION")
        
        self.compartment_id = os.getenv("OCI_COMPARTMENT_ID") 
        self.mllm_model_id = os.getenv("MLLM_MODEL_ID")
        
        # 埋め込みモデルの設定
        self.embed_model_provider = os.getenv("EMBED_MODEL_PROVIDER")
        self.embed_model_id = os.getenv("EMBED_MODEL_ID")
        
        # アプリケーション実行環境の設定
        self.app_mode = os.getenv("APP_MODE", "local").lower()
        
        # リモートモード設定
        self.remote_share = os.getenv("REMOTE_SHARE", "true").lower() == "true"
        self.remote_server_name = os.getenv("REMOTE_SERVER_NAME", "0.0.0.0")
        self.remote_server_port = int(os.getenv("REMOTE_SERVER_PORT", "8899"))
        
        # ローカルモード設定
        self.local_share = os.getenv("LOCAL_SHARE", "false").lower() == "true"
        self.local_inbrowser = os.getenv("LOCAL_INBROWSER", "true").lower() == "true"
        
    def get_cohere_client(self):
        # Cohereクライアントを初期化
        return cohere.Client(api_key=self.cohere_api_key) 
    
    def get_oci_generative_ai_client(self):
        # OCI Generative AI クライアントを初期化
        return oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=self.oci_config, 
            retry_strategy=oci.retry.NoneRetryStrategy(), 
            timeout=(10, 240)
        )
    
    def get_db_pool(self):
        """アプリプロセス内で共有する単一のコネクションプールを返す"""
        existing_pool = oracledb.get_pool(POOL_ALIAS)
        if existing_pool is not None:
            return existing_pool

        return oracledb.create_pool(
            user=self.db_user,
            password=self.db_password,
            dsn=self.db_dsn,
            min=DEFAULT_POOL_MIN,
            max=DEFAULT_POOL_MAX,
            increment=DEFAULT_POOL_INCREMENT,
            timeout=DEFAULT_POOL_TIMEOUT,
            getmode=oracledb.POOL_GETMODE_WAIT,
            ping_interval=DEFAULT_PING_INTERVAL,
            ping_timeout=DEFAULT_PING_TIMEOUT,
            pool_alias=POOL_ALIAS,
        )

    @staticmethod
    def close_db_pool():
        """アプリ終了時にコネクションプールをクローズする"""
        pool = oracledb.get_pool(POOL_ALIAS)
        if pool is not None:
            pool.close()

    def get_launch_config(self):
        """Gradioアプリケーションの起動設定を取得"""
        if self.app_mode == "remote":
            return {
                "share": self.remote_share,
                "server_name": self.remote_server_name,
                "server_port": self.remote_server_port,
                "inbrowser": False
            }
        else:  # local mode
            return {
                "share": self.local_share,
                "inbrowser": self.local_inbrowser
            }