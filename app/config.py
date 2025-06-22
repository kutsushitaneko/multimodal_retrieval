import os
import oci
import cohere
import oracledb
from dotenv import load_dotenv, find_dotenv

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
            "MLLM_MODEL_ID"
        ]
        
        # 環境変数の存在確認
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
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
        
        # アプリケーション実行環境の設定
        self.app_mode = os.getenv("APP_MODE", "local").lower()
        
        # リモートモード設定
        self.remote_share = os.getenv("REMOTE_SHARE", "true").lower() == "true"
        self.remote_server_name = os.getenv("REMOTE_SERVER_NAME", "0.0.0.0")
        self.remote_server_port = int(os.getenv("REMOTE_SERVER_PORT", "8899"))
        
        # ローカルモード設定
        self.local_share = os.getenv("LOCAL_SHARE", "false").lower() == "true"
        self.local_inbrowser = os.getenv("LOCAL_INBROWSER", "true").lower() == "true"
        
    def get_db_connection(self):
        # データベース接続を確立
        db_connection = oracledb.connect(
            user=self.db_user, 
            password=self.db_password, 
            dsn=self.db_dsn
        )
        # print("データベース接続成功!")
        return db_connection
        
    def get_cohere_client(self):
        # Cohereクライアントを初期化
        return cohere.Client(api_key=self.cohere_api_key) 
    
    def get_db_pool(self):
        # コネクションプールを生成
        pool = oracledb.create_pool(
            user=self.db_user,
            password=self.db_password,
            dsn=self.db_dsn,
            min=2,
            max=10,
            increment=1,
            timeout=60,
            getmode=oracledb.POOL_GETMODE_WAIT
        )
        return pool
        
    def check_pool_health(self, pool):
        """コネクションプールの健全性を確認するメソッド"""
        try:
            with pool.acquire() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                cursor.close()
                return result is not None and result[0] == 1
        except oracledb.DatabaseError as e:
            print(f"プール健全性チェックエラー: {e}")
            return False
    
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