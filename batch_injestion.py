import oci
import cohere
import oracledb
from PIL import Image
from io import BytesIO
import os
import array
import time
import glob
import sys
from dotenv import load_dotenv, find_dotenv

from app.embedding_service import EmbeddingService
from app.nlp_service import NLPService
from app.vlm_service_factory import VLMServiceFactory


DEFAULT_CAPTION_PROMPT = """
この画像を詳しく分析してください。

以下の観点で画像を分析してください。
1. 画像に何が写っているか
2. 全体的な印象や特徴
3. 注目すべきポイント
4. 画像に描かれているもののカテゴリと固有の名称
5. 画像に描かれているテキスト
6. 画像に描かれている URL、IDなどの情報
7. 画像が説明、紹介しようとしている内容

テキストはすべて抽出してください。
日本語で詳しく説明してください。
"""


def get_image_caption(nlp_service, image_path, vlm_settings, custom_prompt=None):
    """UIアプリと同じVLMサービス経由で画像キャプションを生成する"""
    caption = nlp_service.generate_caption_with_vlm(
        image_path=image_path,
        vlm_model=vlm_settings["model"],
        prompt_text=custom_prompt or DEFAULT_CAPTION_PROMPT,
        temperature=vlm_settings["temperature"],
        max_tokens=vlm_settings["max_tokens"],
        oci_region=vlm_settings["oci_region"],
    )

    error_markers = (
        "エラー:",
        "API エラー:",
        "キャプション生成中にエラーが発生しました",
    )
    if isinstance(caption, str) and any(marker in caption for marker in error_markers):
        raise RuntimeError(caption)

    print("************************** Generated Caption**************************")
    print(caption)
    return caption


def is_image_registered(db_connection, file_name):
    """画像が既にデータベースに登録されているかチェック"""
    cursor = db_connection.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM IMAGES WHERE file_name = :1", [file_name])
        count = cursor.fetchone()[0]
        return count > 0
    finally:
        cursor.close()


def clean_caption(caption):
    """キャプションから不要な文字列を削除する関数"""
    # 削除対象の文字列リスト
    remove_patterns = [
        "以下に、画像の内容を詳しく分析します。",
        "**画像に何が写っているか**：",
        "画像に何が写っているか",
        "**全体的な印象や特徴**：",
        "全体的な印象や特徴",
        "**注目すべきポイント**：",
        "注目すべきポイント",
        "**画像に描かれているもののカテゴリと固有の名称**：",
        "画像に描かれているもののカテゴリと固有の名称",
        "**画像に描かれているテキスト**：",
        "画像に描かれているテキスト",
        "**画像に描かれている URL、IDなどの情報**：",
        "画像に描かれている URL、IDなどの情報",
        "**画像が説明・紹介しようとしている内容**：",
        "画像が説明・紹介しようとしている内容",
        "この画像にはテキストは一切表示されていません。",
        "画像内には URL や ID、その他の特定の情報は含まれていません。",
        "具体的には以下の内容です：",
        "画像内のテキストは以下の通りです。",
    ]
    
    # 各削除対象文字列を順次削除
    cleaned_caption = caption
    for pattern in remove_patterns:
        cleaned_caption = cleaned_caption.replace(pattern, "")
    
    # 複数の連続する改行を単一の改行に置換
    import re
    cleaned_caption = re.sub(r'\n\s*\n', '\n', cleaned_caption)
    
    # 先頭と末尾の空白・改行を削除
    cleaned_caption = cleaned_caption.strip()
    
    return cleaned_caption


def insert_image_to_db(embedding_service, nlp_service, vlm_settings, db_connection, image_data, image_path, file_name):
    """画像とその説明文をOracle Databaseに挿入"""
    raw_caption = get_image_caption(nlp_service, image_path, vlm_settings)
    
    # キャプションをクリーニング
    caption = clean_caption(raw_caption)
    
    # VARCHAR2(4000)の制限を確実に適用（バイト数ベース）
    caption_bytes = caption.encode('utf-8')
    if len(caption_bytes) > 4000:
        # バイト数で切り詰め、文字境界を考慮
        truncated_bytes = caption_bytes[:4000]
        # 不完全なUTF-8文字を避けるため、最後の文字境界まで戻る
        while len(truncated_bytes) > 0:
            try:
                caption = truncated_bytes.decode('utf-8')
                break
            except UnicodeDecodeError:
                truncated_bytes = truncated_bytes[:-1]
        print(f"警告: キャプションが4000バイトを超えたため切り詰めました。元のバイト数: {len(caption_bytes)}, 文字数: {len(clean_caption(raw_caption))}")
    
    # UIアプリと同じ埋め込みサービスでベクトルを生成
    pil_image = Image.open(BytesIO(image_data))
    image_embedding = array.array('f', embedding_service.get_image_embedding(pil_image))
    caption_embedding = array.array('f', embedding_service.get_text_embedding(caption, "search_document"))
    
    cursor = db_connection.cursor()
    try:
        # 画像データを挿入
        cursor.execute("""
            INSERT INTO IMAGES (file_name, caption, caption_embedding, image_data, image_embedding)
            VALUES (:1, :2, :3, :4, :5)
        """, (
            file_name,
            caption,
            caption_embedding,
            image_data,
            image_embedding
        ))
        
        db_connection.commit()
        print(f"画像 '{file_name}' が正常に挿入されました。")
        return True
    except Exception as e:
        print(f"画像 '{file_name}' の挿入中にエラーが発生しました: {str(e)}")
        raise
    finally:
        cursor.close()


if __name__ == "__main__":
    # 処理開始時間を記録
    start_time = time.time()
    
    # 環境変数を読み込む
    load_dotenv(find_dotenv())
    
    # 必要な環境変数のリスト
    required_env_vars = [
        "TNS_ADMIN",
        "DB_USER",
        "DB_PASSWORD",
        "DB_DSN",
        "EMBED_MODEL_PROVIDER",
        "EMBED_MODEL_ID",
        "OCI_CONFIG_PROFILE",
        "OCI_REGION",
        "OCI_COMPARTMENT_ID",
        "MLLM_MODEL_PROVIDER",
        "MLLM_MODEL_ID"
    ]
    
    # 環境変数の存在確認
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    # CohereAI APIキーをプロバイダーに応じて必須にする
    EMBED_MODEL_PROVIDER = os.getenv("EMBED_MODEL_PROVIDER")
    if EMBED_MODEL_PROVIDER == "CohereAI" and not os.getenv("COHERE_API_KEY"):
        missing_vars.append("COHERE_API_KEY")
    
    if missing_vars:
        print("エラー: 以下の環境変数が設定されていません:")
        for var in missing_vars:
            print(f"  - {var}")
        exit(1)
    
    # 環境変数から設定を読み込む
    USERNAME = os.getenv("DB_USER")
    PASSWORD = os.getenv("DB_PASSWORD")
    DSN = os.getenv("DB_DSN")
    EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
    
    # 画像ディレクトリのパス
    IMAGE_DIR = "images"
    
    # 統計情報の初期化
    total_images = 0
    already_registered = 0
    newly_registered = 0
    failed_registrations = 0

    # OCI Config の設定
    CONFIG_PROFILE = os.getenv("OCI_CONFIG_PROFILE")
    config = oci.config.from_file(file_location='~/.oci/config', profile_name=CONFIG_PROFILE)
    config["region"] = os.getenv("OCI_REGION")

    COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID") 
    MLLM_MODEL_ID = os.getenv("MLLM_MODEL_ID")
    MLLM_MODEL_PROVIDER = os.getenv("MLLM_MODEL_PROVIDER")
    
    try:
        # OCI GenAIクライアントを初期化（埋め込みベクトル生成用）
        generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
        
        # Cohereクライアントを初期化（必要な場合のみ）
        cohere_client = None
        if EMBED_MODEL_PROVIDER == "CohereAI":
            COHERE_API_KEY = os.getenv("COHERE_API_KEY")
            cohere_client = cohere.Client(api_key=COHERE_API_KEY)

        # UIアプリのアップロードタブと同じサービス設定でVLM/埋め込みを初期化
        embedding_service = EmbeddingService(
            embed_model_provider=EMBED_MODEL_PROVIDER,
            embed_model_id=EMBED_MODEL_ID,
            compartment_id=COMPARTMENT_ID,
            cohere_client=cohere_client,
            oci_client=generative_ai_inference_client,
        )
        upload_vlm_service = VLMServiceFactory.create_upload_vlm_service()
        vlm_settings = upload_vlm_service.get_current_vlm_settings()
        nlp_service = NLPService(upload_vlm_service)

        print(f"埋め込みモデルプロバイダー: {EMBED_MODEL_PROVIDER}")
        print(f"埋め込みモデルID: {EMBED_MODEL_ID}")
        print(f"VLM モデルプロバイダー: {MLLM_MODEL_PROVIDER}")
        print(f"VLM モデルID: {MLLM_MODEL_ID}")
        print(f"VLM UIモデル: {vlm_settings['model']}")
        print(f"VLM temperature: {vlm_settings['temperature']}")
        print(f"VLM max_tokens: {vlm_settings['max_tokens']}")
        print(f"VLM OCIリージョン: {vlm_settings['oci_region']}")

        # Oracle接続を確立
        db_connection = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
        print("データベース接続成功!")
        
        # images ディレクトリ内のすべての画像ファイルを取得
        # 大文字小文字の拡張子とjpeg、webp形式にも対応
        image_files = []
        for pattern in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.webp", "*.WEBP"]:
            image_files.extend(glob.glob(os.path.join(IMAGE_DIR, pattern)))
        total_images = len(image_files)
        
        print(f"処理対象の画像ファイル数: {total_images}")
        
        # 各画像ファイルを処理
        for image_path in image_files:
            file_name = os.path.basename(image_path)
            
            # 既に登録されているかチェック
            if is_image_registered(db_connection, file_name):
                print(f"画像 '{file_name}' は既に登録されています。スキップします。")
                already_registered += 1
                continue
            
            print(f"画像 '{file_name}' を処理中...")
            
            try:
                # 画像データを読み込む
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                
                # 画像データを挿入
                if insert_image_to_db(embedding_service, nlp_service, vlm_settings, db_connection, image_data, image_path, file_name):
                    newly_registered += 1
                else:
                    failed_registrations += 1
                    
            except Exception as e:
                print(f"画像 '{file_name}' の処理中にエラーが発生しました: {str(e)}")
                raise
        
        # 処理時間を計算
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 統計情報を表示
        print("\n===== 処理結果サマリー =====")
        print(f"ディレクトリ内の画像ファイル総数: {total_images}")
        print(f"既に登録済みの画像数: {already_registered}")
        print(f"今回新規登録した画像数: {newly_registered}")
        print(f"登録に失敗した画像数: {failed_registrations}")
        print(f"処理時間: {processing_time:.2f} 秒")
        
    except Exception as e:
        print("エラーが発生しました！")
        print(f"エラーの種類: {type(e).__name__}")
        print(f"エラーの内容: {str(e)}")
        print("エラーが発生したため処理を中止します。")
        sys.exit(1)
    
    finally:
        # DB接続を閉じる
        if 'db_connection' in locals():
            db_connection.close()
