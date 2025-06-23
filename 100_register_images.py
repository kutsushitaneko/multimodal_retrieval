import oci
import cohere
import oracledb
from PIL import Image
from io import BytesIO
import base64
import os
import array
import time
import glob
import sys
from dotenv import load_dotenv, find_dotenv

def image_to_base64_data_url(image_data):
    """画像データをBase64エンコードしてData URLに変換"""
    img = Image.open(BytesIO(image_data))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_base64}"
    return data_url

def get_image_caption(generative_ai_inference_client, image_data):
    """画像データからキャプションを生成する関数"""
    PROMPT = """
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
    content1 = oci.generative_ai_inference.models.TextContent()
    content1.text = PROMPT
    content2 = oci.generative_ai_inference.models.ImageContent()
    image_url = oci.generative_ai_inference.models.ImageUrl()
    image_url.url = image_to_base64_data_url(image_data)
    content2.image_url = image_url
    message = oci.generative_ai_inference.models.UserMessage()
    message.content = [content1,content2]

    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = [message]
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.num_generations = 1
    chat_request.max_tokens = 1000
    chat_request.is_stream = False
    chat_request.temperature = 0.70
    chat_request.top_p = 0.7
    chat_request.top_k = -1
    chat_request.frequency_penalty = 0.5
    chat_request.presence_penalty = 0.5

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MLLM_MODEL_ID)
    chat_detail.compartment_id = COMPARTMENT_ID
    chat_detail.chat_request = chat_request

    chat_response = generative_ai_inference_client.chat(chat_detail)

    # Print result
    print("************************** Chat Result *******************************")
    print(vars(chat_response))
    print("************************** Generated Caption**************************")
    
    # 正しいレスポンス構造からテキストを取得
    if hasattr(chat_response, 'data') and hasattr(chat_response.data, 'chat_response'):
        if hasattr(chat_response.data.chat_response, 'choices') and len(chat_response.data.chat_response.choices) > 0:
            choice = chat_response.data.chat_response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                for content in choice.message.content:
                    if hasattr(content, 'text'):
                        print(content.text)
                        # VARCHAR2(4000)の制限を考慮して、キャプションを4000文字以内に切り詰める
                        return content.text[:4000] if len(content.text) > 4000 else content.text
    
    # 上記の方法でテキストを取得できない場合は、代替方法を試す
    print("標準的な方法でテキストを取得できませんでした。代替方法を試みます。")
    try:
        # レスポンスの構造を確認
        response_data = chat_response.data
        if hasattr(response_data, 'chat_response'):
            chat_response_data = response_data.chat_response
            if hasattr(chat_response_data, 'choices') and len(chat_response_data.choices) > 0:
                first_choice = chat_response_data.choices[0]
                if hasattr(first_choice, 'message'):
                    message = first_choice.message
                    if hasattr(message, 'content') and len(message.content) > 0:
                        for content_item in message.content:
                            if hasattr(content_item, 'text'):
                                # VARCHAR2(4000)の制限を考慮して、キャプションを4000文字以内に切り詰める
                                return content_item.text[:4000] if len(content_item.text) > 4000 else content_item.text
    except Exception as e:
        print(f"代替方法でもエラーが発生しました: {str(e)}")
    
    # すべての方法が失敗した場合
    return "画像の説明を生成できませんでした。"

def get_image_embedding_oci(generative_ai_inference_client, image_data):
    """OCI GenAI Serviceを使用して画像の埋め込みベクトルを生成"""
    # 画像データをBase64エンコードしてData URLに変換
    data_url = image_to_base64_data_url(image_data)
    
    # OCI GenAI Serviceを使用して画像の埋め込みベクトルを取得
    # 画像埋め込みもEmbedTextDetailsを使用し、画像データをinputsに渡す
    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
    embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=EMBED_MODEL_ID)
    embed_text_detail.compartment_id = COMPARTMENT_ID
    embed_text_detail.inputs = [data_url]  # 画像のData URLを文字列として渡す
    embed_text_detail.input_type = "IMAGE"
    
    embedding_response = generative_ai_inference_client.embed_text(embed_text_detail)
    
    return embedding_response.data.embeddings[0]

def get_image_embedding_cohere(cohere_client, image_data):
    """CohereAIを使用して画像の埋め込みベクトルを生成"""
    # 画像データをBase64エンコードしてData URLに変換
    data_url = image_to_base64_data_url(image_data)
    
    # Cohere APIを使用して画像の埋め込みベクトルを取得
    response = cohere_client.embed(
        images=[data_url],
        model="embed-v4.0",
        input_type="image",
        embedding_types=["float"],
    )
    
    return response.embeddings.float[0]

def get_text_embedding_oci(generative_ai_inference_client, text):
    """OCI GenAI Serviceを使用してテキストの埋め込みベクトルを生成"""
    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
    embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=EMBED_MODEL_ID)
    embed_text_detail.compartment_id = COMPARTMENT_ID
    embed_text_detail.inputs = [text]
    embed_text_detail.input_type = "SEARCH_DOCUMENT"
    embed_text_detail.truncate = "END"
    
    embedding_response = generative_ai_inference_client.embed_text(embed_text_detail)
    
    return embedding_response.data.embeddings[0]

def get_text_embedding_cohere(cohere_client, text):
    """CohereAIを使用してテキストの埋め込みベクトルを生成"""
    response = cohere_client.embed(
        texts=[text],
        model="embed-v4.0",
        input_type="search_document"
    )
    
    return response.embeddings[0]

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

def insert_image_to_db(generative_ai_inference_client, mllm_client, cohere_client, db_connection, image_data, file_name):
    """画像とその説明文をOracle Databaseに挿入"""
    raw_caption = get_image_caption(mllm_client, image_data)
    
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
    
    # 画像とテキストの埋め込みベクトルを取得（プロバイダーに応じて切り替え）
    if EMBED_MODEL_PROVIDER == "OCI":
        image_embedding = array.array('f', get_image_embedding_oci(generative_ai_inference_client, image_data))
        caption_embedding = array.array('f', get_text_embedding_oci(generative_ai_inference_client, caption))
    elif EMBED_MODEL_PROVIDER == "CohereAI":
        image_embedding = array.array('f', get_image_embedding_cohere(cohere_client, image_data))
        caption_embedding = array.array('f', get_text_embedding_cohere(cohere_client, caption))
    else:
        raise ValueError(f"サポートされていない埋め込みモデルプロバイダーです: {EMBED_MODEL_PROVIDER}")
    
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
        
        # MLLM用のクライアントを初期化（リージョンオーバーライドがある場合は別のクライアントを作成）
        mllm_client = generative_ai_inference_client  # デフォルトは同じクライアントを使用
        
        if MLLM_MODEL_PROVIDER == "OCI":
            mllm_region_override = os.getenv("OCI_REGION_OVERRIDE_FOR_MLLM")
            if mllm_region_override:
                # MLLM専用のリージョンが指定されている場合は、別のクライアントを作成
                mllm_config = oci.config.from_file(file_location='~/.oci/config', profile_name=CONFIG_PROFILE)
                mllm_config["region"] = mllm_region_override
                mllm_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=mllm_config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
                print(f"MLLM専用リージョン: {mllm_region_override}")
            else:
                print(f"MLLM用リージョン: {os.getenv('OCI_REGION')} (デフォルト)")
        else:
            print(f"MLLM用リージョン: {os.getenv('OCI_REGION')} (非OCIプロバイダー)") 

        # Cohereクライアントを初期化（必要な場合のみ）
        cohere_client = None
        if EMBED_MODEL_PROVIDER == "CohereAI":
            COHERE_API_KEY = os.getenv("COHERE_API_KEY")
            cohere_client = cohere.Client(api_key=COHERE_API_KEY)

        print(f"埋め込みモデルプロバイダー: {EMBED_MODEL_PROVIDER}")
        print(f"埋め込みモデルID: {EMBED_MODEL_ID}")
        print(f"VLM モデルプロバイダー: {MLLM_MODEL_PROVIDER}")
        print(f"VLM モデルID: {MLLM_MODEL_ID}")

        # Oracle接続を確立
        db_connection = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
        print("データベース接続成功!")
        
        # images ディレクトリ内のすべての画像ファイルを取得
        # 大文字小文字の拡張子とjpeg形式にも対応
        image_files = []
        for pattern in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
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
                if insert_image_to_db(generative_ai_inference_client, mllm_client, cohere_client, db_connection, image_data, file_name):
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
