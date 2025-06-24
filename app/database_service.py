import time
from io import BytesIO
from PIL import Image
import oracledb
import base64
import array
import re
import oci

class DatabaseService:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.max_retries = 3
        self.retry_delay = 1  # 秒
        
    def _execute_with_retry(self, operation_func):
        """データベース操作を実行し、接続エラー時には再試行する汎用関数"""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                return operation_func()
            except oracledb.DatabaseError as e:
                error, = e.args
                # 接続エラーコードを確認
                if error.code in (3113, 3114, 12541, 12545, 17002, 17008, 17410):
                    # 接続が切れた場合
                    last_error = e
                    retries += 1
                    if retries < self.max_retries:
                        print(f"データベース接続エラー（リトライ {retries}/{self.max_retries}）: {e}")
                        time.sleep(self.retry_delay * retries)  # 指数バックオフ
                        continue
                    else:
                        print(f"データベース接続の再試行回数上限に達しました: {e}")
                raise  # その他のデータベースエラーはそのまま上位に伝播
        
        # 最大リトライ回数に達した場合
        raise last_error
        
    def get_image_caption(self, mllm_client, image_data, mllm_model_id, compartment_id):
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
        
        # 画像データをBase64エンコードしてData URLに変換
        img = Image.open(BytesIO(image_data))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_base64}"
        
        content1 = oci.generative_ai_inference.models.TextContent()
        content1.text = PROMPT
        content2 = oci.generative_ai_inference.models.ImageContent()
        image_url = oci.generative_ai_inference.models.ImageUrl()
        image_url.url = data_url
        content2.image_url = image_url
        message = oci.generative_ai_inference.models.UserMessage()
        message.content = [content1, content2]

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
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=mllm_model_id)
        chat_detail.compartment_id = compartment_id
        chat_detail.chat_request = chat_request

        chat_response = mllm_client.chat(chat_detail)

        # 正しいレスポンス構造からテキストを取得
        if hasattr(chat_response, 'data') and hasattr(chat_response.data, 'chat_response'):
            if hasattr(chat_response.data.chat_response, 'choices') and len(chat_response.data.chat_response.choices) > 0:
                choice = chat_response.data.chat_response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    for content in choice.message.content:
                        if hasattr(content, 'text'):
                            # VARCHAR2(4000)の制限を考慮して、キャプションを4000文字以内に切り詰める
                            raw_caption = content.text[:4000] if len(content.text) > 4000 else content.text
                            return self._clean_caption(raw_caption)
        
        # すべての方法が失敗した場合
        return "画像の説明を生成できませんでした。"

    def _clean_caption(self, caption):
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
        cleaned_caption = re.sub(r'\n\s*\n', '\n', cleaned_caption)
        
        # 先頭と末尾の空白・改行を削除
        cleaned_caption = cleaned_caption.strip()
        
        return cleaned_caption

    def is_image_registered(self, file_name):
        """画像が既にデータベースに登録されているかチェック"""
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT COUNT(*) FROM IMAGES WHERE file_name = :1", [file_name])
                    count = cursor.fetchone()[0]
                    return count > 0
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)

    def insert_image_to_db(self, embedding_service, mllm_client, mllm_model_id, compartment_id, image_data, file_name):
        """画像とその説明文をOracle Databaseに挿入"""
        # キャプションを生成
        caption = self.get_image_caption(mllm_client, image_data, mllm_model_id, compartment_id)
        
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
            print(f"警告: キャプションが4000バイトを超えたため切り詰めました。元のバイト数: {len(caption_bytes)}")
        
        # 画像とテキストの埋め込みベクトルを取得
        pil_image = Image.open(BytesIO(image_data))
        image_embedding = array.array('f', embedding_service.get_image_embedding(pil_image))
        caption_embedding = array.array('f', embedding_service.get_text_embedding(caption, "search_document"))
        
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
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
                    
                    conn.commit()
                    print(f"画像 '{file_name}' が正常に挿入されました。")
                    return True, caption
                except Exception as e:
                    print(f"画像 '{file_name}' の挿入中にエラーが発生しました: {str(e)}")
                    raise
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)

    def update_image_caption(self, embedding_service, image_id, new_caption):
        """画像のキャプションを更新"""
        # VARCHAR2(4000)の制限を確実に適用
        caption_bytes = new_caption.encode('utf-8')
        if len(caption_bytes) > 4000:
            truncated_bytes = caption_bytes[:4000]
            while len(truncated_bytes) > 0:
                try:
                    new_caption = truncated_bytes.decode('utf-8')
                    break
                except UnicodeDecodeError:
                    truncated_bytes = truncated_bytes[:-1]
        
        # 新しいキャプションの埋め込みベクトルを生成
        caption_embedding = array.array('f', embedding_service.get_text_embedding(new_caption, "search_document"))
        
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE IMAGES 
                        SET caption = :1, caption_embedding = :2
                        WHERE image_id = :3
                    """, (new_caption, caption_embedding, image_id))
                    
                    conn.commit()
                    print(f"画像ID {image_id} のキャプションが正常に更新されました。")
                    return True
                except Exception as e:
                    print(f"画像ID {image_id} のキャプション更新中にエラーが発生しました: {str(e)}")
                    raise
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)

    def get_image_by_filename(self, file_name):
        """ファイル名で画像を検索"""
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    sql = """
                        SELECT image_id, file_name, caption, image_data,
                            NULL as distance
                        FROM IMAGES
                        WHERE file_name = :1
                    """
                    cursor.execute(sql, [file_name])
                    
                    results = self._process_query_results(cursor, "ファイル名検索")
                    return results[0] if results else None
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)

    def search_by_caption_vector(self, query_embedding, top_k=5, vector_threshold=0.5):
        """ベクトル埋め込みによるキャプション検索"""
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    sql = """
                        SELECT a.image_id, a.file_name, a.caption, a.image_data,
                            VECTOR_DISTANCE(a.caption_embedding, :1, DOT) as distance
                        FROM IMAGES a
                        WHERE VECTOR_DISTANCE(a.caption_embedding, :2, DOT) <= :3
                        ORDER BY distance
                        FETCH APPROX FIRST :4 ROWS ONLY
                    """
                    cursor.execute(sql, [
                        query_embedding, 
                        query_embedding, 
                        -1 * vector_threshold, 
                        top_k
                    ])
                    
                    executed_sql = sql.replace(":1", ":embedding").replace(":2", ":embedding") \
                                      .replace(":3", str(-1 * vector_threshold)).replace(":4", str(top_k))
                    
                    results = self._process_query_results(cursor, "ベクトル検索")
                    return results, executed_sql
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)
            
    def search_by_fulltext(self, search_query, top_k=5, keyword_threshold=0):
        """全文検索によるキャプション検索"""
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    sql = """
                        SELECT a.image_id, a.file_name, a.caption, a.image_data,
                            score(1) as distance
                        FROM IMAGES a
                        WHERE CONTAINS(caption, :1, 1) > 0
                        AND score(1) >= :2
                        ORDER BY score(1) DESC
                        FETCH FIRST :3 ROWS ONLY
                    """
                    
                    try:
                        cursor.execute(sql, [search_query, keyword_threshold, top_k])
                    except oracledb.DatabaseError as e:
                        error, = e.args
                        # Oracle Text関連のエラー（DRG-50921など）をチェック
                        if error.code in (29902, 30600) or 'DRG-' in str(e):
                            print(f"Oracle Text検索エラー: {e}")
                            print(f"問題のクエリ: {search_query}")
                            
                            # フォールバック：通常のLIKE検索を試行
                            fallback_sql = """
                                SELECT a.image_id, a.file_name, a.caption, a.image_data,
                                    0 as distance
                                FROM IMAGES a
                                WHERE UPPER(caption) LIKE UPPER(:1)
                                ORDER BY a.upload_date DESC
                                FETCH FIRST :2 ROWS ONLY
                            """
                            
                            # 検索クエリをLIKE検索用に変換（AND演算子を削除し、最初のキーワードのみ使用）
                            like_query = search_query.split(' AND ')[0].replace('"', '').strip()
                            like_pattern = f"%{like_query}%"
                            
                            print(f"フォールバック検索を実行: {like_pattern}")
                            cursor.execute(fallback_sql, [like_pattern, top_k])
                            
                            executed_sql = fallback_sql.replace(":1", f"'%{like_query}%'") \
                                                      .replace(":2", str(top_k))
                            executed_sql += f"\n-- 元のOracle Text検索でエラーが発生したため、LIKE検索にフォールバック\n-- 元のクエリ: {search_query}"
                        else:
                            raise  # その他のデータベースエラーは再発生
                    else:
                        executed_sql = sql.replace(":1", f"'{search_query}'") \
                                          .replace(":2", str(keyword_threshold)) \
                                          .replace(":3", str(top_k))
                    
                    results = self._process_query_results(cursor, "全文検索")
                    return results, executed_sql
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)
            
    def search_by_image_vector(self, query_embedding, top_k=5, vector_threshold=0.5):
        """画像ベクトルによる検索"""
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    sql = """
                        SELECT a.image_id, a.file_name, a.caption, a.image_data,
                            VECTOR_DISTANCE(a.image_embedding, :1, DOT) as distance
                        FROM IMAGES a
                        WHERE VECTOR_DISTANCE(a.image_embedding, :2, DOT) <= :3
                        ORDER BY distance
                        FETCH APPROX FIRST :4 ROWS ONLY
                    """
                    cursor.execute(sql, [
                        query_embedding, 
                        query_embedding, 
                        -1 * vector_threshold, 
                        top_k
                    ])
                    
                    executed_sql = sql.replace(":1", ":embedding").replace(":2", ":embedding") \
                                      .replace(":3", str(-1 * vector_threshold)).replace(":4", str(top_k))
                    
                    results = self._process_query_results(cursor, "画像")
                    return results, executed_sql
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)
            
    def get_recent_images(self, top_k=12, offset=0):
        """最近アップロードされた画像を取得"""
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    sql = """
                        SELECT image_id, file_name, caption, image_data,
                            NULL as distance
                        FROM IMAGES
                        ORDER BY upload_date DESC
                        OFFSET :1 ROWS FETCH NEXT :2 ROWS ONLY
                    """
                    cursor.execute(sql, [offset, top_k])
                    executed_sql = sql.replace(":1", str(offset)).replace(":2", str(top_k))
                    
                    results = self._process_query_results(cursor, "最近のアップロード")
                    return results, executed_sql
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)
            
    def get_total_image_count(self):
        """画像の総数を取得"""
        def operation():
            with self.db_pool.acquire() as conn:
                cursor = conn.cursor()
                try:
                    sql = "SELECT COUNT(*) FROM IMAGES"
                    cursor.execute(sql)
                    count = cursor.fetchone()[0]
                    return count
                finally:
                    cursor.close()
        
        return self._execute_with_retry(operation)
            
    def _process_query_results(self, cursor, search_mode):
        """クエリ結果を処理してオブジェクトのリストを返す"""
        results = []
        for row in cursor:
            image_id, file_name, caption, image_data, distance = row
            # BLOBデータをPILイメージに変換
            img = Image.open(BytesIO(image_data.read()))
            caption_text = caption
            results.append({
                'image_id': image_id,
                'file_name': file_name,
                'caption': caption_text,
                'image': img,
                'distance': distance,
                'search_mode': search_mode
            })
        return results

    def delete_image(self, image_id):
        """指定されたIDの画像をデータベースから削除する"""
        def operation():
            try:
                with self.db_pool.acquire() as conn:
                    cursor = conn.cursor()
                    try:
                        # 画像が存在するかチェック
                        cursor.execute("SELECT file_name FROM IMAGES WHERE image_id = :1", (image_id,))
                        result = cursor.fetchone()
                        
                        if not result:
                            print(f"警告: 画像ID {image_id} が見つかりません")
                            return False
                            
                        file_name = result[0]
                        
                        # 画像を削除
                        cursor.execute("DELETE FROM IMAGES WHERE image_id = :1", (image_id,))
                        conn.commit()
                        
                        print(f"画像ID {image_id} (ファイル名: {file_name}) を削除しました")
                        return True
                        
                    finally:
                        cursor.close()
                        
            except Exception as e:
                print(f"画像削除エラー: {e}")
                return False
        
        return self._execute_with_retry(operation) 