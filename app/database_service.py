import time
from io import BytesIO
from PIL import Image
import oracledb
import base64
import array
import re
import oci

CONNECTION_ERROR_CODES = (3113, 3114, 12541, 12545, 17002, 17008, 17410)


class DatabaseService:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.max_retries = 3
        self.retry_delay = 1  # 秒

    @staticmethod
    def _is_connection_error(exc):
        if not isinstance(exc, oracledb.DatabaseError):
            return False
        error, = exc.args
        return error.code in CONNECTION_ERROR_CODES

    def _execute_with_retry(self, operation_func):
        """データベース操作を実行し、接続エラー時には接続を破棄して再試行する"""
        last_error = None

        for attempt in range(self.max_retries):
            conn = None
            try:
                conn = self.db_pool.acquire()
                return operation_func(conn)
            except oracledb.DatabaseError as e:
                if self._is_connection_error(e):
                    last_error = e
                    if conn is not None:
                        try:
                            self.db_pool.drop(conn)
                        except Exception:
                            pass
                        conn = None
                    if attempt < self.max_retries - 1:
                        print(
                            f"データベース接続エラー（リトライ {attempt + 1}/{self.max_retries}）: {e}"
                        )
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    print(f"データベース接続の再試行回数上限に達しました: {e}")
                raise
            finally:
                if conn is not None:
                    conn.close()

        if last_error is not None:
            raise last_error
        
    def get_image_caption(self, mllm_client, image_data, mllm_model_id, compartment_id, custom_prompt=None):
        """画像データからキャプションを生成する関数"""
        if custom_prompt:
            PROMPT = custom_prompt
        else:
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
        
        # RGBA形式の場合はRGB形式に変換（JPEG形式はアルファチャンネルをサポートしないため）
        if img.mode in ('RGBA', 'LA'):
            # 白い背景に合成
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])  # アルファチャンネルをマスクとして使用
            else:  # LA (Luminance + Alpha)
                background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode not in ('RGB', 'L'):
            # その他のモードもRGBに変換
            img = img.convert('RGB')
        
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
                            # VARCHAR2(4000)の制限を考慮して、キャプションを4000バイト以内に切り詰める
                            raw_caption = self._truncate_caption_safely(content.text, 4000)
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

    def _truncate_caption_safely(self, text, max_bytes):
        """テキストを指定したバイト数以内に安全に切り詰める（文字境界を考慮）"""
        if not text:
            return text
            
        # テキストをUTF-8でエンコード
        text_bytes = text.encode('utf-8')
        
        # 指定バイト数以下の場合はそのまま返す
        if len(text_bytes) <= max_bytes:
            return text
        
        # 指定バイト数で切り詰め
        truncated_bytes = text_bytes[:max_bytes]
        
        # 不完全なUTF-8文字を避けるため、最後の有効な文字境界まで戻る
        while len(truncated_bytes) > 0:
            try:
                truncated_text = truncated_bytes.decode('utf-8')
                print(f"警告: キャプションが{max_bytes}バイトを超えたため切り詰めました。元のバイト数: {len(text_bytes)}, 切り詰め後: {len(truncated_bytes)}")
                return truncated_text
            except UnicodeDecodeError:
                # 最後のバイトを削除して再試行
                truncated_bytes = truncated_bytes[:-1]
        
        # 何らかの理由で全て削除された場合は空文字を返す
        return ""

    def is_image_registered(self, file_name):
        """画像が既にデータベースに登録されているかチェック"""
        def operation(conn):
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM IMAGES WHERE file_name = :1", [file_name])
                count = cursor.fetchone()[0]
                return count > 0
            finally:
                cursor.close()

        return self._execute_with_retry(operation)

    def insert_image_to_db(self, embedding_service, mllm_client, mllm_model_id, compartment_id, image_data, file_name, custom_prompt=None):
        """画像とその説明文をOracle Databaseに挿入"""
        # キャプションを生成
        caption = self.get_image_caption(mllm_client, image_data, mllm_model_id, compartment_id, custom_prompt)
        
        # VARCHAR2(4000)の制限を確実に適用（バイト数ベース）
        caption = self._truncate_caption_safely(caption, 4000)
        
        # 画像とテキストの埋め込みベクトルを取得
        pil_image = Image.open(BytesIO(image_data))
        image_embedding = array.array('f', embedding_service.get_image_embedding(pil_image))
        caption_embedding = array.array('f', embedding_service.get_text_embedding(caption, "search_document"))
        
        def operation(conn):
            cursor = conn.cursor()
            try:
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
        new_caption = self._truncate_caption_safely(new_caption, 4000)
        
        # 新しいキャプションの埋め込みベクトルを生成
        caption_embedding = array.array('f', embedding_service.get_text_embedding(new_caption, "search_document"))
        
        def operation(conn):
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
        def operation(conn):
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
        def operation(conn):
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
        def operation(conn):
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
                    if error.code in (29902, 30600) or 'DRG-' in str(e):
                        print(f"Oracle Text検索エラー: {e}")
                        print(f"問題のクエリ: {search_query}")

                        fallback_sql = """
                            SELECT a.image_id, a.file_name, a.caption, a.image_data,
                                0 as distance
                            FROM IMAGES a
                            WHERE UPPER(caption) LIKE UPPER(:1)
                            ORDER BY a.upload_date DESC
                            FETCH FIRST :2 ROWS ONLY
                        """

                        like_query = search_query.split(' AND ')[0].replace('"', '').strip()
                        like_pattern = f"%{like_query}%"

                        print(f"フォールバック検索を実行: {like_pattern}")
                        cursor.execute(fallback_sql, [like_pattern, top_k])

                        executed_sql = fallback_sql.replace(":1", f"'%{like_query}%'") \
                                                  .replace(":2", str(top_k))
                        executed_sql += (
                            f"\n-- 元のOracle Text検索でエラーが発生したため、LIKE検索にフォールバック"
                            f"\n-- 元のクエリ: {search_query}"
                        )
                    else:
                        raise
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
        def operation(conn):
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
        def operation(conn):
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
        def operation(conn):
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
        def operation(conn):
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT file_name FROM IMAGES WHERE image_id = :1", (image_id,))
                result = cursor.fetchone()

                if not result:
                    print(f"警告: 画像ID {image_id} が見つかりません")
                    return False

                file_name = result[0]
                cursor.execute("DELETE FROM IMAGES WHERE image_id = :1", (image_id,))
                conn.commit()

                print(f"画像ID {image_id} (ファイル名: {file_name}) を削除しました")
                return True
            finally:
                cursor.close()

        try:
            return self._execute_with_retry(operation)
        except Exception as e:
            print(f"画像削除エラー: {e}")
            return False 