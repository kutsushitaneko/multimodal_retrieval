import time
from io import BytesIO
from PIL import Image
import oracledb

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
                    cursor.execute(sql, [search_query, keyword_threshold, top_k])
                    
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