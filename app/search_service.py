import array
import re
from PIL import Image

class SearchService:
    def __init__(self, embedding_service, database_service, search_query_generator):
        self.embedding_service = embedding_service
        self.database_service = database_service
        self.search_query_generator = search_query_generator
        
    def normalize_newlines(self, text):
        """3つ以上連続する改行を2つの改行に変換する"""
        if text is None:
            return ""
        return re.sub(r'\n{3,}', '\n\n', str(text))
        
    def search_by_caption(self, query, search_mode="ベクトル検索", top_k=5, vector_threshold=0.5, keyword_threshold=10):
        """テキストクエリに基づいて画像のキャプションを検索"""
        executed_query = query  # デフォルトはオリジナルのクエリ
        
        # クエリーが空の場合
        if not query.strip():
            results, executed_sql = self.database_service.get_recent_images(top_k, 0)  # オフセット0で取得
            executed_query = "（クエリが空のためアップロード日時が新しい順に画像を表示しています）"
            return results, executed_query, executed_sql
            
        elif search_mode == "ベクトル検索":
            # クエリの埋め込みベクトルを取得
            query_embedding = array.array('f', self.embedding_service.get_text_embedding(query, "search_query"))
            
            # ベクトル類似度検索を実行
            results, executed_sql = self.database_service.search_by_caption_vector(
                query_embedding, top_k, vector_threshold
            )
            return results, executed_query, executed_sql
            
        else: # 全文検索
            # 検索クエリーを生成
            search_query = self.search_query_generator.generate(query)
            executed_query = search_query
            
            # 全文検索を実行
            results, executed_sql = self.database_service.search_by_fulltext(
                search_query, top_k, keyword_threshold
            )
            return results, executed_query, executed_sql
            
    def search_by_image_text(self, query, top_k=5, vector_threshold=0.5):
        """テキストクエリに基づいて画像の画像ベクトルを検索"""
        executed_query = query  # デフォルトはオリジナルのクエリ
        
        # クエリーが空の場合
        if not query.strip():
            results, executed_sql = self.database_service.get_recent_images(top_k, 0)  # オフセット0で取得
            executed_query = "（空のクエリ）"
            return results, executed_query, executed_sql
            
        else:
            # クエリのテキスト埋め込みベクトルを取得（画像検索用）
            query_embedding = array.array('f', self.embedding_service.get_text_embedding(query, "search_query"))
            
            # 画像ベクトルに対するベクトル類似度検索を実行
            results, executed_sql = self.database_service.search_by_image_vector(
                query_embedding, top_k, vector_threshold
            )
            return results, executed_query, executed_sql
            
    def search_by_image_embedding(self, uploaded_image, top_k=5, vector_threshold=0.5):
        """アップロードされた画像から画像ベクトル検索を実行"""
        if uploaded_image is None:
            return [], "（画像がアップロードされていません）", ""
            
        # アップロードされた画像の埋め込みベクトルを取得
        image_embedding = array.array('f', self.embedding_service.get_image_embedding(uploaded_image))
        
        # 画像ベクトルに対するベクトル類似度検索を実行
        results, executed_sql = self.database_service.search_by_image_vector(
            image_embedding, top_k, vector_threshold
        )
        return results, "（アップロードされた画像）", executed_sql
        
    def hybrid_search(self, query, top_k=5, vector_threshold=0.5, keyword_threshold=10):
        """ベクトル検索と全文検索の結果を統合する"""
        # ベクトル検索の実行
        vector_results, vector_query, vector_sql = self.search_by_caption(
            query, "ベクトル検索", top_k, vector_threshold, 0
        )
        
        # 全文検索の実行
        keyword_results, keyword_query, keyword_sql = self.search_by_caption(
            query, "全文検索", top_k, 0, keyword_threshold
        )
        
        # print(f"ハイブリッド検索 - ベクトル検索結果数: {len(vector_results)}, 全文検索結果数: {len(keyword_results)}")
        
        # ベクトル検索と全文検索の両方で結果が0件の場合、最新の画像を返す
        if len(vector_results) == 0 and len(keyword_results) == 0:
            # print("両方の検索結果が0件のため、最近の画像を表示します")
            vector_results, _, _ = self.search_by_caption("", "ベクトル検索", top_k, 0, 0)
        
        # 結果の統合（重複を除去しつつ、両方の検索結果を保持）
        combined_results = []
        seen_ids = set()
        
        for result in vector_results + keyword_results:
            if result['image_id'] not in seen_ids:
                seen_ids.add(result['image_id'])
                combined_results.append(result)
        
        return (
            combined_results, 
            vector_results, 
            keyword_results, 
            f"ベクトル検索: {vector_query}\n全文検索: {keyword_query}", 
            f"ベクトル検索: {vector_sql}\n全文検索: {keyword_sql}"
        )
        
    def search_images(self, query, uploaded_image, search_target, search_method, top_k=5, vector_threshold=0.5, keyword_threshold=0):
        """Gradioインターフェース用の統合検索関数"""
        results = []
        executed_query = ""
        executed_sql = ""
        
        # クエリが空の場合や、適切な検索対象/検索方法がない場合は最近の画像を表示
        if (not query or query.strip() == "") and (uploaded_image is None):
            # キャプション検索の場合、検索方法は考慮せずに最近の画像を表示
            search_mode = "ハイブリッド検索" if search_target == "キャプション" else search_method
            results, executed_query, executed_sql = self.search_by_caption(
                query, search_mode, top_k, vector_threshold, keyword_threshold
            )
            
            # 検索結果を整形
            output_images = []
            for result in results:
                if isinstance(result['image'], Image.Image):
                    output_images.append(result['image'])
            
            if results:
                first_result = results[0]
                # distanceがNULLの場合（クエリーが空の場合）はスコアを表示しない
                score_text = ""
                if first_result['distance'] is not None:
                    # 全文検索の場合はスコアをそのまま表示、ベクトル検索の場合はコサイン類似度に変換
                    score_value = first_result['distance'] if search_method == "全文検索" else -1 * first_result['distance']
                    score_text = f"{score_value:.4f}"
                
                # 空のクエリーの場合は、選択された検索方法の結果エリアにのみ表示
                vector_gallery_images = output_images if search_method in ["ベクトル検索", "ハイブリッド検索", "テキスト", "画像"] else []
                keyword_gallery_images = output_images if search_method == "全文検索" else []
                
                return (
                    vector_gallery_images,  # vector_gallery
                    keyword_gallery_images,  # keyword_gallery
                    first_result['file_name'],  # filename_text
                    score_text,  # similarity_text
                    self.normalize_newlines(first_result['caption']),  # caption_text
                    {"combined_results": results, "vector_results": results, "keyword_results": []},  # state - 全文検索結果も含める
                    executed_query,  # executed_query_text
                    executed_sql  # executed_sql_text
                )
            return [], [], "", "", "", [], executed_query, executed_sql
        
        if search_target == "キャプション":
            # キャプション検索の場合は常にハイブリッド検索を使用
            combined_results, vector_results, keyword_results, executed_query, executed_sql = self.hybrid_search(
                query, top_k, vector_threshold, keyword_threshold
            )
            
            # ベクトル検索と全文検索の結果をそれぞれのギャラリー用に整形
            vector_images = []
            keyword_images = []
            
            # print(f"ベクトル検索結果: {len(vector_results)}件")
            # print(f"全文検索結果: {len(keyword_results)}件")
            
            # ベクトル検索結果の画像を抽出
            for result in vector_results:
                if isinstance(result['image'], Image.Image):
                    vector_images.append(result['image'])
                    
            # 全文検索結果の画像を抽出
            for result in keyword_results:
                if isinstance(result['image'], Image.Image):
                    keyword_images.append(result['image'])
                    
            # print(f"検索結果サマリー - ベクトルギャラリー: {len(vector_images)}枚, 全文検索ギャラリー: {len(keyword_images)}枚")
            # print(f"全文検索結果ファイル名: {[r.get('file_name') for r in keyword_results]}")
            
            if combined_results:
                first_result = combined_results[0]
                # distanceがNULLの場合（クエリーが空の場合）はスコアを表示しない
                score_text = ""
                if first_result['distance'] is not None:
                    # 全文検索の場合はスコアをそのまま表示、ベクトル検索の場合はコサイン類似度に変換
                    score_value = first_result['distance'] if first_result.get('search_mode') == "全文検索" else -1 * first_result['distance']
                    score_text = f"{score_value:.4f}"
                
                return (
                    vector_images,  # vector_gallery
                    keyword_images,  # keyword_gallery
                    first_result['file_name'],  # filename_text
                    score_text,  # similarity_text
                    self.normalize_newlines(first_result['caption']),  # caption_text
                    {"combined_results": combined_results, "vector_results": vector_results, "keyword_results": keyword_results},  # state - 全文検索結果も含める
                    executed_query,  # executed_query_text
                    executed_sql  # executed_sql_text
                )
            return [], [], "", "", "", [], executed_query, executed_sql
            
        elif search_target == "画像":
            if search_method == "テキスト":
                results, executed_query, executed_sql = self.search_by_image_text(
                    query, top_k, vector_threshold
                )
            elif search_method == "画像":
                if uploaded_image is not None:
                    results, executed_query, executed_sql = self.search_by_image_embedding(
                        uploaded_image, top_k, vector_threshold
                    )
                else:
                    # 画像がアップロードされていない場合は最近の画像を表示
                    results, executed_sql = self.database_service.get_recent_images(top_k, 0)
                    executed_query = "（画像がアップロードされていないため、最近アップロードされた画像を表示しています）"
            
            # 結果を整形
            output_images = []
            for result in results:
                if isinstance(result['image'], Image.Image):
                    output_images.append(result['image'])
            
            if results:
                first_result = results[0]
                # distanceがNULLの場合（クエリーが空の場合）はスコアを表示しない
                score_text = ""
                if first_result['distance'] is not None:
                    # 全文検索の場合はスコアをそのまま表示、ベクトル検索の場合はコサイン類似度に変換
                    score_value = first_result['distance'] if search_method == "全文検索" else -1 * first_result['distance']
                    score_text = f"{score_value:.4f}"
                
                return (
                    output_images,  # vector_gallery - 画像検索結果は常にベクトルギャラリーに表示
                    [],  # keyword_gallery
                    first_result['file_name'],  # filename_text
                    score_text,  # similarity_text
                    self.normalize_newlines(first_result['caption']),  # caption_text
                    {"combined_results": results, "vector_results": results, "keyword_results": []},  # state
                    executed_query,  # executed_query_text
                    executed_sql  # executed_sql_text
                )
            return [], [], "", "", "", {"combined_results": [], "vector_results": [], "keyword_results": []}, executed_query, executed_sql
            
    def load_recent_images(self, top_k=12):
        """アプリケーション起動時に最近アップロードされた画像を表示する関数"""
        results, executed_sql = self.database_service.get_recent_images(top_k, 0)  # オフセット0で取得
        
        # 結果を整形
        output_images = []
        for result in results:
            if isinstance(result['image'], Image.Image):
                output_images.append(result['image'])
            
        if results:
            first_result = results[0]
            return (
                output_images,  # vector_gallery
                [],  # keyword_gallery
                first_result['file_name'],  # filename_text
                "",  # similarity_text
                self.normalize_newlines(first_result['caption']),  # caption_text
                {"combined_results": results, "vector_results": results, "keyword_results": []},  # state
                "（最近のアップロード）",  # executed_query_text
                executed_sql  # executed_sql_text
            )
        return [], [], "", "", "", {"combined_results": [], "vector_results": [], "keyword_results": []}, "（画像が見つかりません）", executed_sql 