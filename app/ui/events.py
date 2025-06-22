import gradio as gr
from PIL import Image
import math

class UIEvents:
    """UIイベントを管理するクラス"""
    
    def __init__(self, search_service):
        self.search_service = search_service
        
    def register_search_target_events(self, search_target, search_method, query_input, uploaded_image, query_examples, executed_sql_text):
        """検索対象変更時のイベントを登録"""
        search_target.change(
            fn=self.update_search_method_choices,
            inputs=[search_target],
            outputs=[search_method]
        ).then(
            fn=self.update_input_visibility,
            inputs=[search_target, search_method],
            outputs=[query_input, uploaded_image, query_examples]
        ).then(
            fn=self.update_sql_text_lines,
            inputs=[search_target],
            outputs=[executed_sql_text]
        )
        
    def register_search_method_events(self, search_method, query_input, uploaded_image, score_label, executed_query_text, execute_query_button, search_target, query_examples):
        """検索方法変更時のイベントを登録"""
        search_method.change(
            fn=self.update_input_visibility,
            inputs=[search_target, search_method],
            outputs=[query_input, uploaded_image, query_examples]
        ).then(
            fn=self.update_score_label,
            inputs=[search_method],
            outputs=[score_label]
        ).then(
            fn=self.update_query_text_interactivity,
            inputs=[search_method],
            outputs=[executed_query_text, execute_query_button]
        )
        
    def register_search_button_events(self, search_button, query_input, uploaded_image, search_target, search_method, top_k_slider, vector_threshold, keyword_threshold, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, execute_query_button, pagination_row):
        """検索ボタンのイベントを登録"""
        search_button.click(
            fn=self.clear_before_search,
            inputs=[],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text]
        ).then(
            fn=self.update_gallery_labels,
            inputs=[query_input, search_method, search_target],
            outputs=[vector_gallery, keyword_gallery]
        ).then(
            fn=self.search_service.search_images,
            inputs=[query_input, uploaded_image, search_target, search_method, top_k_slider, vector_threshold, keyword_threshold],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text]
        ).then(
            fn=self.update_query_text_interactivity,
            inputs=[search_method],
            outputs=[executed_query_text, execute_query_button]
        ).then(
            fn=self.hide_pagination,
            inputs=[],
            outputs=[pagination_row]
        )
        
    def register_execute_query_button_events(self, execute_query_button, executed_query_text, top_k_slider, keyword_threshold, vector_gallery, filename_text, similarity_text, caption_text, state, executed_query_text_out, executed_sql_text, pagination_row):
        """カスタムクエリ実行ボタンのイベントを登録"""
        execute_query_button.click(
            fn=self.clear_before_custom_search,
            inputs=[],
            outputs=[vector_gallery, filename_text, similarity_text, caption_text, state, executed_sql_text, executed_query_text_out]
        ).then(
            fn=self.execute_custom_query,
            inputs=[executed_query_text, top_k_slider, keyword_threshold],
            outputs=[vector_gallery, filename_text, similarity_text, caption_text, state, executed_query_text_out, executed_sql_text]
        ).then(
            fn=self.hide_pagination,
            inputs=[],
            outputs=[pagination_row]
        )
        
    def register_clear_button_events(self, clear_button, query_input, uploaded_image, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, pagination_row):
        """クリアボタンのイベントを登録"""
        clear_button.click(
            fn=self.clear_results,
            inputs=[],
            outputs=[query_input, uploaded_image, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text]
        ).then(
            fn=self.hide_pagination,
            inputs=[],
            outputs=[pagination_row]
        )
    
    def register_show_all_button_events(self, show_all_button, top_k_slider, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button):
        """全件表示ボタンのイベントを登録"""
        show_all_button.click(
            fn=self.clear_before_search,
            inputs=[],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text]
        ).then(
            fn=self.show_all_images,
            inputs=[top_k_slider, state],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button]
        )
    
    def register_pagination_events(self, prev_button, next_button, top_k_slider, vector_gallery, page_info, state, keyword_gallery, prev_button_out, next_button_out):
        """ページングボタンのイベントを登録"""
        prev_button.click(
            fn=self.prev_page,
            inputs=[top_k_slider, state],
            outputs=[vector_gallery, page_info, state, keyword_gallery, prev_button_out, next_button_out]
        )
        
        next_button.click(
            fn=self.next_page,
            inputs=[top_k_slider, state],
            outputs=[vector_gallery, page_info, state, keyword_gallery, prev_button_out, next_button_out]
        )
        
    def register_gallery_selection_events(self, vector_gallery, keyword_gallery, state, filename_text, similarity_text, caption_text):
        """ギャラリー選択イベントを登録"""
        def handle_vector_selection(evt: gr.SelectData, state_data):
            # vector_galleryを選択した場合の処理
            # print(f"ベクトル検索ギャラリーで選択: インデックス={evt.index}")
            
            # stateの構造を確認
            if state_data is None:
                print("警告: state_dataがNoneです")
                return "", "", "", gr.Gallery(selected_index=None)
                
            # state_dataから直接ベクトル検索結果を取得
            vector_results = state_data.get("vector_results", [])
            # print(f"ベクトル検索結果数: {len(vector_results)}")
            
            # インデックスが有効かチェック
            if len(vector_results) <= evt.index:
                print(f"警告: 無効なインデックス - vector_results長さ={len(vector_results)}, インデックス={evt.index}")
                return "", "", "", gr.Gallery(selected_index=None)
                
            # ベクトル検索結果を取得
            selected_result = vector_results[evt.index]
            # print(f"選択されたベクトル検索結果: {selected_result}")
            
            # 画像情報を表示
            file_name = selected_result['file_name']
            
            # スコアを表示（ベクトル検索なのでコサイン類似度に変換）
            score_text = ""
            if selected_result['distance'] is not None:
                score_text = f"{-1 * selected_result['distance']:.4f}"
            
            # キャプションを表示
            caption = self.search_service.normalize_newlines(selected_result['caption'])
            
            # ドキュメントに基づいた方法で、選択状態のみをリセットしたギャラリーコンポーネントを返す
            return file_name, score_text, caption, gr.Gallery(
                selected_index=None,
            )
            
        def handle_keyword_selection(evt: gr.SelectData, state_data):
            # keyword_galleryは常に全文検索結果を表示するギャラリーなので
            # このギャラリーで選択された画像は常に全文検索結果
            # print(f"全文検索ギャラリーで選択: インデックス={evt.index}")
            # print(f"イベント情報: {evt}")
            
            # state_dataの構造を確認
            if state_data is None:
                print("警告: state_dataがNoneです")
                return "", "", "", gr.Gallery(selected_index=None)
                
            # state_dataから直接全文検索結果を取得
            keyword_results = state_data.get("keyword_results", [])
            # print(f"全文検索結果数: {len(keyword_results)}")
            
            # 詳細なデバッグ情報
            # for idx, res in enumerate(keyword_results):
            #     print(f"全文検索結果[{idx}]: {res.get('file_name')}")
            
            # 全文検索結果が0件の場合
            if len(keyword_results) == 0:
                # print("警告: 全文検索結果が0件です")
                return "", "", "", gr.Gallery(selected_index=None)
                
            # evt.indexが全文検索結果の範囲内かチェック
            if evt.index >= len(keyword_results):
                print(f"警告: インデックスが範囲外です - インデックス={evt.index}, 結果数={len(keyword_results)}")
                # インデックスが範囲外の場合はエラーを返す
                return "", "", "", gr.Gallery(selected_index=None)
            
            try:
                # 選択された全文検索結果を直接取得
                selected_result = keyword_results[evt.index]
                # print(f"選択された全文検索結果: {selected_result.get('file_name')} (インデックス: {evt.index})")
                
                # 画像情報を表示
                file_name = selected_result['file_name']
                
                # スコアを表示
                score_text = ""
                if selected_result['distance'] is not None:
                    score_text = f"{selected_result['distance']:.4f}"
                
                # キャプションを表示
                caption = self.search_service.normalize_newlines(selected_result['caption'])
                
                # 選択を解除して返す
                return file_name, score_text, caption, gr.Gallery(selected_index=None)
            except Exception as e:
                print(f"エラー発生: {str(e)}")
                # エラーが発生した場合は空の値を返す
                return "", "", "", gr.Gallery(selected_index=None)
            
        vector_gallery.select(
            fn=handle_vector_selection,
            inputs=[state],
            outputs=[filename_text, similarity_text, caption_text, keyword_gallery]
        )
        
        keyword_gallery.select(
            fn=handle_keyword_selection,
            inputs=[state],
            outputs=[filename_text, similarity_text, caption_text, vector_gallery]
        )
        
    # イベントハンドラー関数
    def update_search_method_choices(self, search_target):
        """検索対象に応じて検索方法の選択肢を更新する関数"""
        if search_target == "キャプション":
            # キャプション検索の場合は選択肢を表示しない（ハイブリッド検索のみ）
            return gr.Radio(choices=["テキスト", "画像"], value="テキスト", label="検索方法", container=True, visible=False)
        else:  # 画像
            return gr.Radio(choices=["テキスト", "画像"], value="テキスト", label="検索方法", container=True, visible=True)
            
    def update_input_visibility(self, search_target, search_method):
        """検索方法に応じて入力フィールドの表示/非表示を切り替える関数"""
        if search_target == "画像" and search_method == "画像":
            return gr.Textbox(
                label="検索クエリ",
                placeholder="検索したい画像の内容を入力してください",
                visible=False
            ), gr.Image(
                label="画像をアップロード",
                type="pil",
                visible=True,
                height=300,
                width=300
            ), gr.update(visible=False)
        else:
            return gr.Textbox(
                label="検索クエリ",
                placeholder="検索したい画像の内容を入力してください",
                visible=True
            ), gr.Image(
                label="画像をアップロード",
                type="pil",
                visible=False,
                height=300,
                width=300
            ), gr.update(visible=True)
            
    def update_score_label(self, search_method):
        """検索方法に応じてスコアラベルを更新する関数"""
        if search_method == "全文検索":
            return gr.Markdown("**スコア：**")
        else:
            return gr.Markdown("**コサイン類似度：**")
            
    def update_query_text_interactivity(self, search_method):
        """検索方法に応じてクエリテキストボックスの編集可能性とボタンの表示を切り替える関数"""
        if search_method == "全文検索":
            return gr.Textbox(
                label="実行されたクエリ（編集できます。ヒント：AND、OR、\"...\"、ABOUT(\"...\")、NEAR((単語1, 単語2), 距離)）",
                show_label=True,
                interactive=True,
                container=True,
                scale=4,
                lines=2
            ), gr.Button("このクエリを実行", visible=True, scale=1)
        elif search_method == "ハイブリッド検索":
            return gr.Textbox(
                label="実行されたクエリ",
                show_label=True,
                interactive=False,
                container=True,
                scale=4,
                lines=4
            ), gr.Button("このクエリを実行", visible=False, scale=1)
        else:
            return gr.Textbox(
                label="実行されたクエリ",
                show_label=True,
                interactive=False,
                container=True,
                scale=4,
                lines=2
            ), gr.Button("このクエリを実行", visible=False, scale=1)
            
    def update_gallery_labels(self, query, search_method, search_target):
        """クエリの状態に応じてギャラリーのラベルを更新する関数"""
        if not query or query.strip() == "":
            return gr.Gallery(label="最近アップロードされた画像", visible=True), gr.Gallery(label="", visible=False)
        elif search_target == "キャプション":
            # キャプション検索の場合は常にベクトル検索と全文検索のラベルを表示
            return gr.Gallery(label="ベクトル検索", visible=True), gr.Gallery(label="全文検索", visible=True)
        elif search_method == "テキスト":
            return gr.Gallery(label="テキスト検索", visible=True), gr.Gallery(label="", visible=False)
        elif search_method == "画像":
            return gr.Gallery(label="画像検索", visible=True), gr.Gallery(label="", visible=False)
        else:
            return gr.Gallery(label="検索結果", visible=True), gr.Gallery(label="", visible=False)
            
    def clear_before_search(self):
        """検索前にギャラリーとテキスト表示をクリアする関数"""
        # ギャラリーのコンテンツはクリアするが、表示設定は変更しない
        return [], [], "", "", "", {
            "current_page": 1,
            "total_pages": 1,
            "page_size": 0,
            "total_image_count": 0,
            "all_images": [],
            "combined_results": [],
            "vector_results": [],
            "keyword_results": []
        }, "", ""  # vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text
        
    def clear_results(self):
        """検索結果とフォームをクリアする関数"""
        # すべてのフィールドをクリア
        return "", None, [], [], "", "", "", {
            "current_page": 1,
            "total_pages": 1,
            "page_size": 0,
            "total_image_count": 0,
            "all_images": [],
            "combined_results": [],
            "vector_results": [],
            "keyword_results": []
        }, "", ""  # query_input, uploaded_image, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text
        
    def clear_before_custom_search(self):
        """カスタム検索前に結果表示をクリアするがクエリテキストは維持する関数"""
        # 表示フィールドのみクリア（クエリテキストは保持）
        return [], "", "", "", {
            "current_page": 1,
            "total_pages": 1,
            "page_size": 0,
            "total_image_count": 0,
            "all_images": [],
            "combined_results": [],
            "vector_results": [],
            "keyword_results": []
        }, "", ""  # vector_gallery, filename_text, similarity_text, caption_text, state, executed_sql_text, executed_query_text
    
    def show_all_images(self, top_k, state_data=None):
        """全件表示ボタンの処理を行う関数"""
        # state_dataがNoneの場合は初期化
        if state_data is None:
            state_data = {
                "current_page": 1,
                "total_pages": 1,
                "page_size": 0,
                "total_image_count": 0,
                "all_images": [],
                "combined_results": [],
                "vector_results": [],
                "keyword_results": []
            }
            
        # 総画像数を取得
        total_image_count = self.search_service.database_service.get_total_image_count()
        
        # スライダーで設定された数分だけ最新の画像を取得
        current_page = 1
        page_size = top_k
        
        # 1ページ目のデータを取得
        results, executed_sql = self.search_service.database_service.get_recent_images(top_k, (current_page - 1) * top_k)
        
        # 結果を保存
        all_images = results
        
        # 総ページ数を計算
        total_pages = math.ceil(total_image_count / top_k) if top_k > 0 else 1
        
        # 状態を更新
        state_data.update({
            "current_page": current_page,
            "total_pages": total_pages,
            "page_size": page_size,
            "total_image_count": total_image_count,
            "all_images": all_images,
            "combined_results": results,
            "vector_results": results,
            "keyword_results": []
        })
        
        # 結果を整形
        output_images = []
        for result in results:
            if isinstance(result['image'], Image.Image):
                output_images.append(result['image'])
            
        if results:
            first_result = results[0]
            
            # ページング情報を更新
            page_info_text = f"{current_page}/{total_pages} ページ（総合計 {total_image_count} 枚）"
            
            # ページングボタンの状態を更新
            prev_button, next_button = self.update_pagination_buttons(state_data)
            
            return (
                gr.Gallery(label="全件表示", value=output_images),  # vector_gallery - ラベルを「全件表示」に変更
                gr.Gallery(visible=False),  # keyword_gallery - 非表示に設定
                first_result['file_name'],  # filename_text
                "",  # similarity_text
                self.search_service.normalize_newlines(first_result['caption']),  # caption_text
                state_data,  # state
                "（全件表示）",  # executed_query_text
                executed_sql,  # executed_sql_text
                gr.update(visible=True),  # pagination_row
                page_info_text,  # page_info
                prev_button,  # prev_button
                next_button   # next_button
            )
        
        return [], gr.Gallery(visible=False), "", "", "", {
            "current_page": 1,
            "total_pages": 0,
            "page_size": top_k,
            "total_image_count": 0,
            "all_images": [],
            "combined_results": [],
            "vector_results": [],
            "keyword_results": []
        }, "（画像が見つかりません）", executed_sql, gr.update(visible=False), "0/0 ページ", gr.update(interactive=False), gr.update(interactive=False)
    
    def prev_page(self, top_k, state_data=None):
        """前のページに移動する関数"""
        # state_dataがNoneの場合は初期化
        if state_data is None:
            state_data = {
                "current_page": 1,
                "total_pages": 1,
                "page_size": 0,
                "total_image_count": 0,
                "all_images": [],
                "combined_results": [],
                "vector_results": [],
                "keyword_results": []
            }
            return gr.Gallery(label="全件表示", value=[]), "0/0 ページ", state_data, gr.Gallery(visible=False), gr.update(interactive=False), gr.update(interactive=False)
        
        # 現在の状態を取得
        current_page = state_data.get("current_page", 1)
        total_pages = state_data.get("total_pages", 1)
        total_image_count = state_data.get("total_image_count", 0)
        all_images = state_data.get("all_images", [])
        
        # 前のページに移動（1ページ目より前には行かない）
        if current_page > 1:
            current_page -= 1
            
            # 前のページのデータを取得
            results, _ = self.search_service.database_service.get_recent_images(top_k, (current_page - 1) * top_k)
            all_images = results
            
            # 状態を更新
            state_data.update({
                "current_page": current_page,
                "all_images": all_images
            })
        
        # 結果を整形
        output_images = []
        for result in all_images:
            if isinstance(result['image'], Image.Image):
                output_images.append(result['image'])
        
        # ページング情報を更新
        page_info_text = f"{current_page}/{total_pages} ページ（総合計 {total_image_count} 枚）"
        
        # 状態データを更新
        state_data.update({
            "combined_results": all_images,
            "vector_results": all_images,
            "keyword_results": []
        })
        
        # ページングボタンの状態を更新
        prev_button, next_button = self.update_pagination_buttons(state_data)
        
        # 選択状態をリセットしたギャラリーを返す
        return gr.Gallery(label="全件表示", value=output_images, selected_index=None), page_info_text, state_data, gr.Gallery(visible=False), prev_button, next_button
    
    def next_page(self, top_k, state_data=None):
        """次のページに移動する関数"""
        # state_dataがNoneの場合は初期化
        if state_data is None:
            state_data = {
                "current_page": 1,
                "total_pages": 1,
                "page_size": 0,
                "total_image_count": 0,
                "all_images": [],
                "combined_results": [],
                "vector_results": [],
                "keyword_results": []
            }
            return gr.Gallery(label="全件表示", value=[]), "0/0 ページ", state_data, gr.Gallery(visible=False), gr.update(interactive=False), gr.update(interactive=False)
        
        # 現在の状態を取得
        current_page = state_data.get("current_page", 1)
        total_pages = state_data.get("total_pages", 1)
        total_image_count = state_data.get("total_image_count", 0)
        all_images = state_data.get("all_images", [])
        
        # 次のページに移動（最大ページ数を超えないように）
        if current_page < total_pages:
            current_page += 1
            
            # 次のページのデータを取得
            results, _ = self.search_service.database_service.get_recent_images(top_k, (current_page - 1) * top_k)
            
            # 結果がない場合は前のページに戻る
            if not results:
                current_page -= 1
                page_info_text = f"{current_page}/{total_pages} ページ（総合計 {total_image_count} 枚）（最終ページ）"
                
                # 状態を更新
                state_data.update({
                    "current_page": current_page
                })
                
                # 選択状態をリセットしたギャラリーを返す
                output_images = []
                for result in all_images:
                    if isinstance(result['image'], Image.Image):
                        output_images.append(result['image'])
                
                # ページングボタンの状態を更新
                prev_button, next_button = self.update_pagination_buttons(state_data)
                
                return gr.Gallery(label="全件表示", value=output_images, selected_index=None), page_info_text, state_data, gr.Gallery(visible=False), prev_button, next_button
            
            # 結果がある場合は更新
            all_images = results
            
            # 状態を更新
            state_data.update({
                "current_page": current_page,
                "all_images": all_images
            })
        
        # 結果を整形
        output_images = []
        for result in all_images:
            if isinstance(result['image'], Image.Image):
                output_images.append(result['image'])
        
        # ページング情報を更新
        page_info_text = f"{current_page}/{total_pages} ページ（総合計 {total_image_count} 枚）"
        
        # 状態データを更新
        state_data.update({
            "combined_results": all_images,
            "vector_results": all_images,
            "keyword_results": []
        })
        
        # ページングボタンの状態を更新
        prev_button, next_button = self.update_pagination_buttons(state_data)
        
        # 選択状態をリセットしたギャラリーを返す
        return gr.Gallery(label="全件表示", value=output_images, selected_index=None), page_info_text, state_data, gr.Gallery(visible=False), prev_button, next_button
    
    def hide_pagination(self):
        """ページング用UIを非表示にする関数"""
        return gr.update(visible=False)
        
    def update_pagination_buttons(self, state_data):
        """ページングボタンの状態を更新する関数"""
        # 1ページ目では「前へ」ボタンをグレーアウト
        # 最終ページでは「次へ」ボタンをグレーアウト
        prev_interactive = state_data["current_page"] > 1
        next_interactive = state_data["current_page"] < state_data["total_pages"]
        
        return gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive)
        
    def show_selected_image_info(self, evt: gr.SelectData, results):
        """選択された画像の情報を表示する関数"""
        if results is None or evt.index >= len(results):
            return "", "", ""
        
        result = results[evt.index]
        # print(f"選択された画像のインデックス: {evt.index}")
        # print(f"選択された画像の情報: {result}")
        # distanceがNULLの場合（クエリーが空の場合）はスコアを表示しない
        score_text = ""
        if result['distance'] is not None:
            # 全文検索の場合はスコアをそのまま表示、ベクトル検索の場合はコサイン類似度に変換
            score_value = result['distance'] if result.get('search_mode') == "全文検索" else -1 * result['distance']
            score_text = f"{score_value:.4f}"
        
        return (
            result['file_name'],
            score_text,
            self.search_service.normalize_newlines(result['caption'])
        )
        
    def execute_custom_query(self, custom_query, top_k=5, keyword_threshold=0):
        """カスタム検索クエリを実行する関数"""
        results, executed_sql = self.search_service.database_service.search_by_fulltext(
            custom_query, top_k, keyword_threshold
        )
        
        # 結果を整形
        output_images = []
        for result in results:
            if isinstance(result['image'], Image.Image):
                output_images.append(result['image'])
            else:
                print(f"警告: 画像が期待された形式ではありません: {type(result['image'])}")
        
        if results:
            first_result = results[0]
            # スコアを表示
            score_text = ""
            if first_result['distance'] is not None:
                score_text = f"{first_result['distance']:.4f}"
            
            return output_images, first_result['file_name'], score_text, self.search_service.normalize_newlines(first_result['caption']), results, custom_query, executed_sql 

    def update_sql_text_lines(self, search_target):
        """検索対象に応じてSQLテキストボックスの行数を更新する関数"""
        if search_target == "キャプション":
            # キャプション検索の場合は16行表示
            return gr.Textbox(
                label="実行されたSQL",
                show_label=True,
                interactive=False,
                lines=16,
                show_copy_button=True,
                container=True
            )
        else:
            # 画像検索の場合は8行表示
            return gr.Textbox(
                label="実行されたSQL",
                show_label=True,
                interactive=False,
                lines=8,
                show_copy_button=True,
                container=True
            ) 