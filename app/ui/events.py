import gradio as gr
from PIL import Image
import math
import time
from io import BytesIO
from app.vlm_service import VLMService

class UIEvents:
    """UIイベントを管理するクラス"""
    
    def __init__(self, search_service):
        self.search_service = search_service
        self.vlm_service = VLMService()
        
    def register_vlm_settings_events(self, vlm_service_provider, vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region, vlm_status_message):
        """VLM設定のイベントを登録"""
        # サービスプロバイダー変更時のイベント
        vlm_service_provider.change(
            fn=self.vlm_service_provider_changed,
            inputs=[vlm_service_provider],
            outputs=[vlm_model, vlm_max_tokens, vlm_oci_region],
            queue=False  # VLM設定変更は即座に処理
        )
        
        # VLMモデル変更時のイベント
        vlm_model.change(
            fn=self.vlm_model_changed,
            inputs=[vlm_model],
            outputs=[vlm_max_tokens, vlm_oci_region],
            queue=False  # VLMモデル変更は即座に処理
        )
        
    def vlm_service_provider_changed(self, service_provider):
        """VLMサービスプロバイダー変更時の処理"""
        return self.vlm_service.service_provider_changed(service_provider)
    
    def vlm_model_changed(self, model):
        """VLMモデル変更時の処理"""
        return self.vlm_service.model_changed(model)
        
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
        
    def register_search_method_events(self, search_method, query_input, uploaded_image, similarity_text, executed_query_text, execute_query_button, search_target, query_examples, morphological_analysis_text):
        """クエリーの種類変更時のイベントを登録"""
        search_method.change(
            fn=self.update_input_visibility,
            inputs=[search_target, search_method],
            outputs=[query_input, uploaded_image, query_examples]
        ).then(
            fn=self.update_score_label,
            inputs=[search_method],
            outputs=[similarity_text]
        ).then(
            fn=self.update_query_text_interactivity,
            inputs=[search_method],
            outputs=[executed_query_text, execute_query_button]
        ).then(
            fn=lambda search_target, search_method: self.update_morphological_analysis_visibility(search_target, search_method, ""),
            inputs=[search_target, search_method],
            outputs=[morphological_analysis_text]
        )
        
    def register_search_button_events(self, search_button, query_input, uploaded_image, search_target, search_method, top_k_slider, vector_threshold, keyword_threshold, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, execute_query_button, pagination_row, morphological_analysis_text):
        """検索ボタンのイベントを登録"""
        search_button.click(
            fn=self.clear_before_search,
            inputs=[],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, morphological_analysis_text]
        ).then(
            fn=self.update_gallery_labels,
            inputs=[query_input, search_method, search_target],
            outputs=[vector_gallery, keyword_gallery]
        ).then(
            fn=self.search_service.search_images,
            inputs=[query_input, uploaded_image, search_target, search_method, top_k_slider, vector_threshold, keyword_threshold],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, morphological_analysis_text]
        ).then(
            fn=self.update_query_text_interactivity,
            inputs=[search_method],
            outputs=[executed_query_text, execute_query_button]
        ).then(
            fn=self.update_morphological_analysis_result,
            inputs=[search_target, morphological_analysis_text],
            outputs=[morphological_analysis_text]
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
        
    def register_clear_button_events(self, clear_button, query_input, uploaded_image, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, pagination_row, morphological_analysis_text):
        """クリアボタンのイベントを登録"""
        clear_button.click(
            fn=self.clear_results,
            inputs=[],
            outputs=[query_input, uploaded_image, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, morphological_analysis_text]
        ).then(
            fn=self.hide_pagination,
            inputs=[],
            outputs=[pagination_row]
        )
    
    def register_show_all_button_events(self, show_all_button, top_k_slider, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button, morphological_analysis_text):
        """全件表示ボタンのイベントを登録"""
        show_all_button.click(
            fn=self.clear_before_search,
            inputs=[],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, morphological_analysis_text]
        ).then(
            fn=self.show_all_images,
            inputs=[top_k_slider, state],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button, morphological_analysis_text]
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
                return "", gr.Textbox(show_label=True, label="コサイン類似度", interactive=False, container=True, show_copy_button=True, value=""), "", gr.Gallery(selected_index=None)
                
            # state_dataから直接ベクトル検索結果を取得
            vector_results = state_data.get("vector_results", [])
            # print(f"ベクトル検索結果数: {len(vector_results)}")
            
            # インデックスが有効かチェック
            if len(vector_results) <= evt.index:
                print(f"警告: 無効なインデックス - vector_results長さ={len(vector_results)}, インデックス={evt.index}")
                return "", gr.Textbox(show_label=True, label="コサイン類似度", interactive=False, container=True, show_copy_button=True, value=""), "", gr.Gallery(selected_index=None)
                
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
            
            # ベクトル検索の場合はコサイン類似度のラベルで表示
            similarity_textbox = gr.Textbox(show_label=True, label="コサイン類似度", interactive=False, container=True, show_copy_button=True, value=score_text)
            
            # ドキュメントに基づいた方法で、選択状態のみをリセットしたギャラリーコンポーネントを返す
            return file_name, similarity_textbox, caption, gr.Gallery(
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
                return "", gr.Textbox(show_label=True, label="スコア", interactive=False, container=True, show_copy_button=True, value=""), "", gr.Gallery(selected_index=None)
                
            # state_dataから直接全文検索結果を取得
            keyword_results = state_data.get("keyword_results", [])
            # print(f"全文検索結果数: {len(keyword_results)}")
            
            # 詳細なデバッグ情報
            # for idx, res in enumerate(keyword_results):
            #     print(f"全文検索結果[{idx}]: {res.get('file_name')}")
            
            # 全文検索結果が0件の場合
            if len(keyword_results) == 0:
                # print("警告: 全文検索結果が0件です")
                return "", gr.Textbox(show_label=True, label="スコア", interactive=False, container=True, show_copy_button=True, value=""), "", gr.Gallery(selected_index=None)
                
            # evt.indexが全文検索結果の範囲内かチェック
            if evt.index >= len(keyword_results):
                print(f"警告: インデックスが範囲外です - インデックス={evt.index}, 結果数={len(keyword_results)}")
                # インデックスが範囲外の場合はエラーを返す
                return "", gr.Textbox(show_label=True, label="スコア", interactive=False, container=True, show_copy_button=True, value=""), "", gr.Gallery(selected_index=None)
            
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
                
                # 全文検索の場合はスコアのラベルで表示
                similarity_textbox = gr.Textbox(show_label=True, label="スコア", interactive=False, container=True, show_copy_button=True, value=score_text)
                
                # 選択を解除して返す
                return file_name, similarity_textbox, caption, gr.Gallery(selected_index=None)
            except Exception as e:
                print(f"エラー発生: {str(e)}")
                # エラーが発生した場合は空の値を返す
                return "", gr.Textbox(show_label=True, label="スコア", interactive=False, container=True, show_copy_button=True, value=""), "", gr.Gallery(selected_index=None)
            
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
        """検索対象に応じてクエリーの種類の選択肢を更新する関数"""
        if search_target == "キャプション":
            # キャプション検索の場合は選択肢を表示しない（ハイブリッド検索のみ）
            return gr.Radio(choices=["テキスト", "画像"], value="テキスト", label="クエリーの種類", container=True, visible=False)
        else:  # 画像
            return gr.Radio(choices=["テキスト", "画像"], value="テキスト", label="クエリーの種類", container=True, visible=True)
            
    def update_input_visibility(self, search_target, search_method):
        """クエリーの種類に応じて入力フィールドの表示/非表示を切り替える関数"""
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
        """クエリーの種類に応じてスコアラベルを更新する関数"""
        if search_method == "全文検索":
            return gr.Textbox(show_label=True, label="スコア", interactive=False, container=True, show_copy_button=True)
        else:
            return gr.Textbox(show_label=True, label="コサイン類似度", interactive=False, container=True, show_copy_button=True)
            
    def update_query_text_interactivity(self, search_method):
        """クエリーの種類に応じてクエリテキストボックスの編集可能性とボタンの表示を切り替える関数"""
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
            return gr.Gallery(label="テキストベクトルによる画像検索", visible=True), gr.Gallery(label="", visible=False)
        elif search_method == "画像":
            return gr.Gallery(label="画像ベクトルによる画像検索", visible=True), gr.Gallery(label="", visible=False)
        else:
            return gr.Gallery(label="検索結果", visible=True), gr.Gallery(label="", visible=False)
            
    def clear_before_search(self):
        """検索実行前にクリアする関数"""
        return [], [], "", "", "", {"combined_results": [], "vector_results": [], "keyword_results": []}, "", "", ""

    def clear_results(self):
        """すべての結果をクリアする関数"""
        return "", None, [], [], "", "", "", {"combined_results": [], "vector_results": [], "keyword_results": []}, "", "", ""
        
    def clear_before_custom_search(self):
        """カスタム検索実行前にクリアする関数"""
        return [], "", "", "", {"combined_results": [], "vector_results": [], "keyword_results": []}, "", ""
        
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
                next_button,   # next_button
                gr.Textbox(visible=False)  # morphological_analysis_text - 全件表示では非表示
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
        }, "（画像が見つかりません）", executed_sql, gr.update(visible=False), "0/0 ページ", gr.update(interactive=False), gr.update(interactive=False), gr.Textbox(visible=False)
    
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

    def update_morphological_analysis_visibility(self, search_target, search_method, morphological_analysis=""):
        """形態素解析結果の表示/非表示を制御する関数"""
        # 形態素解析結果があり、かつキャプション検索の場合のみ表示
        should_show = bool(morphological_analysis and morphological_analysis.strip())
        return gr.Markdown(
            label="全文検索：形態素解析結果",
            show_label=True,
            container=True,
            visible=should_show,
            value=morphological_analysis if should_show else "",
            elem_id="morphological_analysis"
        )
        
    def update_morphological_analysis_result(self, search_target, morphological_analysis):
        """検索結果の形態素解析結果を処理する関数"""
        # キャプション検索で形態素解析結果がある場合のみ表示
        should_show = search_target == "キャプション" and bool(morphological_analysis and morphological_analysis.strip())
        return gr.Markdown(
            label="全文検索：形態素解析結果",
            show_label=True,
            container=True,
            visible=should_show,
            value=morphological_analysis if should_show else "",
            elem_id="morphological_analysis"
        )

    def register_upload_edit_events(self, upload_image, filename_input, generate_caption_button, search_image_button, clear_button_upload,
                                    display_image, generated_caption, editable_caption, regenerate_caption_button, 
                                    update_database_button, cancel_edit_button, status_message, image_id_state, original_caption_state,
                                    delete_accordion, confirm_delete_checkbox, delete_button,
                                    prompt_template_dropdown, current_prompt_display, prompt_edit_textbox, 
                                    prompt_name_input, save_prompt_button, cancel_prompt_edit_button, prompt_status_message,
                                    confirm_prompt_delete_checkbox, delete_prompt_button,
                                    vlm_service_provider, vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region, vlm_status_message):
        """アップロード・編集機能のイベントを登録"""
        # アップロード時の処理（Gradioのバグ対策でqueue=Falseを設定）
        upload_image.upload(
            fn=self.extract_filename_from_upload,
            inputs=[upload_image],
            outputs=[filename_input],
            show_progress=True,
            queue=False  # ファイルアップロードは即座に処理
        ).then(
            fn=self.update_upload_button_states_on_upload,
            inputs=[upload_image, filename_input],
            outputs=[generate_caption_button, search_image_button, filename_input],
            queue=False  # ボタン状態更新も即座に処理
        ).then(
            fn=self.display_uploaded_image,
            inputs=[upload_image],
            outputs=[display_image],
            queue=False  # 画像表示も即座に処理
        )
        
        # ファイル名入力時の処理
        filename_input.change(
            fn=self.update_upload_button_states_on_filename_input,
            inputs=[upload_image, filename_input],
            outputs=[generate_caption_button, search_image_button, filename_input],
            queue=False  # ボタン状態更新は即座に処理
        )
        
        # キャプション生成ボタンの処理
        generate_caption_button.click(
            fn=lambda *args: (gr.Button("キャプション生成中...", interactive=False, variant="secondary"), ""),
            inputs=[],
            outputs=[generate_caption_button, status_message],
            queue=False  # ボタン無効化は即座に実行
        ).then(
            fn=self.generate_caption_from_upload_with_vlm,
            inputs=[upload_image, filename_input, prompt_template_dropdown, vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region],
            outputs=[generated_caption, editable_caption, status_message, image_id_state, original_caption_state],
            show_progress=True,
            queue=True
        ).then(
            fn=self.update_upload_button_states_after_generation,
            inputs=[],
            outputs=[regenerate_caption_button, update_database_button, cancel_edit_button, generate_caption_button]
        )
        
        # 画像検索ボタンの処理
        search_image_button.click(
            fn=self.search_image_by_filename,
            inputs=[filename_input],
            outputs=[display_image, generated_caption, editable_caption, status_message, image_id_state, original_caption_state],
            show_progress=True,
            queue=True
        ).then(
            fn=self.update_upload_button_states_after_search,
            inputs=[image_id_state],
            outputs=[regenerate_caption_button, update_database_button, cancel_edit_button]
        ).then(
            fn=self.show_delete_accordion_if_existing_image,
            inputs=[image_id_state],
            outputs=[delete_accordion]
        )
        
        # クリアボタンの処理
        clear_button_upload.click(
            fn=lambda: self.clear_upload_tab() + (gr.Textbox(interactive=True),),  # ファイル名欄を有効化
            inputs=[],
            outputs=[upload_image, filename_input, display_image, generated_caption, editable_caption, 
                     generate_caption_button, search_image_button, regenerate_caption_button,
                     update_database_button, cancel_edit_button, status_message, image_id_state, 
                     original_caption_state, delete_accordion, filename_input],
            queue=False  # クリア処理は即座に実行
        )
        
        # キャプション再生成ボタンの処理
        regenerate_caption_button.click(
            fn=self.disable_database_button_during_regeneration,
            inputs=[],
            outputs=[update_database_button]
        ).then(
            fn=self.regenerate_caption_with_vlm,
            inputs=[display_image, image_id_state, prompt_template_dropdown, vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region],
            outputs=[editable_caption, status_message],
            show_progress=True,
            queue=True
        ).then(
            fn=self.enable_database_button_after_regeneration,
            inputs=[],
            outputs=[update_database_button]
        )
        
        # データベース更新ボタン
        update_database_button.click(
            fn=self.update_database_with_registration_and_update,
            inputs=[generated_caption, editable_caption, image_id_state, upload_image, filename_input, original_caption_state],
            outputs=[generated_caption, editable_caption, status_message, original_caption_state, update_database_button, image_id_state],
            show_progress=True,
            queue=True
        ).then(
            fn=self.show_delete_accordion_after_registration,
            inputs=[image_id_state],
            outputs=[delete_accordion]
        )
        
        # 編集取消ボタン
        cancel_edit_button.click(
            fn=self.cancel_edit,
            inputs=[original_caption_state],
            outputs=[editable_caption, status_message],
            queue=False  # 編集取消は即座に処理
        )
        
        # 削除確認チェックボックス
        confirm_delete_checkbox.change(
            fn=self.update_delete_button_state,
            inputs=[confirm_delete_checkbox],
            outputs=[delete_button],
            queue=False  # チェックボックス変更は即座に処理
        )
        
        # 削除ボタン
        delete_button.click(
            fn=self.delete_image_from_database,
            inputs=[image_id_state, filename_input],
            outputs=[status_message, delete_accordion, image_id_state],
            show_progress=True,
            queue=True
        ).then(
            fn=self.clear_after_delete,
            inputs=[],
            outputs=[display_image, generated_caption, editable_caption, 
                    generate_caption_button, search_image_button, regenerate_caption_button, 
                    update_database_button, cancel_edit_button, confirm_delete_checkbox, delete_button]
        )
        
        # プロンプトテンプレート選択
        prompt_template_dropdown.change(
            fn=self.load_prompt_template,
            inputs=[prompt_template_dropdown],
            outputs=[current_prompt_display, prompt_edit_textbox, prompt_status_message],
            queue=False  # プロンプト選択は即座に処理
        )
        
        # プロンプト保存ボタン
        save_prompt_button.click(
            fn=self.save_prompt_template,
            inputs=[prompt_name_input, prompt_edit_textbox],
            outputs=[prompt_template_dropdown, prompt_status_message],
            show_progress=True,
            queue=True
        )
        
        # プロンプト編集取消ボタン
        cancel_prompt_edit_button.click(
            fn=self.cancel_prompt_edit,
            inputs=[prompt_template_dropdown],
            outputs=[prompt_edit_textbox, prompt_status_message],
            queue=False  # プロンプト編集取消は即座に処理
        )
        
        # プロンプト削除確認チェックボックス
        confirm_prompt_delete_checkbox.change(
            fn=self.update_prompt_delete_button_state,
            inputs=[confirm_prompt_delete_checkbox],
            outputs=[delete_prompt_button],
            queue=False  # チェックボックス変更は即座に処理
        )
        
        # プロンプト削除ボタン
        delete_prompt_button.click(
            fn=self.delete_prompt_template,
            inputs=[prompt_template_dropdown],
            outputs=[prompt_template_dropdown, current_prompt_display, prompt_edit_textbox, prompt_status_message, confirm_prompt_delete_checkbox, delete_prompt_button],
            show_progress=True,
            queue=True
        )
        
        # アプリ起動時にプロンプトテンプレートのリストを更新（削除）
        # 代わりに、UIコンポーネント作成時に初期化されます
        
    def update_upload_button_states_on_upload(self, uploaded_file, filename):
        """画像アップロード時のボタン状態更新"""
        if uploaded_file is not None:
            # 画像がアップロードされた場合
            return (
                gr.Button(interactive=True),    # キャプション生成をイネーブル
                gr.Button(interactive=False),   # 画像検索をディスエーブル
                gr.Textbox(interactive=False)   # ファイル名欄をディスエーブル
            )
        else:
            # 画像がない場合は全ボタンをディスエーブル
            return (
                gr.Button(interactive=False),   # キャプション生成をディスエーブル
                gr.Button(interactive=False),   # 画像検索をディスエーブル
                gr.Textbox(interactive=True)    # ファイル名欄をイネーブル
            )
            
    def update_upload_button_states_on_filename_input(self, uploaded_file, filename):
        """ファイル名入力時のボタン状態更新"""
        if uploaded_file is None and filename and filename.strip():
            # 画像未アップロードでファイル名が入力されている場合
            return (
                gr.Button(interactive=False),   # キャプション生成をディスエーブル
                gr.Button(interactive=True),    # 画像検索をイネーブル
                gr.Textbox(interactive=True)    # ファイル名欄をイネーブル（画像未アップロード時）
            )
        else:
            return (
                gr.Button(interactive=bool(uploaded_file)),  # キャプション生成の状態を維持
                gr.Button(interactive=False),   # 画像検索をディスエーブル
                gr.Textbox(interactive=not bool(uploaded_file))  # 画像がある場合は無効、ない場合は有効
            )
            
    def update_upload_button_states_after_generation(self):
        """キャプション生成後のボタン状態更新"""
        return (
            gr.Button(interactive=True),    # キャプション再生成をイネーブル
            gr.Button("データベースへ登録", interactive=True, variant="primary"),    # データベース登録をイネーブル
            gr.Button(interactive=True),    # 編集取消をイネーブル
            gr.Button("キャプション生成", interactive=False, variant="secondary")  # キャプション生成ボタンは無効のまま
        )
        
    def update_upload_button_states_after_search(self, image_id):
        """画像検索後のボタン状態更新"""
        if image_id is not None:
            # 検索成功時（既存画像）
            return (
                gr.Button(interactive=True),    # キャプション再生成をイネーブル
                gr.Button("データベース更新", interactive=True, variant="primary"),    # データベース更新をイネーブル
                gr.Button(interactive=True)     # 編集取消をイネーブル
            )
        else:
            # 検索失敗時
            return (
                gr.Button(interactive=False),   # キャプション再生成をディスエーブル
                gr.Button(interactive=False),   # データベース更新をディスエーブル
                gr.Button(interactive=False)    # 編集取消をディスエーブル
            )
            
    def clear_upload_tab(self):
        """アップロードタブの全コンポーネントをクリア"""
        return (
            None,                           # upload_image
            "",                             # filename_input
            None,                           # display_image
            "",                             # generated_caption
            "",                             # editable_caption
            gr.Button("キャプション生成", interactive=False, variant="primary"),   # generate_caption_button
            gr.Button(interactive=False),   # search_image_button
            gr.Button(interactive=False),   # regenerate_caption_button
            gr.Button("データベースへ登録", interactive=False, variant="primary"),   # update_database_button
            gr.Button(interactive=False),   # cancel_edit_button
            "",                             # status_message
            None,                           # image_id_state
            "",                             # original_caption_state
            gr.update(visible=False)        # delete_accordion
        )
        
    def cancel_edit(self, original_caption):
        """編集を取消して元のキャプションを復元"""
        return (
            original_caption,               # editable_caption
            "編集を取り消しました。"        # status_message
        )
        
    def update_database_with_registration_and_update(self, generated_caption, edited_caption, image_id, uploaded_file, filename_input, original_caption):
        """データベースへの新規登録または更新を行う"""
        filename = filename_input
        if not filename:
            return generated_caption, edited_caption, "❌ ファイル名を入力してください。", original_caption, gr.Button("データベースへ登録", interactive=True, variant="primary"), image_id
            
        try:
            from app.database_service import DatabaseService
            from app.config import Config
            
            config = Config()
            db_pool = config.get_db_pool()
            database_service = DatabaseService(db_pool)
            embedding_service = self.search_service.embedding_service
            
            if image_id is None:
                # 新規登録の場合
                if uploaded_file is None:
                    return generated_caption, edited_caption, "❌ 画像をアップロードしてください。", original_caption, gr.Button("データベースへ登録", interactive=True, variant="primary"), image_id
                    
                # 既に登録されているかチェック
                if database_service.is_image_registered(filename):
                    return generated_caption, edited_caption, f"⚠️ ファイル名 '{filename}' は既に登録されています。", original_caption, gr.Button("データベースへ登録", interactive=True, variant="primary"), image_id
                
                # 新規登録処理
                from PIL import Image
                from io import BytesIO
                
                if hasattr(uploaded_file, 'name'):
                    image_path = uploaded_file.name
                elif isinstance(uploaded_file, str):
                    image_path = uploaded_file
                else:
                    return generated_caption, edited_caption, "❌ ファイル形式が不正です。", original_caption, gr.Button("データベースへ登録", interactive=True, variant="primary"), image_id
                    
                uploaded_image = Image.open(image_path)
                
                # 画像データをバイト配列に変換
                # RGBA形式の場合はRGB形式に変換（JPEG形式はアルファチャンネルをサポートしないため）
                if uploaded_image.mode in ('RGBA', 'LA'):
                    # 白い背景に合成
                    background = Image.new('RGB', uploaded_image.size, (255, 255, 255))
                    if uploaded_image.mode == 'RGBA':
                        background.paste(uploaded_image, mask=uploaded_image.split()[-1])  # アルファチャンネルをマスクとして使用
                    else:  # LA (Luminance + Alpha)
                        background.paste(uploaded_image, mask=uploaded_image.split()[-1])
                    uploaded_image = background
                elif uploaded_image.mode not in ('RGB', 'L'):
                    # その他のモードもRGBに変換
                    uploaded_image = uploaded_image.convert('RGB')
                    
                buffered = BytesIO()
                uploaded_image.save(buffered, format="JPEG")
                image_data = buffered.getvalue()
                
                # 編集されたキャプションを使用して登録
                caption_to_register = edited_caption if edited_caption.strip() else generated_caption
                
                # キャプションを4000バイト以内に制限（データベース制限対応）
                caption_to_register = database_service._truncate_caption_safely(caption_to_register, 4000)
                
                # エンベディングを生成
                pil_image = Image.open(BytesIO(image_data))
                image_embedding = embedding_service.get_image_embedding(pil_image)
                caption_embedding = embedding_service.get_text_embedding(caption_to_register, "search_document")
                
                # データベースに直接挿入
                def operation():
                    with database_service.db_pool.acquire() as conn:
                        cursor = conn.cursor()
                        try:
                            cursor.execute("""
                                INSERT INTO IMAGES (file_name, caption, caption_embedding, image_data, image_embedding)
                                VALUES (:1, :2, :3, :4, :5)
                            """, (
                                filename,
                                caption_to_register,
                                caption_embedding,
                                image_data,
                                image_embedding
                            ))
                            
                            conn.commit()
                            print(f"画像 '{filename}' が正常に挿入されました。")
                            return True
                        finally:
                            cursor.close()
                
                import array
                image_embedding = array.array('f', image_embedding)
                caption_embedding = array.array('f', caption_embedding)
                
                success = database_service._execute_with_retry(operation)
                
                if success:
                    # 登録された画像のIDを取得
                    new_image = database_service.get_image_by_filename(filename)
                    new_image_id = new_image['image_id'] if new_image else None
                    return caption_to_register, caption_to_register, f"✅ 画像 '{filename}' をデータベースに登録しました。", caption_to_register, gr.Button("データベース更新", interactive=True, variant="primary"), new_image_id
                else:
                    return generated_caption, edited_caption, "❌ データベースへの登録に失敗しました。", original_caption, gr.Button("データベースへ登録", interactive=True, variant="primary"), None
                    
            else:
                # 既存画像の更新の場合
                # キャプションが変更されているかチェック
                if generated_caption.strip() == edited_caption.strip():
                    return generated_caption, edited_caption, f"画像 {filename} のキャプションが編集されていません。", original_caption, gr.Button("データベース更新", interactive=True, variant="primary"), image_id
                    
                # データベース更新
                success = database_service.update_image_caption(embedding_service, image_id, edited_caption)
                
                if success:
                    return edited_caption, edited_caption, f"✅ 画像ID {image_id} のキャプションが正常に更新されました。", edited_caption, gr.Button("データベース更新", interactive=True, variant="primary"), image_id
                else:
                    return generated_caption, edited_caption, "❌ データベースの更新に失敗しました。", original_caption, gr.Button("データベース更新", interactive=True, variant="primary"), image_id
                    
        except Exception as e:
            return generated_caption, edited_caption, f"❌ エラーが発生しました: {str(e)}", original_caption, gr.Button("データベースへ登録", interactive=True, variant="primary"), image_id

    def extract_filename_from_upload(self, uploaded_file):
        """アップロードされたファイルのファイル名を抽出"""
        if uploaded_file is None:
            return ""
        
        try:
            import os
            # Gradioのファイルコンポーネントからファイル名を抽出
            if hasattr(uploaded_file, 'name') and uploaded_file.name:
                # ファイルパスからファイル名のみを抽出
                return os.path.basename(uploaded_file.name)
            elif isinstance(uploaded_file, str):
                # パス文字列の場合
                return os.path.basename(uploaded_file)
            else:
                # デフォルトのファイル名を生成
                import time
                timestamp = int(time.time())
                return f"uploaded_image_{timestamp}.jpg"
        except Exception as e:
            print(f"ファイル名抽出エラー: {e}")
            # エラーの場合はタイムスタンプベースのファイル名を生成
            import time
            timestamp = int(time.time())
            return f"uploaded_image_{timestamp}.jpg"

    def generate_caption_from_upload_with_vlm(self, uploaded_file, filename, selected_prompt_template, vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region):
        """VLM設定を使用してアップロードされたファイルからキャプションを生成"""
        if uploaded_file is None:
            return "", "", "❌ 画像をアップロードしてください。", None, ""
            
        if not filename:
            return "", "", "❌ ファイル名を入力してください。", None, ""
        
        try:
            # ファイルパスを取得
            if hasattr(uploaded_file, 'name'):
                image_path = uploaded_file.name
            elif isinstance(uploaded_file, str):
                image_path = uploaded_file
            else:
                return "", "", "❌ ファイル形式が不正です。", None, ""
            
            # 選択されたプロンプトテンプレートを読み込み
            custom_prompt = self.get_current_prompt(selected_prompt_template)
            
            # NLPサービスを使用してVLMでキャプション生成
            from app.nlp_service import NLPService
            nlp_service = NLPService()
            
            caption = nlp_service.generate_caption_with_vlm(
                image_path=image_path,
                vlm_model=vlm_model,
                prompt_text=custom_prompt,
                temperature=vlm_temperature,
                max_tokens=vlm_max_tokens,
                oci_region=vlm_oci_region
            )
            
            return caption, caption, f"✅ VLM（{vlm_model}）でキャプションを生成しました。", None, caption
                
        except Exception as e:
            print(f"VLMキャプション生成エラー: {e}")
            return "", "", f"❌ エラーが発生しました: {str(e)}", None, ""
    
    def regenerate_caption_with_vlm(self, display_image, image_id, selected_prompt_template, vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region):
        """VLM設定を使用してキャプションを再生成"""
        if display_image is None:
            return "", "❌ 画像が表示されていません。"
            
        try:
            # 一時ファイルに画像を保存
            import tempfile
            import os
            
            # 画像データを一時ファイルに保存
            processed_image = display_image
            if processed_image.mode in ('RGBA', 'LA'):
                # 白い背景に合成
                from PIL import Image
                background = Image.new('RGB', processed_image.size, (255, 255, 255))
                if processed_image.mode == 'RGBA':
                    background.paste(processed_image, mask=processed_image.split()[-1])
                else:  # LA (Luminance + Alpha)
                    background.paste(processed_image, mask=processed_image.split()[-1])
                processed_image = background
            elif processed_image.mode not in ('RGB', 'L'):
                processed_image = processed_image.convert('RGB')
            
            # 一時ファイルを作成
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                processed_image.save(temp_file.name, format="JPEG")
                temp_image_path = temp_file.name
            
            try:
                # 選択されたプロンプトテンプレートを読み込み
                custom_prompt = self.get_current_prompt(selected_prompt_template)
                
                # NLPサービスを使用してVLMでキャプション生成
                from app.nlp_service import NLPService
                nlp_service = NLPService()
                
                new_caption = nlp_service.generate_caption_with_vlm(
                    image_path=temp_image_path,
                    vlm_model=vlm_model,
                    prompt_text=custom_prompt,
                    temperature=vlm_temperature,
                    max_tokens=vlm_max_tokens,
                    oci_region=vlm_oci_region
                )
                
                return new_caption, f"✅ VLM（{vlm_model}）でキャプションを再生成しました。"
                
            finally:
                # 一時ファイルを削除
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
            
        except Exception as e:
            print(f"VLMキャプション再生成エラー: {e}")
            return "", f"❌ エラーが発生しました: {str(e)}"

    def search_image_by_filename(self, filename):
        """ファイル名で画像を検索"""
        if not filename:
            return None, "", "", "❌ ファイル名を入力してください。", None, ""
            
        try:
            database_service = self.search_service.database_service
            result = database_service.get_image_by_filename(filename)
            
            if result:
                return (result['image'], result['caption'], result['caption'], 
                        f"✅ ファイル名 '{filename}' の画像を取得しました。", result['image_id'], result['caption'])
            else:
                return None, "", "", f"❌ ファイル名 '{filename}' の画像が見つかりませんでした。", None, ""
                
        except Exception as e:
            print(f"画像検索エラー: {e}")
            return None, "", "", f"❌ エラーが発生しました: {str(e)}", None, ""

    def regenerate_caption(self, display_image, image_id, selected_prompt_template="デフォルト"):
        """キャプションを再生成（編集側にのみ新しいキャプションを表示）"""
        if display_image is None:
            return "", "❌ 画像が表示されていません。"
            
        try:
            # 画像データをバイト配列に変換
            # RGBA形式の場合はRGB形式に変換（JPEG形式はアルファチャンネルをサポートしないため）
            processed_image = display_image
            if processed_image.mode in ('RGBA', 'LA'):
                # 白い背景に合成
                background = Image.new('RGB', processed_image.size, (255, 255, 255))
                if processed_image.mode == 'RGBA':
                    background.paste(processed_image, mask=processed_image.split()[-1])  # アルファチャンネルをマスクとして使用
                else:  # LA (Luminance + Alpha)
                    background.paste(processed_image, mask=processed_image.split()[-1])
                processed_image = background
            elif processed_image.mode not in ('RGB', 'L'):
                # その他のモードもRGBに変換
                processed_image = processed_image.convert('RGB')
                
            buffered = BytesIO()
            processed_image.save(buffered, format="JPEG")
            image_data = buffered.getvalue()
            
            # データベースサービスを通じてキャプションを生成
            database_service = self.search_service.database_service
            
            # 設定を取得
            from app.config import Config
            config = Config()
            oci_client = config.get_oci_generative_ai_client()
            
            # 選択されたプロンプトテンプレートを読み込み
            custom_prompt = self.get_current_prompt(selected_prompt_template)
            
            new_caption = database_service.get_image_caption(
                oci_client, image_data, config.mllm_model_id, config.compartment_id, custom_prompt
            )
            
            # 右側（編集可能なキャプション）にのみ新しいキャプションを表示
            # 左側は変更しないので出力対象から除外
            return new_caption, "✅ キャプションを再生成しました。"
            
        except Exception as e:
            print(f"キャプション再生成エラー: {e}")
            return "", f"❌ エラーが発生しました: {str(e)}"

    def disable_database_button_during_regeneration(self):
        """キャプション再生成中にデータベース更新ボタンを無効化"""
        return gr.update(interactive=False)

    def enable_database_button_after_regeneration(self):
        """キャプション再生成後にデータベース更新ボタンを有効化"""
        return gr.update(interactive=True)
        
    def display_uploaded_image(self, uploaded_file):
        """アップロードされた画像を表示"""
        if uploaded_file is None:
            return None
            
        try:
            from PIL import Image
            
            # ファイルパスからPIL画像を読み込み
            if hasattr(uploaded_file, 'name'):
                image_path = uploaded_file.name
            elif isinstance(uploaded_file, str):
                image_path = uploaded_file
            else:
                return None
                
            uploaded_image = Image.open(image_path)
            return uploaded_image
            
        except Exception as e:
            print(f"画像表示エラー: {e}")
            return None

    def show_delete_accordion_if_existing_image(self, image_id):
        """既存画像がある場合に削除アコーディオンを表示"""
        return gr.update(visible=True) if image_id else gr.update(visible=False)

    def show_delete_accordion_after_registration(self, image_id):
        """新規登録後に削除アコーディオンを表示"""
        return gr.update(visible=True) if image_id else gr.update(visible=False)

    def update_delete_button_state(self, confirm_delete):
        """削除確認チェックボックスの状態に応じて削除ボタンを有効化"""
        return gr.Button(interactive=confirm_delete)

    def delete_image_from_database(self, image_id, filename):
        """データベースから画像を削除"""
        if image_id:
            try:
                database_service = self.search_service.database_service
                success = database_service.delete_image(image_id)
                if success:
                    return "✅ 画像を削除しました。", gr.update(visible=False), None
                else:
                    return "❌ データベースからの削除に失敗しました。", gr.update(visible=False), None
            except Exception as e:
                return f"❌ エラーが発生しました: {str(e)}", gr.update(visible=False), None
        else:
            return "❌ 画像IDが指定されていません。", gr.update(visible=False), None

    def clear_after_delete(self):
        """削除後にアップロードタブの全コンポーネントをクリア"""
        return (
            None,                           # display_image
            "",                             # generated_caption
            "",                             # editable_caption
            gr.Button("キャプション生成", interactive=False, variant="primary"),   # generate_caption_button
            gr.Button(interactive=False),   # search_image_button
            gr.Button(interactive=False),   # regenerate_caption_button
            gr.Button("データベースへ登録", interactive=False, variant="primary"),   # update_database_button
            gr.Button(interactive=False),   # cancel_edit_button
            gr.Checkbox(value=False, interactive=True),  # confirm_delete_checkbox
            gr.Button(interactive=False)    # delete_button
        )
    
    # プロンプト関連のメソッド
    def load_prompt_template(self, template_name):
        """プロンプトテンプレートを読み込み"""
        from app.prompt_service import PromptService
        
        prompt_service = PromptService()
        prompt_text = prompt_service.load_template(template_name)
        
        if prompt_text:
            return prompt_text, prompt_text, f"プロンプトテンプレート '{template_name}' を読み込みました。"
        else:
            return "", "", f"❌ プロンプトテンプレート '{template_name}' の読み込みに失敗しました。"
    
    def save_prompt_template(self, template_name, prompt_text):
        """プロンプトテンプレートを保存"""
        from app.prompt_service import PromptService
        
        if not template_name or not template_name.strip():
            return gr.update(), "❌ プロンプト名を入力してください。"
        
        if not prompt_text or not prompt_text.strip():
            return gr.update(), "❌ プロンプトの内容を入力してください。"
        
        prompt_service = PromptService()
        success = prompt_service.save_template(template_name.strip(), prompt_text.strip())
        
        if success:
            # テンプレートリストを更新（ただし、値は変更せず選択肢のみ更新）
            template_names = prompt_service.get_template_names()
            return gr.update(choices=template_names), f"✅ プロンプトテンプレート '{template_name}' を保存しました。"
        else:
            return gr.update(), f"❌ プロンプトテンプレート '{template_name}' の保存に失敗しました。"
    
    def cancel_prompt_edit(self, current_template_name):
        """プロンプト編集を取消"""
        from app.prompt_service import PromptService
        
        prompt_service = PromptService()
        original_prompt = prompt_service.load_template(current_template_name)
        
        if original_prompt:
            return original_prompt, "プロンプト編集を取り消しました。"
        else:
            return "", "元のプロンプトの読み込みに失敗しました。"
    
    def update_prompt_template_list(self, prompt_template_dropdown, current_prompt_display, prompt_edit_textbox):
        """プロンプトテンプレートのリストを更新"""
        from app.prompt_service import PromptService
        
        prompt_service = PromptService()
        template_names = prompt_service.get_template_names()
        default_template_name = prompt_service.get_default_template_name()
        
        # デフォルトプロンプトを読み込み
        default_prompt = prompt_service.load_template(default_template_name)
        
        # コンポーネントを直接更新
        prompt_template_dropdown.choices = template_names
        prompt_template_dropdown.value = default_template_name
        current_prompt_display.value = default_prompt or ""
        prompt_edit_textbox.value = default_prompt or ""
    
    def get_current_prompt(self, template_name):
        """現在選択されているプロンプトテンプレートを取得"""
        from app.prompt_service import PromptService
        
        prompt_service = PromptService()
        return prompt_service.load_template(template_name)
    
    def update_prompt_delete_button_state(self, confirm_delete):
        """プロンプト削除確認チェックボックスの状態に応じて削除ボタンを有効化"""
        return gr.Button(interactive=confirm_delete)
    
    def delete_prompt_template(self, current_template_name):
        """プロンプトテンプレートを削除"""
        from app.prompt_service import PromptService
        
        prompt_service = PromptService()
        
        # デフォルトテンプレートの削除を防止
        if current_template_name == prompt_service.get_default_template_name():
            return (
                gr.update(),  # prompt_template_dropdown（変更なし）
                gr.update(),  # current_prompt_display（変更なし）
                gr.update(),  # prompt_edit_textbox（変更なし）
                "❌ デフォルトプロンプトは削除できません。",  # prompt_status_message
                gr.Checkbox(value=False, interactive=True),  # confirm_prompt_delete_checkbox（リセット）
                gr.Button(interactive=False)  # delete_prompt_button（無効化）
            )
        
        # プロンプトテンプレートを削除
        success = prompt_service.delete_template(current_template_name)
        
        if success:
            # テンプレートリストを更新し、デフォルトテンプレートを選択
            template_names = prompt_service.get_template_names()
            default_template_name = prompt_service.get_default_template_name()
            default_prompt = prompt_service.load_template(default_template_name)
            
            return (
                gr.update(choices=template_names, value=default_template_name),  # prompt_template_dropdown
                default_prompt or "",  # current_prompt_display
                default_prompt or "",  # prompt_edit_textbox
                f"✅ プロンプトテンプレート '{current_template_name}' を削除しました。",  # prompt_status_message
                gr.Checkbox(value=False, interactive=True),  # confirm_prompt_delete_checkbox（リセット）
                gr.Button(interactive=False)  # delete_prompt_button（無効化）
            )
        else:
            return (
                gr.update(),  # prompt_template_dropdown（変更なし）
                gr.update(),  # current_prompt_display（変更なし）
                gr.update(),  # prompt_edit_textbox（変更なし）
                f"❌ プロンプトテンプレート '{current_template_name}' の削除に失敗しました。",  # prompt_status_message
                gr.Checkbox(value=False, interactive=True),  # confirm_prompt_delete_checkbox（リセット）
                gr.Button(interactive=False)  # delete_prompt_button（無効化）
            ) 