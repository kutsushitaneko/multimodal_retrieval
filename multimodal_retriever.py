import gradio as gr
import time
import threading
import uuid
from app.config import Config
from app.embedding_service import EmbeddingService
from app.database_service import DatabaseService  
from app.search_service import SearchService
from app.ui.components import UIComponents
from app.ui.events import UIEvents
from app.search_query_generator import SearchQueryGenerator
from app.cleanup_service import CleanupService
import os

def check_db_connection(config, db_pool, interval=60):
    """定期的にデータベース接続の健全性をチェックするバックグラウンドスレッド"""
    while True:
        try:
            if not config.check_pool_health(db_pool):
                print("データベース接続プールが不健全です。再接続を試みます...")
                # 古いプールを閉じる
                try:
                    db_pool.close()
                except Exception as e:
                    print(f"プールのクローズ中にエラーが発生しました: {e}")
                
                # 新しいプールを作成
                db_pool = config.get_db_pool()
                print("データベース接続プールを再作成しました。")
        except Exception as e:
            print(f"接続チェック中にエラーが発生しました: {e}")
        
        # 指定された間隔で次のチェックを実行
        time.sleep(interval)

def main():
    # ベースの一時ディレクトリを作成（Gradio一時ディレクトリの親）
    base_temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(base_temp_dir, exist_ok=True)
    
    # Gradio用の一時ディレクトリをカレントディレクトリ内に作成
    gradio_temp_dir = os.path.join(base_temp_dir, "gradio")
    os.makedirs(gradio_temp_dir, exist_ok=True)
    
    # 書き込み権限を確認
    try:
        test_file = os.path.join(gradio_temp_dir, "test_write_permission")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Gradio一時ディレクトリの書き込み権限確認完了: {gradio_temp_dir}")
    except Exception as e:
        print(f"❌ Gradio一時ディレクトリ権限エラー: {e}")
        raise
    
    # Gradioの一時ディレクトリを環境変数で指定
    os.environ['GRADIO_TEMP_DIR'] = gradio_temp_dir
    print(f"📁 Gradio一時ディレクトリを設定しました: {gradio_temp_dir}")
    
    # 一時ディレクトリクリーンアップサービスを開始
    cleanup_service = CleanupService(base_temp_dir, max_age_hours=24)
    cleanup_service.start_cleanup_thread(interval_minutes=120)  # 2時間ごとにクリーンアップ
    
    # 設定と基本サービスを初期化
    config = Config()
    db_pool = config.get_db_pool()  # コネクションプールを取得
    cohere_client = config.get_cohere_client()
    oci_client = config.get_oci_generative_ai_client()
    search_query_generator = SearchQueryGenerator()
    
    # データベース接続監視スレッドを開始
    db_monitor_thread = threading.Thread(
        target=check_db_connection,
        args=(config, db_pool, 60),  # 60秒ごとにチェック
        daemon=True  # メインスレッド終了時に自動的に終了
    )
    db_monitor_thread.start()
    
    # 各サービスを初期化
    embedding_service = EmbeddingService(
        embed_model_provider=config.embed_model_provider,
        embed_model_id=config.embed_model_id,
        compartment_id=config.compartment_id,
        cohere_client=cohere_client if config.embed_model_provider == "CohereAI" else None,
        oci_client=oci_client if config.embed_model_provider == "OCI" else None
    )
    print(f"🔗 マルチモーダル埋め込みモデル: {config.embed_model_provider} - {config.embed_model_id}")
    database_service = DatabaseService(db_pool)  # プールを渡す
    search_service = SearchService(embedding_service, database_service, search_query_generator)
    
    # UIコンポーネントとイベントを初期化
    ui_components = UIComponents()
    ui_events = UIEvents(search_service)
    
    # Gradioインターフェースの作成
    with gr.Blocks(title="🐕マルチモーダル・レトリバー🐕", delete_cache=(86400, 86400)) as demo:
        gr.Markdown(f"# 🐕マルチモーダル・レトリバー🐕 by {config.embed_model_id}")
        gr.Markdown("画像を自然言語やアップロードした画像で検索できます。例: 「ハリーにホグワーツの入学案内を持ってきたのは誰？」など")
        
        state = gr.State({
            "current_page": 1,
            "total_pages": 1,
            "page_size": 0,
            "total_image_count": 0,
            "all_images": [],
            "combined_results": [],
            "vector_results": [],
            "keyword_results": []
        })
        
        # タブ機能を追加
        with gr.Tabs():
            # タブ1: 検索機能
            with gr.Tab("検索と回答生成"):
                # 検索セクションのUIコンポーネントを作成
                search_target, search_method, query_input, uploaded_image, search_button, clear_button, show_all_button, query_examples = ui_components.create_search_section()
                
                # 検索結果セクションのUIコンポーネントを作成
                vector_gallery, keyword_gallery = ui_components.create_results_section()
                
                # ページングセクションのUIコンポーネントを作成
                pagination_row, prev_button, page_info, next_button = ui_components.create_pagination_section()
                
                # 画像詳細セクションのUIコンポーネントを作成
                filename_text, similarity_text, caption_text = ui_components.create_detail_section()
                
                # クエリ詳細セクションのUIコンポーネントを作成
                executed_query_text, execute_query_button, executed_sql_text, morphological_analysis_text = ui_components.create_query_detail_section()
                
                # 自然言語による回答セクションのUIコンポーネントを作成
                (reference_image_text, answer_generate_button, answer_text, reference_type_radio, answer_question_input) = ui_components.create_answer_generation_section()
                
                # 回答生成プロンプト設定セクションのUIコンポーネントを作成
                (answer_prompt_template_dropdown, current_answer_prompt_display, answer_prompt_edit_textbox,
                 answer_prompt_name_input, save_answer_prompt_button, cancel_answer_prompt_edit_button, 
                 answer_prompt_status_message, confirm_answer_prompt_delete_checkbox, delete_answer_prompt_button) = ui_components.create_answer_prompt_settings_section()
                
                # 検索タブ専用VLM設定セクションのUIコンポーネントを作成
                (search_vlm_service_provider, search_vlm_model, search_vlm_temperature, 
                 search_vlm_max_tokens, search_vlm_oci_region, search_vlm_status_message) = ui_components.create_search_vlm_settings()
                
                # 高度な設定セクションのUIコンポーネントを作成
                vector_threshold, keyword_threshold, top_k_slider = ui_components.create_advanced_settings_section()
                
                # 各種イベントを登録
                ui_events.register_search_target_events(
                    search_target, search_method, query_input, uploaded_image, query_examples, executed_sql_text
                )
                
                ui_events.register_search_method_events(
                    search_method, query_input, uploaded_image, similarity_text, 
                    executed_query_text, execute_query_button, search_target, query_examples, morphological_analysis_text
                )
                
                ui_events.register_search_button_events(
                    search_button, query_input, uploaded_image, search_target, 
                    search_method, top_k_slider, vector_threshold, keyword_threshold, 
                    vector_gallery, keyword_gallery, filename_text, similarity_text, 
                    caption_text, state, executed_query_text, executed_sql_text, execute_query_button, pagination_row, morphological_analysis_text, reference_image_text, answer_question_input, answer_generate_button, reference_type_radio
                )
                
                ui_events.register_execute_query_button_events(
                    execute_query_button, executed_query_text, top_k_slider, keyword_threshold,
                    vector_gallery, filename_text, similarity_text, caption_text, 
                    state, executed_query_text, executed_sql_text, pagination_row, answer_question_input
                )
                
                ui_events.register_clear_button_events(
                    clear_button, query_input, uploaded_image, vector_gallery, 
                    keyword_gallery, filename_text, similarity_text, caption_text, 
                    state, executed_query_text, executed_sql_text, pagination_row, morphological_analysis_text,
                    answer_generate_button, answer_text, reference_image_text, reference_type_radio, answer_question_input
                )
                
                ui_events.register_show_all_button_events(
                    show_all_button, top_k_slider, vector_gallery, 
                    keyword_gallery, filename_text, similarity_text, caption_text, 
                    state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button, morphological_analysis_text, reference_image_text, answer_question_input, search_target, search_method, answer_generate_button, reference_type_radio
                )
                
                ui_events.register_pagination_events(
                    prev_button, next_button, top_k_slider, vector_gallery, page_info, state, keyword_gallery, prev_button, next_button
                )
                
                ui_events.register_gallery_selection_events(
                    vector_gallery, keyword_gallery, state, 
                    filename_text, similarity_text, caption_text,
                    search_target, search_method, answer_generate_button, reference_image_text, reference_type_radio
                )
                
                ui_events.register_answer_generation_events(
                    answer_generate_button, answer_text, search_target, search_method, 
                    vector_gallery, keyword_gallery, state, reference_type_radio, answer_question_input, 
                    answer_prompt_template_dropdown, current_answer_prompt_display, answer_prompt_edit_textbox,
                    answer_prompt_name_input, save_answer_prompt_button, cancel_answer_prompt_edit_button, 
                    answer_prompt_status_message, confirm_answer_prompt_delete_checkbox, delete_answer_prompt_button
                )
                
                # 検索タブVLM設定のイベントを登録
                ui_events.register_search_vlm_settings_events(
                    search_vlm_service_provider, search_vlm_model, search_vlm_temperature, 
                    search_vlm_max_tokens, search_vlm_oci_region, search_vlm_status_message
                )
                
                # アプリケーションの初期読み込み時のイベントを登録
                demo.load(
                    fn=lambda: None,
                    outputs=None
                ).then(
                    fn=ui_events.show_all_images,
                    inputs=[top_k_slider, state],
                    outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, 
                             caption_text, state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button, morphological_analysis_text, reference_image_text]
                )
            
            # タブ2: アップロード・編集機能
            with gr.Tab("イメージ管理"):
                gr.Markdown("## 画像のアップロード・キャプション生成・編集・画像の削除")
                gr.Markdown("画像をアップロードしてキャプションを生成したり、既存の画像のキャプションを編集、画像の削除ができます。")
                
                # アップロード・編集セクションのUIコンポーネントを作成
                (upload_image, filename_input, generate_caption_button, search_image_button, copy_filename_button, clear_button_upload,
                 display_image, generated_caption, editable_caption, regenerate_caption_button, 
                 update_database_button, cancel_edit_button, status_message, image_id_state, original_caption_state,
                 delete_accordion, confirm_delete_checkbox, delete_button,
                 prompt_template_dropdown, current_prompt_display, prompt_edit_textbox, 
                 prompt_name_input, save_prompt_button, cancel_prompt_edit_button, prompt_status_message,
                 confirm_prompt_delete_checkbox, delete_prompt_button,
                 vlm_service_provider_upload, vlm_model_upload, vlm_temperature_upload, vlm_max_tokens_upload, vlm_oci_region_upload, vlm_status_message) = ui_components.create_upload_edit_section()
                
                # VLM設定のイベントを登録
                ui_events.register_vlm_settings_events(
                    vlm_service_provider_upload, vlm_model_upload, vlm_temperature_upload, vlm_max_tokens_upload, vlm_oci_region_upload, vlm_status_message
                )
                
                # アップロード・編集機能のイベントを登録
                ui_events.register_upload_edit_events(
                    upload_image, filename_input, generate_caption_button, search_image_button, copy_filename_button, clear_button_upload,
                    display_image, generated_caption, editable_caption, regenerate_caption_button, 
                    update_database_button, cancel_edit_button, status_message, image_id_state, original_caption_state,
                    delete_accordion, confirm_delete_checkbox, delete_button,
                    prompt_template_dropdown, current_prompt_display, prompt_edit_textbox, 
                    prompt_name_input, save_prompt_button, cancel_prompt_edit_button, prompt_status_message,
                    confirm_prompt_delete_checkbox, delete_prompt_button,
                    vlm_service_provider_upload, vlm_model_upload, vlm_temperature_upload, vlm_max_tokens_upload, vlm_oci_region_upload, vlm_status_message
                )
                
                # 検索結果からコピーボタンのイベントを登録（タブ間連携）
                copy_filename_button.click(
                    fn=ui_events.copy_filename_from_search_result,
                    inputs=[state],
                    outputs=[filename_input, display_image, generated_caption, editable_caption, status_message, image_id_state, original_caption_state],
                    queue=False  # ファイル名コピーは即座に処理
                ).then(
                    fn=ui_events.update_upload_button_states_after_search,
                    inputs=[image_id_state],
                    outputs=[regenerate_caption_button, update_database_button, cancel_edit_button]
                ).then(
                    fn=ui_events.show_delete_accordion_if_existing_image,
                    inputs=[image_id_state],
                    outputs=[delete_accordion]
                )
                
                # copy_filename_buttonの状態更新（ギャラリー選択時）
                vector_gallery.select(
                    fn=ui_events.update_copy_button_state,
                    inputs=[state],
                    outputs=[copy_filename_button]
                )
                
                keyword_gallery.select(
                    fn=ui_events.update_copy_button_state,
                    inputs=[state],
                    outputs=[copy_filename_button]
                )
    
    # アプリケーションの起動
    launch_config = config.get_launch_config()
    demo.launch(**launch_config)

if __name__ == "__main__":
    main()