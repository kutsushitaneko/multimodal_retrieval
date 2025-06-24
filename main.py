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

def create_session_temp_dir():
    """セッション固有の一時ディレクトリを作成
    
    マルチセッション環境でファイル名衝突を防ぐため、
    各セッションに固有のディレクトリを作成します。
    
    Returns:
        str: セッション固有の一時ディレクトリパス
    """
    session_id = str(uuid.uuid4())[:8]  # 短縮版UUIDを使用
    temp_dir = os.path.join(os.getcwd(), "temp", f"session_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

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
    # ベースの一時ディレクトリを作成（セッション固有ディレクトリの親）
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
    database_service = DatabaseService(db_pool)  # プールを渡す
    search_service = SearchService(embedding_service, database_service, search_query_generator)
    
    # UIコンポーネントとイベントを初期化
    ui_components = UIComponents()
    ui_events = UIEvents(search_service)
    
    # Gradioインターフェースの作成
    with gr.Blocks(title="マルチモーダル画像検索") as demo:
        gr.Markdown("# マルチモーダル画像検索")
        gr.Markdown("画像を自然言語やアップロードした画像で検索できます。例: 「富士山と寺院」、「縞模様の猫」、「三匹の白い子猫」、「ホグワーツ魔法学校」、「上海のビル」、「2312.10997」など")
        
        # セッション固有の一時ディレクトリを設定
        def setup_session_temp_dir():
            session_temp_dir = create_session_temp_dir()
            # このセッション用の環境変数を設定（ただし、実際にはGradioが内部的に管理）
            return session_temp_dir
        
        # 隠しステートでセッション固有の設定を管理
        session_temp_dir = gr.State(setup_session_temp_dir)
        
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
            with gr.Tab("検索"):
                # 検索セクションのUIコンポーネントを作成
                search_target, search_method, query_input, uploaded_image, search_button, clear_button, show_all_button, query_examples = ui_components.create_search_section()
                
                # 検索結果セクションのUIコンポーネントを作成
                vector_gallery, keyword_gallery = ui_components.create_results_section()
                
                # ページングセクションのUIコンポーネントを作成
                pagination_row, prev_button, page_info, next_button = ui_components.create_pagination_section()
                
                # 画像詳細セクションのUIコンポーネントを作成
                filename_text, similarity_text, caption_text, score_label = ui_components.create_detail_section()
                
                # クエリ詳細セクションのUIコンポーネントを作成
                executed_query_text, execute_query_button, executed_sql_text, morphological_analysis_text = ui_components.create_query_detail_section()
                
                # 高度な設定セクションのUIコンポーネントを作成
                vector_threshold, keyword_threshold, top_k_slider = ui_components.create_advanced_settings_section()
                
                # 各種イベントを登録
                ui_events.register_search_target_events(
                    search_target, search_method, query_input, uploaded_image, query_examples, executed_sql_text
                )
                
                ui_events.register_search_method_events(
                    search_method, query_input, uploaded_image, score_label, 
                    executed_query_text, execute_query_button, search_target, query_examples, morphological_analysis_text
                )
                
                ui_events.register_search_button_events(
                    search_button, query_input, uploaded_image, search_target, 
                    search_method, top_k_slider, vector_threshold, keyword_threshold, 
                    vector_gallery, keyword_gallery, filename_text, similarity_text, 
                    caption_text, state, executed_query_text, executed_sql_text, execute_query_button, pagination_row, morphological_analysis_text
                )
                
                ui_events.register_execute_query_button_events(
                    execute_query_button, executed_query_text, top_k_slider, keyword_threshold,
                    vector_gallery, filename_text, similarity_text, caption_text, 
                    state, executed_query_text, executed_sql_text, pagination_row
                )
                
                ui_events.register_clear_button_events(
                    clear_button, query_input, uploaded_image, vector_gallery, 
                    keyword_gallery, filename_text, similarity_text, caption_text, 
                    state, executed_query_text, executed_sql_text, pagination_row, morphological_analysis_text
                )
                
                ui_events.register_show_all_button_events(
                    show_all_button, top_k_slider, vector_gallery, 
                    keyword_gallery, filename_text, similarity_text, caption_text, 
                    state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button, morphological_analysis_text
                )
                
                ui_events.register_pagination_events(
                    prev_button, next_button, top_k_slider, vector_gallery, page_info, state, keyword_gallery, prev_button, next_button
                )
                
                ui_events.register_gallery_selection_events(
                    vector_gallery, keyword_gallery, state, 
                    filename_text, similarity_text, caption_text
                )
                
                # アプリケーションの初期読み込み時のイベントを登録
                demo.load(
                    fn=lambda: None,
                    outputs=None
                ).then(
                    fn=ui_events.show_all_images,
                    inputs=[top_k_slider, state],
                    outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, 
                             caption_text, state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button, morphological_analysis_text]
                )
            
            # タブ2: アップロード・編集機能
            with gr.Tab("イメージ管理"):
                gr.Markdown("## 画像のアップロード・キャプション生成・編集・画像の削除")
                gr.Markdown("画像をアップロードしてキャプションを生成したり、既存の画像のキャプションを編集、画像の削除ができます。")
                
                # アップロード・編集セクションのUIコンポーネントを作成
                (upload_image, filename_input, generate_caption_button, search_image_button, clear_button_upload,
                 display_image, generated_caption, editable_caption, regenerate_caption_button, 
                 update_database_button, cancel_edit_button, status_message, image_id_state, original_caption_state,
                 delete_accordion, confirm_delete_checkbox, delete_button) = ui_components.create_upload_edit_section()
                
                # アップロード・編集機能のイベントを登録
                ui_events.register_upload_edit_events(
                    upload_image, filename_input, generate_caption_button, search_image_button, clear_button_upload,
                    display_image, generated_caption, editable_caption, regenerate_caption_button, 
                    update_database_button, cancel_edit_button, status_message, image_id_state, original_caption_state,
                    delete_accordion, confirm_delete_checkbox, delete_button
                )
    
    # アプリケーションの起動
    launch_config = config.get_launch_config()
    demo.launch(**launch_config)

if __name__ == "__main__":
    main()