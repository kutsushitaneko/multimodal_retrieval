import gradio as gr

class UIComponents:
    """UIコンポーネントを管理するクラス"""
    
    def create_search_section(self):
        """検索セクションのUIコンポーネントを作成"""
        with gr.Row():
            with gr.Accordion("検索", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row(): 
                            with gr.Column():
                                search_target = gr.Radio(
                                    choices=["画像", "キャプション"],
                                    value="画像",
                                    label="検索対象",
                                    container=True
                                )
                            with gr.Column():   
                                # 検索方法のラジオボタン（キャプション検索の場合は非表示）
                                search_method = gr.Radio(
                                    choices=["テキスト", "画像"],
                                    value="テキスト",
                                    label="検索方法",
                                    container=True,
                                    visible=True
                                )
                        with gr.Row():
                            # 初期状態でクエリ入力フィールドを表示
                            query_input = gr.Textbox(
                                label="検索クエリ",
                                placeholder="検索したい画像の内容を入力してください",
                                visible=True
                            )
                            
                            # 検索クエリのサンプル例を追加（Columnで囲む）
                            with gr.Column(visible=True) as query_examples:
                                query_examples_inner = gr.Examples(
                                    examples=[
                                        "富士山と寺院", 
                                        "縞模様の猫", 
                                        "三匹の白い子猫", 
                                        "ホグワーツ魔法学校", 
                                        "上海のビル", 
                                        "2312.10997", 
                                        "https://qiita.com/yuji-arakawa/items/28f30a5434ba429f3f16"
                                    ],
                                    inputs=query_input,
                                    label="検索クエリの例"
                                )
                    with gr.Column(scale=1):
                        # 画像アップロードフィールドは初期状態では非表示
                        uploaded_image = gr.Image(
                            label="画像をアップロード",
                            type="pil",
                            visible=False,
                            height=300,
                            width=300
                        )
                
                with gr.Row():
                    search_button = gr.Button("検索")
                    clear_button = gr.Button("クリア")
                    show_all_button = gr.Button("全件表示")
                    
        return search_target, search_method, query_input, uploaded_image, search_button, clear_button, show_all_button, query_examples

    def create_upload_edit_section(self):
        """アップロード・編集セクションのUIコンポーネントを作成"""
        with gr.Row():
            with gr.Column(scale=1):
                # 画像アップロード（ファイル名情報も取得できるように変更）
                upload_image = gr.File(
                    label="画像をアップロード",
                    file_types=["image"],
                    file_count="single"
                )
                
                # ファイル名入力（コピーボタン追加）
                filename_input = gr.Textbox(
                    label="ファイル名",
                    placeholder="ファイル名を入力してください（例: image001.jpg）",
                    interactive=True,
                    show_copy_button=True
                )
                
                # ボタン群
                with gr.Row():
                    generate_caption_button = gr.Button("キャプション生成", variant="primary", interactive=False)
                    search_image_button = gr.Button("画像を検索", interactive=False)
                    clear_button = gr.Button("クリア")
                
                # 表示画像
                display_image = gr.Image(
                    label="表示画像",
                    type="pil",
                    height=400,
                    width=400,
                    interactive=False
                )
                
            with gr.Column(scale=2):
                # キャプションの編集アコーディオン（デフォルトでオープン）
                with gr.Accordion("キャプションの編集と登録", open=True):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # 左側：生成されたキャプション（読み取り専用）
                            gr.Markdown("### 生成されたキャプション")
                            generated_caption = gr.Textbox(
                                label="",
                                lines=12,
                                interactive=False,
                                show_copy_button=True,
                                placeholder="キャプションがここに表示されます"
                            )
                            
                        with gr.Column(scale=1):
                            # 右側：編集可能なキャプション
                            gr.Markdown("### キャプションの編集")
                            editable_caption = gr.Textbox(
                                label="",
                                lines=12,
                                interactive=True,
                                show_copy_button=True,
                                placeholder="キャプションを編集してください"
                            )
                    
                    # ボタン群を一行に配置
                    with gr.Row():
                        regenerate_caption_button = gr.Button("キャプション再生成", interactive=False)
                        update_database_button = gr.Button("データベースへ登録", variant="primary", interactive=False)
                        cancel_edit_button = gr.Button("編集を取消", interactive=False)
                    
                    # ステータス表示
                    status_message = gr.Markdown("")
                
                # プロンプトの編集セクション（デフォルトでクローズ）
                with gr.Accordion("プロンプトの設定と編集", open=False) as settings_accordion:
                        # プロンプトテンプレートの初期化
                        def initialize_prompt_template_choices():
                            try:
                                from app.prompt_service import PromptService
                                prompt_service = PromptService()
                                template_names = prompt_service.get_template_names()
                                default_template_name = prompt_service.get_default_template_name()
                                return template_names, default_template_name
                            except Exception as e:
                                print(f"プロンプトテンプレート初期化エラー: {e}")
                                return ["デフォルト"], "デフォルト"
                        
                        initial_choices, initial_value = initialize_prompt_template_choices()
                        
                        # プロンプトテンプレート選択
                        prompt_template_dropdown = gr.Dropdown(
                            label="プロンプトテンプレート",
                            choices=initial_choices,
                            value=initial_value,
                            interactive=True
                        )
                        
                        # デフォルトプロンプトを読み込み
                        def get_initial_prompt():
                            try:
                                from app.prompt_service import PromptService
                                prompt_service = PromptService()
                                default_prompt = prompt_service.load_template(initial_value)
                                return default_prompt or ""
                            except Exception as e:
                                print(f"デフォルトプロンプト読み込みエラー: {e}")
                                return ""
                        
                        initial_prompt = get_initial_prompt()
                        
                        # プロンプト表示・編集を左右に配置
                        with gr.Row():
                            with gr.Column(scale=1):
                                # 左側：現在のプロンプト（表示のみ）
                                current_prompt_display = gr.Textbox(
                                    label="現在のプロンプト",
                                    lines=8,
                                    interactive=False,
                                    show_copy_button=True,
                                    value=initial_prompt,
                                    placeholder="選択されたプロンプトがここに表示されます"
                                )
                            
                            with gr.Column(scale=1):
                                # 右側：プロンプトの編集
                                prompt_edit_textbox = gr.Textbox(
                                    label="プロンプトの設定・編集",
                                    lines=8,
                                    interactive=True,
                                    value=initial_prompt,
                                    placeholder="プロンプトを編集してください"
                                )
                        
                        # プロンプト名と操作ボタン
                        with gr.Row():
                            prompt_name_input = gr.Textbox(
                                label="プロンプトの名前",
                                placeholder="新しいプロンプト名を入力",
                                scale=2
                            )
                        
                        with gr.Row():
                            save_prompt_button = gr.Button("プロンプトを保存", variant="primary")
                            cancel_prompt_edit_button = gr.Button("プロンプト編集を取消")
                        
                        # プロンプトの削除アコーディオン（デフォルトでクローズ）
                        with gr.Accordion("プロンプトの削除", open=False) as prompt_delete_accordion:
                            gr.Markdown("⚠️ **危険な操作**: 選択されたプロンプトテンプレートを完全に削除します。この操作は元に戻せません。")
                            with gr.Row():
                                confirm_prompt_delete_checkbox = gr.Checkbox(
                                    label="削除することを確認しました",
                                    value=False,
                                    interactive=True
                                )
                            delete_prompt_button = gr.Button(
                                "🗑️ プロンプトを削除",
                                variant="stop",
                                interactive=False
                            )
                        
                        # プロンプト操作のステータス
                        prompt_status_message = gr.Markdown("")
                
                # イメージの削除アコーディオン（デフォルトでクローズ）
                with gr.Accordion("イメージの削除", open=False, visible=False) as delete_accordion:
                    gr.Markdown("⚠️ **危険な操作**: この画像をデータベースから完全に削除します。この操作は元に戻せません。")
                    with gr.Row():
                        confirm_delete_checkbox = gr.Checkbox(
                            label="削除することを確認しました",
                            value=False,
                            interactive=True
                        )
                    delete_button = gr.Button(
                        "🗑️ データベースから削除",
                        variant="stop",
                        interactive=False
                    )
                
        # 隠し状態でimage_idと元のキャプションを管理
        image_id_state = gr.State(value=None)
        original_caption_state = gr.State(value="")
        
        return (upload_image, filename_input, generate_caption_button, search_image_button, clear_button,
                display_image, generated_caption, editable_caption, regenerate_caption_button, 
                update_database_button, cancel_edit_button, status_message, image_id_state, original_caption_state,
                delete_accordion, confirm_delete_checkbox, delete_button,
                prompt_template_dropdown, current_prompt_display, prompt_edit_textbox, 
                prompt_name_input, save_prompt_button, cancel_prompt_edit_button, prompt_status_message,
                confirm_prompt_delete_checkbox, delete_prompt_button)
        
    def create_results_section(self):
        """検索結果セクションのUIコンポーネントを作成"""
        with gr.Accordion("検索結果", open=True):            
            with gr.Row():
                vector_gallery = gr.Gallery(
                    label="最近アップロードされた画像",
                    show_label=True,
                    columns=[8],
                    rows=[1],
                    object_fit="contain",
                    container=True,
                    preview=False,
                    allow_preview=True,
                    selected_index=None
                )
            with gr.Row():
                keyword_gallery = gr.Gallery(
                    label="全文検索",
                    show_label=True,
                    columns=[8],
                    rows=[1],
                    object_fit="contain",
                    container=True,
                    preview=False,
                    allow_preview=True,
                    selected_index=None,
                    visible=False
                )
                
        return vector_gallery, keyword_gallery
        
    def create_pagination_section(self):
        """ページング用のUIコンポーネントを作成"""
        with gr.Row(visible=False) as pagination_row:
            prev_button = gr.Button("前へ")
            page_info = gr.Markdown("1/1 ページ（合計 0 枚）")
            next_button = gr.Button("次へ")
            
        return pagination_row, prev_button, page_info, next_button
        
    def create_detail_section(self):
        """画像詳細セクションのUIコンポーネントを作成"""
        with gr.Accordion("画像詳細", open=False):
            with gr.Row():        
                with gr.Column():
                    with gr.Row():
                        filename_text = gr.Textbox(show_label=True, label="ファイル名", interactive=False, container=True, show_copy_button=True)
                        similarity_text = gr.Textbox(show_label=True, label="コサイン類似度", interactive=False, container=True, show_copy_button=True)
                    with gr.Row():
                        caption_text = gr.Textbox(
                            show_label=True,
                            label="キャプション",
                            interactive=False,
                            lines=10,
                            container=True,
                            show_copy_button=True,
                            placeholder="キャプションがここに表示されます"
                        )
                
        return filename_text, similarity_text, caption_text
        
    def create_query_detail_section(self):
        """クエリ詳細セクションのUIコンポーネントを作成"""
        with gr.Accordion("クエリ詳細", open=False):
            with gr.Row():
                # 実行されたクエリと実行ボタンのコンポーネント
                executed_query_text = gr.Textbox(
                    label="実行されたクエリ",
                    show_label=True,
                    interactive=False,
                    container=True,
                    show_copy_button=True,
                    scale=4,
                    lines=2
                )
                execute_query_button = gr.Button("このクエリを実行", visible=False, scale=1)

            # 形態素解析結果表示のコンポーネント（初期状態では非表示）
            morphological_analysis_text = gr.Markdown(
                label="全文検索：形態素解析結果",
                show_label=True,
                container=True,
                show_copy_button=True,
                visible=False,
                elem_id="morphological_analysis"
            )

            with gr.Row():
                # 初期状態ではデフォルトの8行表示を使用
                # 検索対象選択のイベントハンドラーで後から動的に更新
                executed_sql_text = gr.Textbox(
                    label="実行されたSQL",
                    show_label=True,
                    interactive=False,
                    lines=8,
                    show_copy_button=True,
                    container=True
                )
                
        return executed_query_text, execute_query_button, executed_sql_text, morphological_analysis_text
        
    def create_advanced_settings_section(self):
        """高度な設定セクションのUIコンポーネントを作成"""
        with gr.Accordion("高度な設定", open=False):
            with gr.Row():
                vector_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    label="ベクトル検索の閾値",
                    info="値が小さいほど多くの結果が表示されます（0.0～1.0）"
                )
                keyword_threshold = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="全文検索の閾値",
                    info="値が小さいほど多くの結果が表示されます（0～100）"
                )
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1, 
                    maximum=48, 
                    value=16, 
                    step=1, 
                    label="表示する結果の最大数"
                )
                
        return vector_threshold, keyword_threshold, top_k_slider 