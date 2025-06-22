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
                                        "search_queries_only"
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
                        gr.Markdown("**ファイル名：**")
                        filename_text = gr.Textbox(show_label=False, interactive=False, container=False)
                        score_label = gr.Markdown("**コサイン類似度：**")
                        similarity_text = gr.Textbox(show_label=False, interactive=False, container=False)
                    with gr.Row():
                        caption_text = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            lines=10,
                            show_copy_button=True,
                            placeholder="説明"
                        )
                
        return filename_text, similarity_text, caption_text, score_label
        
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
                    scale=4,
                    lines=2
                )
                execute_query_button = gr.Button("このクエリを実行", visible=False, scale=1)

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
                
        return executed_query_text, execute_query_button, executed_sql_text
        
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