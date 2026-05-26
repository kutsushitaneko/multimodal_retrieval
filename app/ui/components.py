import os

import gradio as gr

# 定数定義
QUERY_INPUT_PLACEHOLDER_TEXT = "検索したい画像の内容を入力してください"
REFERENCE_DOCUMENT_LABEL_TEXT = "参照するドキュメント（画像）"
REFERENCE_IMAGE_PLACEHOLDER_TEXT = "条件に合致する画像を選択すると、ファイル名がここに表示されます"
REFERENCE_TYPE_LABEL_TEXT = "参照する情報の種類"
REFERENCE_TYPE_ALL = "すべて"
REFERENCE_TYPE_CAPTION_ONLY = "キャプションのみ"
REFERENCE_TYPE_IMAGE_ONLY = "画像のみ"
ANSWER_GENERATION_MODE_LABEL_TEXT = "回答生成モード"
ANSWER_MODE_SINGLE_IMAGE = "先頭画像あるいは選択した１つの画像"
ANSWER_MODE_LISTWISE = "VLMによるフィルタリングと並べ替え"

class UIComponents:
    """UIコンポーネントを管理するクラス"""
    
    def create_search_section(self):
        """検索セクションのUIコンポーネントを作成"""
        with gr.Row():
            with gr.Accordion("検索", open=True):
                with gr.Accordion("検索設定", open=False):
                    with gr.Row(): 
                        with gr.Column():
                            search_target = gr.Radio(
                                choices=["画像ベクトル", "キャプション（テキストベクトルと全文）"],
                                value="画像ベクトル",
                                label="検索対象",
                                container=True
                            )
                        with gr.Column():   
                            # クエリーの種類のラジオボタン（キャプション検索の場合は非表示）
                            search_method = gr.Radio(
                                choices=["テキスト", "画像"],
                                value="テキスト",
                                label="クエリーの種類",
                                container=True,
                                visible=True
                            )
                        with gr.Column():
                            search_count_input = gr.Number(
                                minimum=1,
                                maximum=24,
                                value=8,
                                step=1,
                                label="検索件数",
                                precision=0,
                                interactive=True
                            )
                with gr.Accordion("検索結果の選別・並べ替え設定", open=False):
                    answer_generation_mode_radio = gr.Radio(
                        choices=[ANSWER_MODE_SINGLE_IMAGE, ANSWER_MODE_LISTWISE],
                        value=ANSWER_MODE_LISTWISE,
                        label=ANSWER_GENERATION_MODE_LABEL_TEXT,
                        container=True,
                        interactive=True
                    )
                with gr.Row():
                    with gr.Column(scale=2):
                        
                        with gr.Row():
                            # 初期状態でクエリ入力フィールドを表示
                            query_input = gr.Textbox(
                                label="検索クエリ",
                                placeholder=QUERY_INPUT_PLACEHOLDER_TEXT,
                                visible=True
                            )
                            
                            # 検索クエリのサンプル例を追加（Columnで囲む）
                            with gr.Column(visible=True) as query_examples:
                                with gr.Accordion("検索クエリの例", open=False):
                                    gr.Examples(
                                        examples=[
                                            "富士山と寺院", 
                                            "縞模様の猫", 
                                            "三匹の白い子猫", 
                                            "ハリーにホグワーツの入学案内を持ってきたのは誰？", 
                                            "スターライトブレイカーとは何？",
                                            "練馬区がアニメ発祥の地と謳っている看板があるのはどこですか？",
                                            "lost in the middle とは？",
                                            "MCPは、アプリ開発者にとってどんなメリットがありますか？",
                                            "ORA-00923 とは何ですか？",
                                            "Oracle Autonomous Database は、Generative AI Agents のナレッジベースとして利用できますか？",
                                            "ベクトル検索のチャンクサイズと件数と精度の関係は？",
                                            "企業がコーディング・エージェントではなく独自のエージェントを開発する意義はどこにありますか？",
                                            "川口市栄町の7月の金属ゴミの回収日はいつ？",
                                            "2312.10997のタイトルは？", 
                                            "https://qiita.com/yuji-arakawa/items/28f30a5434ba429f3f16"
                                        ],
                                        inputs=query_input,
                                        label="クリックして選択"
                                    )
                    with gr.Column(scale=1, visible=False) as uploaded_image_column:
                        # 画像アップロードフィールドは初期状態では非表示
                        uploaded_image = gr.Image(
                            label="画像をアップロード",
                            type="pil",
                            visible=False,
                            height=300,
                            width=300
                        )
                
                with gr.Row():
                    search_button = gr.Button("検索", variant="primary")
                    search_and_answer_button = gr.Button("検索と回答生成", variant="primary")
                    clear_button = gr.Button("クリア")
                    show_all_button = gr.Button("全件表示")
                    
        return search_target, search_method, search_count_input, query_input, uploaded_image, uploaded_image_column, search_button, search_and_answer_button, clear_button, show_all_button, query_examples, answer_generation_mode_radio

    def create_search_vlm_settings(self):
        """検索タブ専用VLM設定セクションのUIコンポーネントを作成"""
        with gr.Accordion("VLM設定（検索・回答生成用）", open=False) as search_vlm_settings_accordion:
            # モデル設定の初期化
            try:
                from app.vlm_service import build_vlm_ui_initialization
                print("🔍 検索タブVLM設定セクション初期化")
                (
                    search_vlm_choices,
                    search_default_vlm,
                    search_provider_choices,
                    search_vlm_models,
                    _search_vlm_service,
                ) = build_vlm_ui_initialization()
            except Exception as e:
                print(f"検索タブVLMモデル初期化エラー: {e}")
                import traceback
                traceback.print_exc()
                search_vlm_choices, search_default_vlm, search_provider_choices, search_vlm_models = (
                    ["エラー"],
                    "エラー",
                    ["すべて"],
                    {},
                )
            
            # サービスプロバイダー選択
            search_vlm_service_provider = gr.Dropdown(
                label="サービスプロバイダ",
                choices=search_provider_choices,
                value="すべて",
                interactive=True
            )
            
            # VLMモデル選択
            search_vlm_model = gr.Dropdown(
                label="VLMモデル",
                choices=search_vlm_choices,
                value=search_default_vlm,
                interactive=True
            )
            
            # 温度設定（モデル設定から初期値を取得）
            try:
                from app.vlm_service import VLMService
                _temp_vlm = VLMService()
                search_initial_temperature = (
                    _temp_vlm.get_model_default_temperature(search_default_vlm)
                    if search_default_vlm != "エラー"
                    else 0.0
                )
            except Exception:
                search_initial_temperature = 0.0
            search_vlm_temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=search_initial_temperature,
                interactive=True
            )
            
            # Max tokens設定
            search_initial_max_tokens = search_vlm_models.get(search_default_vlm, {}).get("max_tokens", 4096) if search_default_vlm != "エラー" else 4096
            search_initial_default_tokens = search_vlm_models.get(search_default_vlm, {}).get("default_tokens", 4096) if search_default_vlm != "エラー" else 4096
            
            search_vlm_max_tokens = gr.Slider(
                label="Max tokens",
                minimum=1,
                maximum=search_initial_max_tokens,
                step=1,
                value=search_initial_default_tokens,
                interactive=True
            )
            
            # OCIリージョン設定（OCIモデルの場合のみ表示）
            OCI_REGIONS = {
                "Brazil East (Sao Paulo)": "sa-saopaulo-1",
                "Germany Central (Frankfurt)": "eu-frankfurt-1", 
                "Japan Central (Osaka)": "ap-osaka-1",
                "UAE East (Dubai)": "me-dubai-1",
                "UK South (London)": "uk-london-1",
                "US Midwest (Chicago)": "us-chicago-1"
            }
            
            search_initial_is_oci = search_default_vlm != "エラー" and search_vlm_models.get(search_default_vlm, {}).get("api_type", "").startswith("oci")
            
            # 初期リージョンを設定（モデルのdefault_regionを優先）
            search_initial_region_name = "Japan Central (Osaka)"  # デフォルトのフォールバック値
            if search_initial_is_oci:
                search_default_region_id = search_vlm_models.get(search_default_vlm, {}).get("default_region")
                if search_default_region_id:
                    # region_idからregion_nameを逆引き
                    for name, region_id in OCI_REGIONS.items():
                        if region_id == search_default_region_id:
                            search_initial_region_name = name
                            break
            
            search_vlm_oci_region = gr.Dropdown(
                label="OCIリージョン",
                choices=list(OCI_REGIONS.keys()),
                value=search_initial_region_name,
                interactive=True,
                visible=search_initial_is_oci
            )
            
            # VLM設定のステータス
            search_vlm_status_message = gr.Markdown("")
        
        return (search_vlm_service_provider, search_vlm_model, search_vlm_temperature, 
                search_vlm_max_tokens, search_vlm_oci_region, search_vlm_status_message)

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
                    label="画像ファイル名",
                    placeholder="画像ファイル名を入力してください（例: image001.jpg）",
                    interactive=True,
                    show_copy_button=True
                )
                
                # ボタン群
                with gr.Row():
                    generate_caption_button = gr.Button("キャプション生成", variant="primary", interactive=False)
                    search_image_button = gr.Button("画像を検索", interactive=False)
                    copy_filename_button = gr.Button("検索結果からコピー", variant="secondary", interactive=True)
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
                
                # キャプション生成プロンプトの編集セクション（デフォルトでクローズ）
                with gr.Accordion("キャプション生成プロンプトの設定と編集", open=False) as settings_accordion:
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
                
                # VLM設定セクション（デフォルトでクローズ）
                with gr.Accordion("VLM設定", open=False) as vlm_settings_accordion:
                    # モデル設定の初期化
                    try:
                        from app.vlm_service import build_vlm_ui_initialization
                        print("🔍 UIコンポーネント初期化 - VLM設定セクション")
                        vlm_choices, default_vlm, provider_choices, vlm_models, _upload_vlm_service = (
                            build_vlm_ui_initialization()
                        )
                    except Exception as e:
                        print(f"VLMモデル初期化エラー: {e}")
                        import traceback
                        traceback.print_exc()
                        vlm_choices, default_vlm, provider_choices, vlm_models = (
                            ["エラー"],
                            "エラー",
                            ["すべて"],
                            {},
                        )
                    
                    # サービスプロバイダー選択
                    vlm_service_provider = gr.Dropdown(
                        label="サービスプロバイダ",
                        choices=provider_choices,
                        value="すべて",
                        interactive=True
                    )
                    
                    # VLMモデル選択
                    vlm_model = gr.Dropdown(
                        label="VLMモデル",
                        choices=vlm_choices,
                        value=default_vlm,
                        interactive=True
                    )
                    
                    # 温度設定（モデル設定から初期値を取得）
                    try:
                        from app.vlm_service import VLMService
                        _temp_vlm = VLMService()
                        initial_temperature = _temp_vlm.get_model_default_temperature(default_vlm) if default_vlm != "エラー" else 0.0
                    except Exception:
                        initial_temperature = 0.0
                    vlm_temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=initial_temperature,
                        interactive=True
                    )
                    
                    # Max tokens設定
                    initial_max_tokens = vlm_models.get(default_vlm, {}).get("max_tokens", 4096) if default_vlm != "エラー" else 4096
                    initial_default_tokens = vlm_models.get(default_vlm, {}).get("default_tokens", 4096) if default_vlm != "エラー" else 4096
                    
                    vlm_max_tokens = gr.Slider(
                        label="Max tokens",
                        minimum=1,
                        maximum=initial_max_tokens,
                        step=1,
                        value=initial_default_tokens,
                        interactive=True
                    )
                    
                    # OCIリージョン設定（OCIモデルの場合のみ表示）
                    OCI_REGIONS = {
                        "Brazil East (Sao Paulo)": "sa-saopaulo-1",
                        "Germany Central (Frankfurt)": "eu-frankfurt-1", 
                        "Japan Central (Osaka)": "ap-osaka-1",
                        "UAE East (Dubai)": "me-dubai-1",
                        "UK South (London)": "uk-london-1",
                        "US Midwest (Chicago)": "us-chicago-1"
                    }
                    
                    initial_is_oci = default_vlm != "エラー" and vlm_models.get(default_vlm, {}).get("api_type", "").startswith("oci")
                    
                    # 初期リージョンを設定（モデルのdefault_regionを優先）
                    initial_region_name = "Japan Central (Osaka)"  # デフォルトのフォールバック値
                    if initial_is_oci:
                        default_region_id = vlm_models.get(default_vlm, {}).get("default_region")
                        if default_region_id:
                            # region_idからregion_nameを逆引き
                            for name, region_id in OCI_REGIONS.items():
                                if region_id == default_region_id:
                                    initial_region_name = name
                                    break
                    
                    vlm_oci_region = gr.Dropdown(
                        label="OCIリージョン",
                        choices=list(OCI_REGIONS.keys()),
                        value=initial_region_name,
                        interactive=True,
                        visible=initial_is_oci
                    )
                    
                    # VLM設定のステータス
                    vlm_status_message = gr.Markdown("")
                
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
        
        return (upload_image, filename_input, generate_caption_button, search_image_button, copy_filename_button, clear_button,
                display_image, generated_caption, editable_caption, regenerate_caption_button, 
                update_database_button, cancel_edit_button, status_message, image_id_state, original_caption_state,
                delete_accordion, confirm_delete_checkbox, delete_button,
                prompt_template_dropdown, current_prompt_display, prompt_edit_textbox, 
                prompt_name_input, save_prompt_button, cancel_prompt_edit_button, prompt_status_message,
                confirm_prompt_delete_checkbox, delete_prompt_button,
                vlm_service_provider, vlm_model, vlm_temperature, vlm_max_tokens, vlm_oci_region, vlm_status_message)
        
    def create_results_section(self):
        """検索結果セクションのUIコンポーネントを作成"""
        with gr.Accordion("検索結果", open=True):            
            with gr.Row():
                vector_gallery = gr.Gallery(
                    label="最近アップロードされた画像",
                    show_label=True,
                    columns=[4],
                    rows=[2],
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
                    columns=[4],
                    rows=[2],
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
                    value=0.25,
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
                
        return vector_threshold, keyword_threshold
        
    def create_answer_generation_section(self):
        """自然言語による回答セクションのUIコンポーネントを作成"""
        with gr.Accordion("自然言語による回答", open=True):
            with gr.Row():
                # 参照するドキュメント（画像）の表示領域（コピーボタン付き）
                reference_image_text = gr.Textbox(
                    label=REFERENCE_DOCUMENT_LABEL_TEXT,
                    show_label=True,
                    interactive=False,
                    container=True,
                    show_copy_button=True,
                    placeholder=REFERENCE_IMAGE_PLACEHOLDER_TEXT,
                    scale=1
                )
                # 参照する情報の種類ラジオボタン
                reference_type_radio = gr.Radio(
                    choices=[REFERENCE_TYPE_ALL, REFERENCE_TYPE_CAPTION_ONLY, REFERENCE_TYPE_IMAGE_ONLY],
                    value=REFERENCE_TYPE_ALL,
                    label=REFERENCE_TYPE_LABEL_TEXT,
                    container=True,
                    interactive=True,
                    scale=1
                )
            with gr.Row():
                # 質問文入力エリア
                answer_question_input = gr.Textbox(
                    label="質問文",
                    placeholder="回答生成で使用する質問文を入力してください",
                    interactive=True,
                    lines=3,
                    show_copy_button=True,
                    container=True,
                    scale=3
                )
                # 回答生成ボタン
                answer_generate_button = gr.Button(
                    "回答生成", 
                    variant="primary", 
                    interactive=False,
                    scale=1
                )
            with gr.Row():
                answer_text = gr.Textbox(
                    label="回答",
                    show_label=True,
                    interactive=False,
                    lines=10,
                    container=True,
                    show_copy_button=True,
                    placeholder="回答がここに表示されます"
                )
            with gr.Row():
                referenced_images_gallery = gr.Gallery(
                    label="参照した画像",
                    show_label=True,
                    columns=4,
                    height=240,
                    object_fit="contain",
                    visible=False
                )
            with gr.Row():
                listwise_reason_text = gr.Textbox(
                    label="reason",
                    show_label=True,
                    interactive=False,
                    lines=3,
                    container=True,
                    show_copy_button=True,
                    visible=False,
                    placeholder="VLMによるフィルタリングと並べ替えの選別理由がここに表示されます"
                )
                
        return (reference_image_text, answer_generate_button, answer_text, reference_type_radio, answer_question_input, referenced_images_gallery, listwise_reason_text)

    def create_agentic_rag_section(self):
        """Agentic RAGタブのUIコンポーネントを作成"""
        gr.Markdown("## Agentic マルチモーダル RAG")
        gr.Markdown(
            "自然文質問から質問分解、複数検索、十分性判定、再検索、選別・並べ替え、回答生成までを自動実行します。"
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="質問",
                    placeholder="調べたい内容を自然文で入力してください",
                    lines=4,
                    interactive=True,
                    show_copy_button=True,
                )
            with gr.Column(scale=1):
                uploaded_image = gr.Image(
                    label="任意の入力画像",
                    type="pil",
                    height=240,
                    width=240,
                    visible=True,
                )

        with gr.Row():
            reference_type_radio = gr.Radio(
                choices=[REFERENCE_TYPE_ALL, REFERENCE_TYPE_CAPTION_ONLY, REFERENCE_TYPE_IMAGE_ONLY],
                value=REFERENCE_TYPE_ALL,
                label=REFERENCE_TYPE_LABEL_TEXT,
                interactive=True,
            )
            top_k_input = gr.Number(
                minimum=1,
                maximum=24,
                value=8,
                step=1,
                label="検索件数",
                precision=0,
                interactive=True,
            )
            max_iterations_input = gr.Number(
                minimum=0,
                maximum=3,
                value=2,
                step=1,
                label="再検索回数上限",
                precision=0,
                interactive=True,
            )

        with gr.Row():
            run_button = gr.Button("Agentic RAG 実行", variant="primary")
            clear_button = gr.Button("クリア")

        with gr.Row():
            answer_text = gr.Textbox(
                label="回答",
                lines=10,
                interactive=False,
                show_copy_button=True,
                placeholder="回答がここに表示されます",
            )

        with gr.Row():
            referenced_images_gallery = gr.Gallery(
                label="参照した画像",
                columns=4,
                rows=2,
                height=480,
                object_fit="contain",
                visible=False,
            )

        with gr.Row():
            trace_text = gr.Textbox(
                label="実行トレース",
                lines=10,
                interactive=False,
                show_copy_button=True,
                placeholder="質問分解、検索、再検索、選別の流れがここに表示されます",
            )

        with gr.Row():
            selection_reason_text = gr.Textbox(
                label="選別・並べ替え理由",
                lines=4,
                interactive=False,
                show_copy_button=True,
                placeholder="回答に使用した evidence の選別理由がここに表示されます",
            )

        with gr.Accordion("回答生成プロンプト", open=False):
            answer_prompt_template_dropdown = gr.Dropdown(
                label="回答生成プロンプトテンプレート",
                choices=self._get_answer_prompt_template_names(),
                value=self._get_default_answer_prompt_template_name(),
                interactive=True,
            )

        return (
            question_input,
            uploaded_image,
            reference_type_radio,
            top_k_input,
            max_iterations_input,
            run_button,
            clear_button,
            answer_text,
            referenced_images_gallery,
            trace_text,
            selection_reason_text,
            answer_prompt_template_dropdown,
        )

    def create_agentic_vlm_settings(self):
        """Agentic RAGタブ専用VLM設定セクションのUIコンポーネントを作成"""
        with gr.Accordion("VLM設定（Agentic RAG用）", open=False):
            try:
                from app.vlm_service import build_vlm_ui_initialization, VLMService

                (
                    vlm_choices,
                    default_vlm,
                    provider_choices,
                    vlm_models,
                    _vlm_service,
                ) = build_vlm_ui_initialization()
                all_models = _vlm_service.model_settings
                agentic_model_choices = self._get_agentic_model_choices(all_models, vlm_choices)
            except Exception as e:
                print(f"Agentic RAG VLMモデル初期化エラー: {e}")
                vlm_choices, default_vlm, provider_choices, vlm_models, all_models, agentic_model_choices = (
                    ["エラー"],
                    "エラー",
                    ["すべて"],
                    {},
                    {},
                    ["エラー"],
                )

            vlm_service_provider = gr.Dropdown(
                label="サービスプロバイダ",
                choices=provider_choices,
                value="すべて",
                interactive=True,
            )
            vlm_model = gr.Dropdown(
                label="VLMモデル",
                choices=vlm_choices,
                value=default_vlm,
                interactive=True,
            )

            gr.Markdown("### Agentic 判定系モデル")
            decompose_model = gr.Dropdown(
                label="質問分解モデル",
                choices=agentic_model_choices,
                value=self._resolve_agentic_model_default(
                    all_models,
                    os.getenv("AGENTIC_DECOMPOSE_MODEL_ID"),
                    default_vlm,
                ),
                interactive=True,
            )
            sufficiency_model = gr.Dropdown(
                label="十分性判定モデル",
                choices=agentic_model_choices,
                value=self._resolve_agentic_model_default(
                    all_models,
                    os.getenv("AGENTIC_SUFFICIENCY_MODEL_ID"),
                    default_vlm,
                ),
                interactive=True,
            )
            followup_query_model = gr.Dropdown(
                label="追加検索クエリー生成モデル",
                choices=agentic_model_choices,
                value=self._resolve_agentic_model_default(
                    all_models,
                    os.getenv("AGENTIC_FOLLOWUP_QUERY_MODEL_ID"),
                    default_vlm,
                ),
                interactive=True,
            )

            try:
                _temp_vlm = VLMService()
                initial_temperature = (
                    _temp_vlm.get_model_default_temperature(default_vlm)
                    if default_vlm != "エラー"
                    else 0.0
                )
            except Exception:
                initial_temperature = 0.0

            vlm_temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=initial_temperature,
                interactive=True,
            )
            initial_max_tokens = vlm_models.get(default_vlm, {}).get("max_tokens", 4096) if default_vlm != "エラー" else 4096
            initial_default_tokens = vlm_models.get(default_vlm, {}).get("default_tokens", 4096) if default_vlm != "エラー" else 4096
            vlm_max_tokens = gr.Slider(
                label="Max tokens",
                minimum=1,
                maximum=initial_max_tokens,
                step=1,
                value=initial_default_tokens,
                interactive=True,
            )

            oci_regions = {
                "Brazil East (Sao Paulo)": "sa-saopaulo-1",
                "Germany Central (Frankfurt)": "eu-frankfurt-1",
                "Japan Central (Osaka)": "ap-osaka-1",
                "UAE East (Dubai)": "me-dubai-1",
                "UK South (London)": "uk-london-1",
                "US Midwest (Chicago)": "us-chicago-1",
            }
            initial_is_oci = default_vlm != "エラー" and vlm_models.get(default_vlm, {}).get("api_type", "").startswith("oci")
            initial_region_name = "Japan Central (Osaka)"
            default_region_id = vlm_models.get(default_vlm, {}).get("default_region")
            if default_region_id:
                for name, region_id in oci_regions.items():
                    if region_id == default_region_id:
                        initial_region_name = name
                        break

            vlm_oci_region = gr.Dropdown(
                label="OCIリージョン",
                choices=list(oci_regions.keys()),
                value=initial_region_name,
                interactive=True,
                visible=initial_is_oci,
            )

        return (
            vlm_service_provider,
            vlm_model,
            vlm_temperature,
            vlm_max_tokens,
            vlm_oci_region,
            decompose_model,
            sufficiency_model,
            followup_query_model,
        )

    @staticmethod
    def _resolve_agentic_model_default(vlm_models, env_model_id, fallback_model):
        """Agentic判定系モデルの .env 指定を Gradio Dropdown の表示名へ解決する。"""
        if not env_model_id:
            return fallback_model
        env_model_id = env_model_id.strip()
        if env_model_id in vlm_models:
            return env_model_id
        for display_name, model_info in vlm_models.items():
            if model_info.get("model_name") == env_model_id:
                return display_name
        for display_name in vlm_models:
            if env_model_id in display_name:
                return display_name
        return fallback_model

    @staticmethod
    def _get_agentic_model_choices(all_models, fallback_choices):
        """Agentic判定系は画像を渡さないため、定義済み全モデルを選択肢にする。"""
        return list(all_models.keys()) if all_models else fallback_choices

    def _get_answer_prompt_template_names(self):
        import glob

        default_name = self._get_default_answer_prompt_template_name()
        prompt_dir = "answer_prompt"
        template_names = []
        for file_path in glob.glob(os.path.join(prompt_dir, "*.txt")):
            template_names.append(os.path.splitext(os.path.basename(file_path))[0])
        if default_name not in template_names:
            template_names.insert(0, default_name)
        return template_names

    def _get_default_answer_prompt_template_name(self):
        return "デフォルト（回答生成）"
    
    def create_answer_prompt_settings_section(self):
        """回答生成プロンプトの設定と編集セクションのUIコンポーネントを作成"""
        with gr.Accordion("回答生成プロンプトの設定と編集", open=False):
            # 回答生成プロンプトテンプレートの初期化
            def initialize_answer_prompt_template_choices():
                try:
                    import os
                    import glob
                    answer_prompt_dir = "answer_prompt"
                    if not os.path.exists(answer_prompt_dir):
                        os.makedirs(answer_prompt_dir, exist_ok=True)
                    
                    # answer_promptフォルダーからテンプレート一覧を取得
                    template_files = glob.glob(os.path.join(answer_prompt_dir, "*.txt"))
                    template_names = []
                    for file_path in template_files:
                        basename = os.path.basename(file_path)
                        template_name = os.path.splitext(basename)[0]
                        template_names.append(template_name)
                    
                    # デフォルトテンプレートを確認
                    default_template_name = "デフォルト（回答生成）"
                    if not template_names:
                        template_names = [default_template_name]
                    elif default_template_name not in template_names:
                        template_names.insert(0, default_template_name)
                    
                    return template_names, default_template_name
                except Exception as e:
                    print(f"回答生成プロンプトテンプレート初期化エラー: {e}")
                    return ["デフォルト（回答生成）"], "デフォルト（回答生成）"
            
            initial_answer_choices, initial_answer_value = initialize_answer_prompt_template_choices()
            
            # 回答生成プロンプトテンプレート選択
            answer_prompt_template_dropdown = gr.Dropdown(
                label="回答生成プロンプトテンプレート",
                choices=initial_answer_choices,
                value=initial_answer_value,
                interactive=True
            )
            
            # デフォルト回答生成プロンプトを読み込み
            def get_initial_answer_prompt():
                try:
                    import os
                    answer_prompt_path = os.path.join("answer_prompt", f"{initial_answer_value}.txt")
                    if os.path.exists(answer_prompt_path):
                        with open(answer_prompt_path, 'r', encoding='utf-8') as f:
                            return f.read()
                    return ""
                except Exception as e:
                    print(f"デフォルト回答生成プロンプト読み込みエラー: {e}")
                    return ""
            
            initial_answer_prompt = get_initial_answer_prompt()
            
            # 回答生成プロンプト表示・編集を左右に配置
            with gr.Row():
                with gr.Column(scale=1):
                    # 左側：現在の回答生成プロンプト（表示のみ）
                    current_answer_prompt_display = gr.Textbox(
                        label="現在の回答生成プロンプト",
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                        value=initial_answer_prompt,
                        placeholder="選択された回答生成プロンプトがここに表示されます"
                    )
                
                with gr.Column(scale=1):
                    # 右側：回答生成プロンプトの編集
                    answer_prompt_edit_textbox = gr.Textbox(
                        label="回答生成プロンプトの設定・編集",
                        lines=8,
                        interactive=True,
                        value=initial_answer_prompt,
                        placeholder="回答生成プロンプトを編集してください"
                    )
            
            # 回答生成プロンプト名と操作ボタン
            with gr.Row():
                answer_prompt_name_input = gr.Textbox(
                    label="回答生成プロンプトの名前",
                    placeholder="新しい回答生成プロンプト名を入力",
                    scale=2
                )
            
            with gr.Row():
                save_answer_prompt_button = gr.Button("回答生成プロンプトを保存", variant="primary")
                cancel_answer_prompt_edit_button = gr.Button("回答生成プロンプト編集を取消")
            
            # 回答生成プロンプトの削除アコーディオン（デフォルトでクローズ）
            with gr.Accordion("回答生成プロンプトの削除", open=False):
                gr.Markdown("⚠️ **危険な操作**: 選択された回答生成プロンプトテンプレートを完全に削除します。この操作は元に戻せません。")
                with gr.Row():
                    confirm_answer_prompt_delete_checkbox = gr.Checkbox(
                        label="削除することを確認しました",
                        value=False,
                        interactive=True
                    )
                delete_answer_prompt_button = gr.Button(
                    "🗑️ 回答生成プロンプトを削除",
                    variant="stop",
                    interactive=False
                )
            
            # 回答生成プロンプト操作のステータス
            answer_prompt_status_message = gr.Markdown("")
            
        return (answer_prompt_template_dropdown, current_answer_prompt_display, answer_prompt_edit_textbox,
                answer_prompt_name_input, save_answer_prompt_button, cancel_answer_prompt_edit_button, 
                answer_prompt_status_message, confirm_answer_prompt_delete_checkbox, delete_answer_prompt_button) 