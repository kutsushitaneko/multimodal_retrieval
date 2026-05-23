import json
import os


def resolve_default_vlm_display_name(vlm_models, mllm_model_id=None):
    """環境変数 MLLM_MODEL_ID または Vision モデル一覧の先頭からデフォルト表示名を決定する"""
    if not vlm_models:
        return "Vision対応モデルがありません"

    if mllm_model_id:
        model_id = mllm_model_id.strip()
        if model_id in vlm_models:
            return model_id
        for display_name, model_info in vlm_models.items():
            if model_info.get("model_name") == model_id:
                return display_name
        for display_name in vlm_models:
            if model_id in display_name:
                return display_name
        print(
            f"警告: MLLM_MODEL_ID='{model_id}' に一致する Vision モデルが見つかりません。"
            f" 先頭モデルを使用します: {next(iter(vlm_models))}"
        )

    return next(iter(vlm_models))


def build_vlm_ui_initialization():
    """両タブの VLM 設定 UI 用に共通の初期化データを構築する"""
    vlm_service = VLMService()
    provider_choices = vlm_service.get_available_service_providers()
    vlm_models = vlm_service.get_vlm_models()
    vlm_choices = list(vlm_models.keys()) if vlm_models else ["Vision対応モデルがありません"]
    default_vlm = resolve_default_vlm_display_name(vlm_models, os.getenv("MLLM_MODEL_ID"))

    all_models = vlm_service.model_settings
    non_vision_count = len(all_models) - len(vlm_models)
    print(f"✅ Vision対応モデル数: {len(vlm_models)}")
    print(f"❌ Vision非対応モデル数: {non_vision_count}")
    print(f"📊 総モデル数: {len(all_models)}")
    print(f"🎯 デフォルトVLMモデル: {default_vlm}")
    print(f"🌐 利用可能なプロバイダー: {provider_choices}")

    return vlm_choices, default_vlm, provider_choices, vlm_models, vlm_service


class VLMService:
    def __init__(self):
        self.model_settings_path = "model_settings.json"
        self.model_settings = self._load_model_settings()
        
        # 現在選択されているVLM設定を保持する辞書
        self.current_vlm_settings = {
            "model": None,
            "temperature": 0.0,
            "max_tokens": 4096,
            "oci_region": "ap-osaka-1"
        }
        
        # OCIリージョン定義
        self.OCI_REGIONS = {
            "Brazil East (Sao Paulo)": "sa-saopaulo-1",
            "Germany Central (Frankfurt)": "eu-frankfurt-1", 
            "Japan Central (Osaka)": "ap-osaka-1",
            "UAE East (Dubai)": "me-dubai-1",
            "UK South (London)": "uk-london-1",
            "US Midwest (Chicago)": "us-chicago-1"
        }
    
    def _load_model_settings(self):
        """モデル設定ファイルを読み込み"""
        try:
            with open(self.model_settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"モデル設定ファイル読み込みエラー: {e}")
            return {}
    
    def resolve_oci_region_id(self, oci_region):
        """Dropdown 表示名または region_id を API 用 region_id に正規化する"""
        if oci_region in self.OCI_REGIONS.values():
            return oci_region
        return self.OCI_REGIONS.get(oci_region, oci_region)

    def get_current_vlm_settings(self):
        """現在のVLM設定を取得"""
        # デフォルトモデルが設定されていない場合は、MLLM_MODEL_ID または先頭モデルを設定
        if self.current_vlm_settings["model"] is None:
            vlm_models = self.get_vlm_models()
            if vlm_models:
                first_model = resolve_default_vlm_display_name(vlm_models, os.getenv("MLLM_MODEL_ID"))
                self.current_vlm_settings["model"] = first_model
                self.current_vlm_settings["max_tokens"] = self.get_model_default_tokens(first_model)
                # Temperature の初期値をモデル設定から反映
                self.current_vlm_settings["temperature"] = self.get_model_default_temperature(first_model)
                api_type = self.get_api_type(first_model)
                if api_type.startswith("oci"):
                    default_region = self.get_model_default_region(first_model)
                    if default_region:
                        self.current_vlm_settings["oci_region"] = default_region
        
        return self.current_vlm_settings.copy()
    
    def update_current_vlm_settings(self, model=None, temperature=None, max_tokens=None, oci_region=None):
        """現在のVLM設定を更新"""
        if model is not None:
            self.current_vlm_settings["model"] = model
        if temperature is not None:
            self.current_vlm_settings["temperature"] = temperature
        if max_tokens is not None:
            self.current_vlm_settings["max_tokens"] = max_tokens
        if oci_region is not None:
            self.current_vlm_settings["oci_region"] = oci_region
    
    def get_vlm_models(self):
        """Vision対応モデルのみを取得（厳密なフィルタリング）"""
        vlm_models = {}
        
        for model_name, model_info in self.model_settings.items():
            # visionフィールドが明示的にTrueに設定されているもののみを取得
            is_vision_model = model_info.get("vision") is True
            
            if is_vision_model:
                vlm_models[model_name] = model_info
        
        return vlm_models
    
    def get_model_name(self, model_display_name):
        """表示名から実際のモデル名を取得"""
        return self.model_settings.get(model_display_name, {}).get("model_name", model_display_name)
    
    def get_api_type(self, model_display_name):
        """表示名からAPIタイプを取得"""
        return self.model_settings.get(model_display_name, {}).get("api_type", "unknown")
    
    def get_model_max_tokens(self, model_display_name):
        """モデルの最大トークン数を取得"""
        return self.model_settings.get(model_display_name, {}).get("max_tokens", 4096)
    
    def get_model_default_tokens(self, model_display_name):
        """モデルのデフォルトトークン数を取得"""
        return self.model_settings.get(model_display_name, {}).get("default_tokens", 4096)
    
    def get_model_vision_support(self, model_display_name):
        """モデルのVision対応可否を取得"""
        return self.model_settings.get(model_display_name, {}).get("vision", False)
    
    def get_model_default_region(self, model_display_name):
        """モデルのデフォルトリージョンを取得"""
        return self.model_settings.get(model_display_name, {}).get("default_region", None)

    def get_model_default_temperature(self, model_display_name):
        """モデルのデフォルト温度を取得
        
        優先順: default_temperature -> temperature -> 0.0
        値域は [0.0, 1.0] にクランプし、数値以外は 0.0 にフォールバック。
        """
        model_info = self.model_settings.get(model_display_name, {})
        # 優先順に取得
        raw_value = None
        if "default_temperature" in model_info:
            raw_value = model_info.get("default_temperature")
        elif "temperature" in model_info:
            raw_value = model_info.get("temperature")
        else:
            raw_value = 0.0

        # 数値化とクランプ
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0

        # クランプ
        if value < 0.0:
            value = 0.0
        if value > 1.0:
            value = 1.0
        return value
    
    def get_service_provider_from_api_type(self, api_type):
        """APIタイプからサービスプロバイダーを取得"""
        # 判定順序を修正：bedrockを先にチェック
        if "oci" in api_type.lower():
            return "OCI"
        elif "bedrock" in api_type.lower():
            return "AWS"
        elif "anthropic" in api_type.lower():
            return "Anthropic"
        elif "cohere" in api_type.lower():
            return "Cohere"
        elif "openai" in api_type.lower():
            return "OpenAI"
        else:
            return "Unknown"
    
    def filter_vlm_models_by_provider(self, service_provider):
        """サービスプロバイダーでVLMモデルをフィルタリング"""
        vlm_models = self.get_vlm_models()
        
        print(f"🔍 プロバイダーフィルタリング開始: {service_provider}")
        print(f"📋 対象Vision対応モデル数: {len(vlm_models)}")
        
        if service_provider == "すべて":
            filtered_models = list(vlm_models.keys())
            print(f"✅ 全てのVision対応モデルを返します: {len(filtered_models)}個")
            return filtered_models
        
        filtered_models = []
        for model_name, model_info in vlm_models.items():
            api_type = model_info.get("api_type", "")
            provider = self.get_service_provider_from_api_type(api_type)
            if provider == service_provider:
                filtered_models.append(model_name)
        
        print(f"✅ {service_provider}プロバイダーのVision対応モデル: {len(filtered_models)}個")
        if len(filtered_models) <= 3:
            for model in filtered_models:
                print(f"  - {model}")
        else:
            for model in filtered_models[:3]:
                print(f"  - {model}")
            print(f"  - ... その他 {len(filtered_models) - 3} 個")
        
        return filtered_models
    
    def get_available_service_providers(self):
        """利用可能なサービスプロバイダー一覧を取得（OCIを先頭に配置）"""
        vlm_models = self.get_vlm_models()
        service_providers = set()
        
        for model_info in vlm_models.values():
            api_type = model_info.get("api_type", "")
            provider = self.get_service_provider_from_api_type(api_type)
            if provider != "Unknown":
                service_providers.add(provider)
        
        # OCIを先頭にするカスタム順序を定義
        preferred_order = ["OCI", "Anthropic", "AWS", "OpenAI", "Cohere"]
        
        # 利用可能なプロバイダーを優先順序でソート
        ordered_providers = []
        for provider in preferred_order:
            if provider in service_providers:
                ordered_providers.append(provider)
        
        # 優先順序にないプロバイダーがあれば最後に追加
        remaining_providers = sorted(service_providers - set(ordered_providers))
        ordered_providers.extend(remaining_providers)
        
        return ["すべて"] + ordered_providers
    
    def service_provider_changed(self, service_provider):
        """サービスプロバイダー変更時の処理"""
        import gradio as gr
        
        filtered_models = self.filter_vlm_models_by_provider(service_provider)
        
        # フィルタリング後にモデルが空の場合は全VLMモデルを表示
        if not filtered_models:
            filtered_models = list(self.get_vlm_models().keys())
        
        # 最初のモデルを選択
        selected_model = filtered_models[0] if filtered_models else "モデルがありません"
        
        # モデルのドロップダウンを更新
        model_dropdown = gr.Dropdown(
            label="VLMモデル", 
            choices=filtered_models, 
            value=selected_model, 
            interactive=True
        )
        
        # 選択されたモデルに応じて他の設定も更新
        max_tokens_limit = self.get_model_max_tokens(selected_model)
        default_tokens = self.get_model_default_tokens(selected_model)
        default_temperature = self.get_model_default_temperature(selected_model)
        api_type = self.get_api_type(selected_model)
        
        # OCIモデルかどうかをチェック
        is_oci_model = api_type.startswith("oci")
        
        # モデルのデフォルトリージョンを取得
        default_region = self.get_model_default_region(selected_model)
        
        # リージョン設定の決定
        if is_oci_model:
            if default_region and default_region in self.OCI_REGIONS.values():
                # default_regionがOCI_REGIONSの値（region_id）に存在する場合、対応するキー（region_name）を取得
                region_name = None
                for name, region_id in self.OCI_REGIONS.items():
                    if region_id == default_region:
                        region_name = name
                        break
                selected_region = region_name if region_name else "Japan Central (Osaka)"
            else:
                selected_region = "Japan Central (Osaka)"
        else:
            selected_region = "Japan Central (Osaka)"
        
        # Temperature スライダー
        temperature_slider = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            value=default_temperature,
            interactive=True
        )

        max_tokens_slider = gr.Slider(
            label="Max tokens", 
            minimum=1, 
            maximum=max_tokens_limit, 
            step=1, 
            value=default_tokens, 
            interactive=True
        )
        
        oci_region_dropdown = gr.Dropdown(
            label="OCIリージョン", 
            choices=list(self.OCI_REGIONS.keys()), 
            value=selected_region, 
            interactive=True, 
            visible=is_oci_model
        )
        # 内部状態も同期
        self.update_current_vlm_settings(model=selected_model, temperature=default_temperature, max_tokens=default_tokens)

        return model_dropdown, temperature_slider, max_tokens_slider, oci_region_dropdown
    
    def model_changed(self, model):
        """モデル変更時の処理"""
        import gradio as gr
        
        max_tokens_limit = self.get_model_max_tokens(model)
        default_tokens = self.get_model_default_tokens(model)
        default_temperature = self.get_model_default_temperature(model)
        api_type = self.get_api_type(model)
        
        # OCIモデルかどうかをチェック
        is_oci_model = api_type.startswith("oci")
        
        # モデルのデフォルトリージョンを取得
        default_region = self.get_model_default_region(model)
        
        # リージョン設定の決定
        if is_oci_model:
            if default_region and default_region in self.OCI_REGIONS.values():
                region_name = None
                for name, region_id in self.OCI_REGIONS.items():
                    if region_id == default_region:
                        region_name = name
                        break
                selected_region = region_name if region_name else "Japan Central (Osaka)"
            else:
                selected_region = "Japan Central (Osaka)"
        else:
            selected_region = "Japan Central (Osaka)"
        # Temperature スライダー
        temperature_slider = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            value=default_temperature,
            interactive=True
        )

        max_tokens_slider = gr.Slider(
            label="Max tokens", 
            minimum=1, 
            maximum=max_tokens_limit, 
            step=1, 
            value=default_tokens, 
            interactive=True
        )
        
        oci_region_dropdown = gr.Dropdown(
            label="OCIリージョン", 
            choices=list(self.OCI_REGIONS.keys()), 
            value=selected_region, 
            interactive=True, 
            visible=is_oci_model
        )
        # 内部状態も同期
        self.update_current_vlm_settings(temperature=default_temperature, max_tokens=default_tokens)

        return temperature_slider, max_tokens_slider, oci_region_dropdown