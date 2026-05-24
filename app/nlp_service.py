import spacy
import threading
import base64
import os
from io import BytesIO
from PIL import Image
from app.vlm_service import VLMService
from oci.generative_ai_inference.models import TextContent, ImageContent, ImageUrl, UserMessage, GenericChatRequest, BaseChatRequest

class NLPService:
    """spaCyモデルとVLMキャプション生成を管理するサービスクラス
    
    依存関係注入パターンを使用してVLMServiceインスタンスを外部から受け取り、
    各タブで独立したVLM設定を使用できるようにします。
    spaCyモデルはインスタンス内でスレッドセーフに管理されます。
    """
    
    def __init__(self, vlm_service_instance=None):
        """NLPServiceを初期化
        
        Args:
            vlm_service_instance (VLMService, optional): 使用するVLMServiceインスタンス。
                                                       Noneの場合は新しいインスタンスを作成。
        """
        self.vlm_service = vlm_service_instance or VLMService()
        self._nlp = None
        self._lock = threading.Lock()
    
    def get_nlp(self):
        """spaCyのja_ginzaモデルを取得
        
        初回呼び出し時にモデルをロードし、以降は同じインスタンスを返します。
        スレッドセーフな実装により、マルチセッション環境でも安全に使用できます。
        
        Returns:
            spacy.Language: ja_ginzaモデルのインスタンス
        """
        if self._nlp is None:
            with self._lock:
                if self._nlp is None:
                    print("spaCy ja_ginzaモデルを初期化中...")
                    self._nlp = spacy.load("ja_ginza")
                    print("spaCy ja_ginzaモデルの初期化が完了しました。")
        return self._nlp
    
    def is_initialized(self):
        """モデルが初期化済みかどうかを確認
        
        Returns:
            bool: 初期化済みの場合True
        """
        return self._nlp is not None
    
    def generate_caption_with_vlm(self, image_path, vlm_model, prompt_text, temperature=0.3, max_tokens=4096, oci_region="Japan Central (Osaka)"):
        """VLMを使用して画像のキャプションを生成"""
        try:
            # VLMサービスからモデル情報を取得
            api_type = self.vlm_service.get_api_type(vlm_model)
            model_name = self.vlm_service.get_model_name(vlm_model)
            # 画像をBase64エンコード
            image_data_url = self._image_to_base64_data_url(image_path)
            # APIタイプに応じてキャプション生成
            if api_type.startswith("anthropic"):
                return self._generate_caption_anthropic(model_name, image_data_url, prompt_text, temperature, max_tokens)
            elif api_type.startswith("oci"):
                return self._generate_caption_oci(vlm_model, model_name, image_data_url, prompt_text, temperature, max_tokens, oci_region)
            elif api_type.startswith("openai"):
                return self._generate_caption_openai(model_name, api_type, image_data_url, prompt_text, temperature, max_tokens)
            elif "bedrock" in api_type.lower():
                return self._generate_caption_bedrock(model_name, image_data_url, prompt_text, temperature, max_tokens)
            elif "vertex" in api_type.lower() or "google" in api_type.lower():
                return self._generate_caption_vertex(model_name, image_data_url, prompt_text, temperature, max_tokens)
            else:
                return f"エラー: サポートされていないAPIタイプ: {api_type}"
        except Exception as e:
            print(f"VLMキャプション生成エラー: {e}")
            return f"キャプション生成中にエラーが発生しました: {str(e)}"

    def generate_answer_with_vlm_images(self, image_paths, vlm_model, prompt_text, temperature=0.3, max_tokens=4096, oci_region="Japan Central (Osaka)"):
        """複数画像をVLMに渡して回答を生成する"""
        try:
            image_paths = [path for path in (image_paths or []) if path]
            if not image_paths:
                return "エラー: VLMに渡す画像がありません"
            if len(image_paths) == 1:
                return self.generate_caption_with_vlm(
                    image_paths[0],
                    vlm_model,
                    prompt_text,
                    temperature,
                    max_tokens,
                    oci_region,
                )

            api_type = self.vlm_service.get_api_type(vlm_model)
            model_name = self.vlm_service.get_model_name(vlm_model)
            image_data_urls = [self._image_to_base64_data_url(image_path) for image_path in image_paths]

            if api_type.startswith("anthropic"):
                return self._generate_multi_image_answer_anthropic(model_name, image_data_urls, prompt_text, temperature, max_tokens)
            elif api_type.startswith("oci"):
                return self._generate_multi_image_answer_oci(vlm_model, model_name, image_data_urls, prompt_text, temperature, max_tokens, oci_region)
            elif api_type.startswith("openai"):
                return self._generate_multi_image_answer_openai(model_name, api_type, image_data_urls, prompt_text, temperature, max_tokens)
            elif "bedrock" in api_type.lower():
                return self._generate_multi_image_answer_bedrock(model_name, image_data_urls, prompt_text, temperature, max_tokens)
            elif "vertex" in api_type.lower() or "google" in api_type.lower():
                return self._generate_multi_image_answer_vertex(model_name, image_data_urls, prompt_text, temperature, max_tokens)
            else:
                return f"エラー: サポートされていないAPIタイプ: {api_type}"
        except Exception as e:
            print(f"複数画像VLM回答生成エラー: {e}")
            return f"複数画像回答生成中にエラーが発生しました: {str(e)}"

    def _split_data_url(self, image_data_url):
        media_type = image_data_url.split(';')[0].split(':')[1]
        base64_data = image_data_url.split(',')[1]
        return media_type, base64_data

    def _generate_multi_image_answer_anthropic(self, model_name, image_data_urls, prompt_text, temperature, max_tokens):
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            content = []
            for image_data_url in image_data_urls:
                media_type, base64_data = self._split_data_url(image_data_url)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                })
            content.append({"type": "text", "text": prompt_text})

            params = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if model_name != "claude-opus-4-7":
                params["temperature"] = temperature

            response = client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API エラー: {str(e)}"

    def _generate_multi_image_answer_openai(self, model_name, api_type, image_data_urls, prompt_text, temperature, max_tokens):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if "reasoning" in api_type:
                content = [{"type": "input_text", "text": prompt_text}]
                content.extend({"type": "input_image", "image_url": image_data_url} for image_data_url in image_data_urls)
                response = client.responses.create(
                    model=model_name,
                    input=[{"role": "user", "content": content}],
                    reasoning={"effort": "medium"},
                )
                return response.output_text

            content = [{"type": "text", "text": prompt_text}]
            content.extend({"type": "image_url", "image_url": {"url": image_data_url}} for image_data_url in image_data_urls)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API エラー: {str(e)}"

    def _generate_multi_image_answer_oci(self, model_display_name, model_name, image_data_urls, prompt_text, temperature, max_tokens, oci_region):
        try:
            import oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient
            from oci.generative_ai_inference.models import ChatDetails, OnDemandServingMode

            region_id = self.vlm_service.resolve_oci_region_id(oci_region)
            config = oci.config.from_file()
            config["region"] = region_id
            client = GenerativeAiInferenceClient(config)

            compartment_id = os.getenv("OCI_COMPARTMENT_ID")
            if not compartment_id:
                return "エラー: OCI_COMPARTMENT_IDが設定されていません"

            text_content = TextContent()
            text_content.text = prompt_text
            content = [text_content]
            for image_data_url in image_data_urls:
                media_type, base64_data = self._split_data_url(image_data_url)
                image_content = ImageContent()
                image_url_obj = ImageUrl()
                image_url_obj.url = f"data:{media_type};base64,{base64_data}"
                image_content.image_url = image_url_obj
                content.append(image_content)

            message = UserMessage()
            message.content = content

            chat_request = GenericChatRequest()
            chat_request.messages = [message]
            chat_request.api_format = BaseChatRequest.API_FORMAT_GENERIC
            chat_request.num_generations = 1
            chat_request.max_tokens = max_tokens
            chat_request.is_stream = False
            chat_request.temperature = temperature
            chat_request.top_p = 1.0 if self.vlm_service.get_api_type(model_display_name) in ["oci.xai.chat", "oci.gemini.chat"] else 0.9
            if self.vlm_service.get_api_type(model_display_name) == "oci.llama.chat":
                chat_request.top_k = -1
                chat_request.frequency_penalty = 0.5
                chat_request.presence_penalty = 0.5

            chat_details = ChatDetails()
            chat_details.serving_mode = OnDemandServingMode(model_id=model_name)
            chat_details.compartment_id = compartment_id
            chat_details.chat_request = chat_request

            response = client.chat(chat_details)
            if hasattr(response, 'data') and hasattr(response.data, 'chat_response'):
                if hasattr(response.data.chat_response, 'choices') and len(response.data.chat_response.choices) > 0:
                    choice = response.data.chat_response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        for content_item in choice.message.content:
                            if hasattr(content_item, 'text'):
                                return content_item.text
            return f"OCI APIからのレスポンス構造が予期しない形式です: {type(response.data)}"
        except Exception as e:
            return f"OCI API エラー: {str(e)}"

    def _generate_multi_image_answer_bedrock(self, model_name, image_data_urls, prompt_text, temperature, max_tokens):
        try:
            import boto3
            import json

            bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
            if "claude" not in model_name.lower():
                return self._generate_caption_bedrock(model_name, image_data_urls[0], prompt_text, temperature, max_tokens)

            content = [{"type": "text", "text": prompt_text}]
            for image_data_url in image_data_urls:
                media_type, base64_data = self._split_data_url(image_data_url)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                })

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content}],
            }
            response = bedrock_runtime.invoke_model(
                modelId=model_name,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            response_body = json.loads(response['body'].read())
            for content_item in response_body.get('content', []):
                if content_item.get('type') == 'text':
                    return content_item.get('text', '')
            return response_body.get('completion', '')
        except Exception as e:
            return f"Bedrock API エラー: {str(e)}"

    def _generate_multi_image_answer_vertex(self, model_name, image_data_urls, prompt_text, temperature, max_tokens):
        try:
            if "gemini" not in model_name.lower():
                return self._generate_caption_vertex(model_name, image_data_urls[0], prompt_text, temperature, max_tokens)

            import base64
            import google.cloud.aiplatform as aiplatform
            from vertexai.preview.generative_models import GenerativeModel, Part

            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if not project_id:
                return "エラー: GOOGLE_CLOUD_PROJECTが設定されていません"

            aiplatform.init(project=project_id, location=location)
            parts = [Part.from_text(prompt_text)]
            for image_data_url in image_data_urls:
                media_type, base64_data = self._split_data_url(image_data_url)
                parts.append(Part.from_data(data=base64.b64decode(base64_data), mime_type=media_type))

            model = GenerativeModel(model_name)
            response = model.generate_content(
                parts,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.9,
                },
            )
            return response.text
        except Exception as e:
            return f"Vertex AI エラー: {str(e)}"
    
    def _image_to_base64_data_url(self, image_path):
        """画像をBase64データURLに変換（WebP対応強化）"""
        import os
        verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')
        
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
        except Exception as e:
            error_msg = f"画像ファイル読み込みエラー: {image_path} - {str(e)}"
            if verbose:
                print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
            
        # 画像をPILで開く
        try:
            image = Image.open(BytesIO(image_data))
            original_format = image.format.lower() if image.format else 'unknown'
            original_mode = image.mode
            original_size = image.size
            
            if verbose:
                print(f"[DEBUG] 画像変換処理開始: {os.path.basename(image_path)}")
                print(f"[DEBUG] 元の形式: {original_format}, モード: {original_mode}, サイズ: {original_size}")
        except Exception as e:
            error_msg = f"画像データ解析エラー: {os.path.basename(image_path)} - {str(e)}"
            if verbose:
                print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        # WebP画像やアルファチャンネルを持つ画像はJPEG形式に変換
        if original_format == 'webp' or image.mode in ('RGBA', 'LA'):
            if verbose:
                print(f"[DEBUG] WebP/アルファチャンネル画像を検出 - JPEG形式に変換中")
            
            # アルファチャンネルがある場合は白い背景に合成
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:  # LA (Luminance + Alpha)
                    background.paste(image, mask=image.split()[-1])
                image = background
                if verbose:
                    print(f"[DEBUG] アルファチャンネルを白背景に合成: {original_mode} -> RGB")
            elif image.mode not in ('RGB', 'L'):
                # その他のモードもRGBに変換
                image = image.convert('RGB')
                if verbose:
                    print(f"[DEBUG] カラーモード変換: {original_mode} -> RGB")
            
            # JPEG形式で保存
            try:
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=95)
                base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                if verbose:
                    jpeg_size = len(buffered.getvalue())
                    print(f"[DEBUG] JPEG変換完了: サイズ {jpeg_size} bytes, Base64長 {len(base64_data)}")
                
                return f"data:image/jpeg;base64,{base64_data}"
            except Exception as e:
                error_msg = f"WebP→JPEG変換エラー: {os.path.basename(image_path)} - {str(e)}"
                if verbose:
                    print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)
        
        # JPEG/PNGの場合は元のデータを使用（品質劣化を避けるため）
        elif original_format in ['jpeg', 'jpg']:
            base64_data = base64.b64encode(image_data).decode('utf-8')
            if verbose:
                print(f"[DEBUG] JPEG画像: 元データを使用, Base64長 {len(base64_data)}")
            return f"data:image/jpeg;base64,{base64_data}"
        elif original_format == 'png':
            base64_data = base64.b64encode(image_data).decode('utf-8')
            if verbose:
                print(f"[DEBUG] PNG画像: 元データを使用, Base64長 {len(base64_data)}")
            return f"data:image/png;base64,{base64_data}"
        else:
            # その他の形式はJPEG形式に変換
            if verbose:
                print(f"[DEBUG] 未知の形式 {original_format} - JPEG形式に変換中")
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
                if verbose:
                    print(f"[DEBUG] カラーモード変換: {original_mode} -> RGB")
            
            try:
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=95)
                base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                if verbose:
                    jpeg_size = len(buffered.getvalue())
                    print(f"[DEBUG] JPEG変換完了: サイズ {jpeg_size} bytes, Base64長 {len(base64_data)}")
                
                return f"data:image/jpeg;base64,{base64_data}"
            except Exception as e:
                error_msg = f"その他形式→JPEG変換エラー: {os.path.basename(image_path)} - {str(e)}"
                if verbose:
                    print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)
    
    def _generate_caption_anthropic(self, model_name, image_data_url, prompt_text, temperature, max_tokens):
        """Anthropic APIを使用してキャプション生成"""
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            # 画像データからmedia_typeとdataを抽出
            media_type = image_data_url.split(';')[0].split(':')[1]
            base64_data = image_data_url.split(',')[1]
            
            params = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    }
                ]
            }
            if model_name != "claude-opus-4-7":
                params["temperature"] = temperature
            
            response = client.messages.create(**params)
            
            return response.content[0].text
            
        except Exception as e:
            return f"Anthropic API エラー: {str(e)}"
    
    def _generate_caption_oci(self, model_display_name, model_name, image_data_url, prompt_text, temperature, max_tokens, oci_region):
        api_type = self.vlm_service.get_api_type(model_display_name)
        if api_type in ["oci.llama.chat"]:
            try:
                media_type = image_data_url.split(';')[0].split(':')[1]
                base64_data = image_data_url.split(',')[1]
            except Exception:
                return "エラー: 画像データの形式が不正です"

            try:
                import oci
                from oci.generative_ai_inference import GenerativeAiInferenceClient
                from oci.generative_ai_inference.models import ChatDetails, OnDemandServingMode
                import os

                # OCIリージョンIDを取得
                # oci_regionが既にregion_id（例：us-chicago-1）かregion_name（例：US Midwest (Chicago)）かを判定
                import os
                verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')
                
                if oci_region in self.vlm_service.OCI_REGIONS.values():
                    # 既にregion_idの場合はそのまま使用
                    region_id = oci_region
                    if verbose:
                        print(f"[DEBUG] OCIリージョン: region_idとして使用 {oci_region}")
                else:
                    # region_nameの場合は変換
                    region_id = self.vlm_service.OCI_REGIONS.get(oci_region, "ap-osaka-1")
                    if verbose:
                        print(f"[DEBUG] OCIリージョン: {oci_region} -> {region_id} に変換")
                
                # OCI設定
                config = oci.config.from_file()
                config["region"] = region_id
                
                client = GenerativeAiInferenceClient(config)
                
                # Compartment IDを取得
                compartment_id = os.getenv("OCI_COMPARTMENT_ID")
                if not compartment_id:
                    return "エラー: OCI_COMPARTMENT_IDが設定されていません"
                
                # TextContentとImageContentを作成
                content1 = TextContent()
                content1.text = prompt_text
                content2 = ImageContent()
                image_url_obj = ImageUrl()
                image_url_obj.url = f"data:{media_type};base64,{base64_data}"
                content2.image_url = image_url_obj
                message = UserMessage()
                message.content = [content1, content2]

                chat_request = GenericChatRequest()
                chat_request.messages = [message]
                chat_request.api_format = BaseChatRequest.API_FORMAT_GENERIC
                chat_request.num_generations = 1
                chat_request.max_tokens = max_tokens
                chat_request.is_stream = False
                chat_request.temperature = temperature
                chat_request.top_p = 0.9
                chat_request.top_k = -1
                chat_request.frequency_penalty = 0.5
                chat_request.presence_penalty = 0.5

                chat_details = ChatDetails()
                chat_details.serving_mode = OnDemandServingMode(model_id=model_name)
                chat_details.compartment_id = compartment_id
                chat_details.chat_request = chat_request

                # APIリクエスト実行
                response = client.chat(chat_details)

                # レスポンス処理（元のdatabase_service方式）
                if hasattr(response, 'data') and hasattr(response.data, 'chat_response'):
                    if hasattr(response.data.chat_response, 'choices') and len(response.data.chat_response.choices) > 0:
                        choice = response.data.chat_response.choices[0]
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            for content in choice.message.content:
                                if hasattr(content, 'text'):
                                    return content.text
                return f"OCI APIからのレスポンス構造が予期しない形式です: {type(response.data)}"

            except Exception as e:
                import traceback
                import os
                verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')
                error_details = traceback.format_exc()
                if verbose:
                    print(f"[DEBUG] OCI API詳細エラー: {error_details}")
                return f"OCI API エラー: {str(e)}"
        elif api_type in ["oci.xai.chat"]:
            try:
                media_type = image_data_url.split(';')[0].split(':')[1]
                base64_data = image_data_url.split(',')[1]
            except Exception:
                return "エラー: 画像データの形式が不正です"

            try:
                import oci
                from oci.generative_ai_inference import GenerativeAiInferenceClient
                from oci.generative_ai_inference.models import ChatDetails, OnDemandServingMode
                import os

                # OCIリージョンIDを取得
                # oci_regionが既にregion_id（例：us-chicago-1）かregion_name（例：US Midwest (Chicago)）かを判定
                import os
                verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')
                
                if oci_region in self.vlm_service.OCI_REGIONS.values():
                    # 既にregion_idの場合はそのまま使用
                    region_id = oci_region
                    if verbose:
                        print(f"[DEBUG] OCIリージョン: region_idとして使用 {oci_region}")
                else:
                    # region_nameの場合は変換
                    region_id = self.vlm_service.OCI_REGIONS.get(oci_region, "ap-osaka-1")
                    if verbose:
                        print(f"[DEBUG] OCIリージョン: {oci_region} -> {region_id} に変換")
                
                # OCI設定
                config = oci.config.from_file()
                config["region"] = region_id
                
                client = GenerativeAiInferenceClient(config)
                
                # Compartment IDを取得
                compartment_id = os.getenv("OCI_COMPARTMENT_ID")
                if not compartment_id:
                    return "エラー: OCI_COMPARTMENT_IDが設定されていません"
                
                # TextContentとImageContentを作成
                content1 = TextContent()
                content1.text = prompt_text
                content2 = ImageContent()
                image_url_obj = ImageUrl()
                image_url_obj.url = f"data:{media_type};base64,{base64_data}"
                content2.image_url = image_url_obj
                message = UserMessage()
                message.content = [content1, content2]

                chat_request = GenericChatRequest()
                chat_request.messages = [message]
                chat_request.api_format = BaseChatRequest.API_FORMAT_GENERIC
                chat_request.num_generations = 1
                chat_request.max_tokens = max_tokens
                chat_request.is_stream = False
                chat_request.temperature = temperature
                chat_request.top_p = 1.0
                # chat_request.top_k = -1
                # chat_request.frequency_penalty = 0.5
                # chat_request.presence_penalty = 0.5

                chat_details = ChatDetails()
                chat_details.serving_mode = OnDemandServingMode(model_id=model_name)
                chat_details.compartment_id = compartment_id
                chat_details.chat_request = chat_request

                # APIリクエスト実行
                response = client.chat(chat_details)

                # レスポンス処理（元のdatabase_service方式）
                if hasattr(response, 'data') and hasattr(response.data, 'chat_response'):
                    if hasattr(response.data.chat_response, 'choices') and len(response.data.chat_response.choices) > 0:
                        choice = response.data.chat_response.choices[0]
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            for content in choice.message.content:
                                if hasattr(content, 'text'):
                                    return content.text
                return f"OCI APIからのレスポンス構造が予期しない形式です: {type(response.data)}"

            except Exception as e:
                import traceback
                import os
                verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')
                error_details = traceback.format_exc()
                if verbose:
                    print(f"[DEBUG] OCI API詳細エラー: {error_details}")
                return f"OCI API エラー: {str(e)}"
        elif api_type in ["oci.gemini.chat"]:
            try:
                media_type = image_data_url.split(';')[0].split(':')[1]
                base64_data = image_data_url.split(',')[1]
            except Exception:
                return "エラー: 画像データの形式が不正です"

            try:
                import oci
                from oci.generative_ai_inference import GenerativeAiInferenceClient
                from oci.generative_ai_inference.models import ChatDetails, OnDemandServingMode
                import os

                verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')

                if oci_region in self.vlm_service.OCI_REGIONS.values():
                    region_id = oci_region
                    if verbose:
                        print(f"[DEBUG] OCIリージョン: region_idとして使用 {oci_region}")
                else:
                    region_id = self.vlm_service.OCI_REGIONS.get(oci_region, "us-chicago-1")
                    if verbose:
                        print(f"[DEBUG] OCIリージョン: {oci_region} -> {region_id} に変換")

                config = oci.config.from_file()
                config["region"] = region_id

                client = GenerativeAiInferenceClient(config)

                compartment_id = os.getenv("OCI_COMPARTMENT_ID")
                if not compartment_id:
                    return "エラー: OCI_COMPARTMENT_IDが設定されていません"

                content1 = TextContent()
                content1.text = prompt_text
                content2 = ImageContent()
                image_url_obj = ImageUrl()
                image_url_obj.url = f"data:{media_type};base64,{base64_data}"
                content2.image_url = image_url_obj
                message = UserMessage()
                message.content = [content1, content2]

                chat_request = GenericChatRequest()
                chat_request.messages = [message]
                chat_request.api_format = BaseChatRequest.API_FORMAT_GENERIC
                chat_request.num_generations = 1
                chat_request.max_tokens = max_tokens
                chat_request.is_stream = False
                chat_request.temperature = temperature
                chat_request.top_p = 1.0

                chat_details = ChatDetails()
                chat_details.serving_mode = OnDemandServingMode(model_id=model_name)
                chat_details.compartment_id = compartment_id
                chat_details.chat_request = chat_request

                response = client.chat(chat_details)

                if hasattr(response, 'data') and hasattr(response.data, 'chat_response'):
                    if hasattr(response.data.chat_response, 'choices') and len(response.data.chat_response.choices) > 0:
                        choice = response.data.chat_response.choices[0]
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            for content in choice.message.content:
                                if hasattr(content, 'text'):
                                    return content.text
                return f"OCI APIからのレスポンス構造が予期しない形式です: {type(response.data)}"

            except Exception as e:
                import traceback
                import os
                verbose = os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes')
                error_details = traceback.format_exc()
                if verbose:
                    print(f"[DEBUG] OCI Gemini API詳細エラー: {error_details}")
                return f"OCI Gemini API エラー: {str(e)}"
        else:
            return "エラー: このモデルはOCI Generative AI の Vision 対応モデルとして認識されませんでした"

    def _generate_caption_openai(self, model_name, api_type, image_data_url, prompt_text, temperature, max_tokens):
        """OpenAI APIを使用してキャプション生成"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            if "reasoning" in api_type:
                # Reasoningモデル用のエンドポイント呼び出し
                params = {
                    "model": model_name,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt_text},
                                {"type": "input_image", "image_url": image_data_url}
                            ]
                        }
                    ],
                    "reasoning": {"effort": "medium"}  # デフォルト値を設定
                }
                
                response = client.responses.create(**params)
                return response.output_text

            else:
                # Chatモデル用のエンドポイント呼び出し
                params = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": image_data_url}}
                            ]
                        }
                    ]
                }
                
                # パラメータを動的に設定
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature
                
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content
            
        except Exception as e:
            return f"OpenAI API エラー: {str(e)}"

    def _generate_caption_bedrock(self, model_name, image_data_url, prompt_text, temperature, max_tokens):
        """Bedrock APIを使用してキャプション生成"""
        try:
            import boto3
            import json
            import base64
            
            # Bedrock Runtimeクライアントを作成
            bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-west-2'  # Bedrockが利用可能なリージョン
            )
            
            # 画像データからBase64データを抽出
            try:
                media_type = image_data_url.split(';')[0].split(':')[1]
                base64_data = image_data_url.split(',')[1]
            except:
                return "エラー: 画像データの形式が不正です"
            
            # モデルに応じてリクエスト形式を調整
            if "claude" in model_name.lower():
                # Anthropic Claude Vision の場合
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data
                                    }
                                }
                            ]
                        }
                    ]
                }
            elif "llama" in model_name.lower():
                # Meta Llama Vision の場合
                body = {
                    "prompt": f"<image>{base64_data}</image>\n{prompt_text}",
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
            else:
                # その他のモデル（汎用）
                body = {
                    "inputText": prompt_text,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": 0.9
                    },
                    "image": base64_data
                }
            
            # API呼び出し
            response = bedrock_runtime.invoke_model(
                modelId=model_name,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            # レスポンスを解析
            response_body = json.loads(response['body'].read())
            
            # モデルタイプに応じてレスポンスからテキストを抽出
            if "claude" in model_name.lower():
                if 'content' in response_body:
                    for content_item in response_body['content']:
                        if content_item.get('type') == 'text':
                            return content_item.get('text', '')
                return response_body.get('completion', '')
            elif "llama" in model_name.lower():
                return response_body.get('generation', '')
            else:
                # その他のモデル
                return response_body.get('results', [{}])[0].get('outputText', '')
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Bedrock API詳細エラー: {error_details}")
            return f"Bedrock API エラー: {str(e)}"

    def _generate_caption_vertex(self, model_name, image_data_url, prompt_text, temperature, max_tokens):
        """Vertex AIを使用してキャプション生成"""
        try:
            import os
            import google.cloud.aiplatform as aiplatform
            import base64
            import json
            
            # プロジェクトIDとリージョンを設定
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            if not project_id:
                return "エラー: GOOGLE_CLOUD_PROJECTが設定されていません"
            
            # Vertex AIクライアントを初期化
            aiplatform.init(project=project_id, location=location)
            
            # 画像データからBase64データを抽出
            try:
                media_type = image_data_url.split(';')[0].split(':')[1]
                base64_data = image_data_url.split(',')[1]
            except:
                return "エラー: 画像データの形式が不正です"
            
            # モデルに応じてリクエスト形式を調整
            if "gemini" in model_name.lower():
                # Gemini Pro Vision の場合
                from vertexai.preview.generative_models import GenerativeModel, Part
                model = GenerativeModel(model_name)
                
                # 画像パートを作成
                image_part = Part.from_data(
                    data=base64.b64decode(base64_data),
                    mime_type=media_type
                )
                
                # テキストプロンプト
                text_part = Part.from_text(prompt_text)
                
                # 生成設定
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.9
                }
                
                # 予測実行
                response = model.generate_content(
                    [text_part, image_part],
                    generation_config=generation_config
                )
                
                return response.text
                
            else:
                # その他のモデル（汎用）
                return f"Vertex AI: モデル {model_name} はサポートされていません"
             
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Vertex AI詳細エラー: {error_details}")
            return f"Vertex AI エラー: {str(e)}" 