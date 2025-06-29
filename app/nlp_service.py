import spacy
import threading
import base64
import os
from io import BytesIO
from PIL import Image
from app.vlm_service import VLMService
from oci.generative_ai_inference.models import TextContent, ImageContent, ImageUrl, UserMessage, GenericChatRequest, BaseChatRequest

class NLPService:
    """spaCyモデルとVLMキャプション生成をシングルトンパターンで管理するサービスクラス
    
    マルチセッション環境で重いspaCyモデルの重複初期化を防ぐため、
    アプリケーション全体で1つのモデルインスタンスを共有します。
    """
    _instance = None
    _nlp = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.vlm_service = VLMService()
        return cls._instance
    
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
    
    def _image_to_base64_data_url(self, image_path):
        """画像をBase64データURLに変換"""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        # 画像形式を判定
        image = Image.open(BytesIO(image_data))
        image_format = image.format.lower()
        
        # Base64エンコード
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        # データURLを作成
        if image_format == 'jpeg':
            return f"data:image/jpeg;base64,{base64_data}"
        elif image_format == 'png':
            return f"data:image/png;base64,{base64_data}"
        else:
            return f"data:image/{image_format};base64,{base64_data}"
    
    def _generate_caption_anthropic(self, model_name, image_data_url, prompt_text, temperature, max_tokens):
        """Anthropic APIを使用してキャプション生成"""
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            # 画像データからmedia_typeとdataを抽出
            media_type = image_data_url.split(';')[0].split(':')[1]
            base64_data = image_data_url.split(',')[1]
            
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
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
            )
            
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
                region_id = self.vlm_service.OCI_REGIONS.get(oci_region, "ap-osaka-1")
                
                # OCI設定
                config = oci.config.from_file()
                config["region"] = region_id
                
                client = GenerativeAiInferenceClient(config)
                
                # Compartment IDを取得
                compartment_id = os.getenv("OCI_COMPARTMENT_ID")
                if not compartment_id:
                    return "エラー: OCI_COMPARTMENT_IDが設定されていません"
                
                # TextContentとImageContentを作成（元のdatabase_service方式）
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
                error_details = traceback.format_exc()
                print(f"[DEBUG] OCI API詳細エラー: {error_details}")
                return f"OCI API エラー: {str(e)}"
        else:
            return "エラー: このモデルはOCI Llama Visionとして認識されませんでした"

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
                    "reasoning": {"effort": "medium"}  # サンプルに基づき固定値を設定
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