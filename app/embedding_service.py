from io import BytesIO
import base64
from PIL import Image
import oci
import array

class EmbeddingService:
    def __init__(self, embed_model_provider, embed_model_id, compartment_id, cohere_client=None, oci_client=None):
        self.embed_model_provider = embed_model_provider
        self.embed_model_id = embed_model_id
        self.compartment_id = compartment_id
        self.cohere_client = cohere_client
        self.oci_client = oci_client
        
    def get_text_embedding(self, text, input_type="search_query"):
        """クエリーテキストから埋め込みベクトルを生成"""
        if self.embed_model_provider == "CohereAI":
            return self._get_text_embedding_cohere(text, input_type)
        elif self.embed_model_provider == "OCI":
            return self._get_text_embedding_oci(text, input_type)
        else:
            raise ValueError(f"サポートされていない埋め込みモデルプロバイダーです: {self.embed_model_provider}")
    
    def get_image_embedding(self, image):
        """アップロードされた画像から埋め込みベクトルを生成"""
        if self.embed_model_provider == "CohereAI":
            return self._get_image_embedding_cohere(image)
        elif self.embed_model_provider == "OCI":
            return self._get_image_embedding_oci(image)
        else:
            raise ValueError(f"サポートされていない埋め込みモデルプロバイダーです: {self.embed_model_provider}")
            
    def _get_text_embedding_cohere(self, text, input_type="search_query"):
        """CohereAIを使用してテキストの埋め込みベクトルを生成"""
        response = self.cohere_client.embed(
            texts=[text],
            model=self.embed_model_id,
            input_type=input_type
        )
        return response.embeddings[0]
    
    def _get_image_embedding_cohere(self, image):
        """CohereAIを使用して画像の埋め込みベクトルを生成"""
        # 画像データをPILからBase64エンコードしてData URLに変換
        # RGBA形式の場合はRGB形式に変換（JPEG形式はアルファチャンネルをサポートしないため）
        if image.mode in ('RGBA', 'LA'):
            # 白い背景に合成
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])  # アルファチャンネルをマスクとして使用
            else:  # LA (Luminance + Alpha)
                background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode not in ('RGB', 'L'):
            # その他のモードもRGBに変換
            image = image.convert('RGB')
            
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_base64}"
        
        # Cohere APIを使用して画像の埋め込みベクトルを取得
        response = self.cohere_client.embed(
            images=[data_url],
            model=self.embed_model_id,
            input_type="image",
            embedding_types=["float"],
        )
        return response.embeddings.float[0]
    
    def _get_text_embedding_oci(self, text, input_type="search_query"):
        """OCIを使用してテキストの埋め込みベクトルを生成"""
        # input_typeをOCI APIに適した形式に変換
        oci_input_type = "SEARCH_QUERY" if input_type == "search_query" else "SEARCH_DOCUMENT"
        
        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.embed_model_id)
        embed_text_detail.compartment_id = self.compartment_id
        embed_text_detail.inputs = [text]
        embed_text_detail.input_type = oci_input_type
        embed_text_detail.truncate = "END"
        
        embedding_response = self.oci_client.embed_text(embed_text_detail)
        return embedding_response.data.embeddings[0]
    
    def _get_image_embedding_oci(self, image):
        """OCIを使用して画像の埋め込みベクトルを生成"""
        # 画像データをPILからBase64エンコードしてData URLに変換
        # RGBA形式の場合はRGB形式に変換（JPEG形式はアルファチャンネルをサポートしないため）
        if image.mode in ('RGBA', 'LA'):
            # 白い背景に合成
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])  # アルファチャンネルをマスクとして使用
            else:  # LA (Luminance + Alpha)
                background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode not in ('RGB', 'L'):
            # その他のモードもRGBに変換
            image = image.convert('RGB')
            
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_base64}"
        
        # 画像埋め込みもEmbedTextDetailsを使用し、画像データをinputsに渡す
        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.embed_model_id)
        embed_text_detail.compartment_id = self.compartment_id
        embed_text_detail.inputs = [data_url]  # 画像のData URLを文字列として渡す
        embed_text_detail.input_type = "IMAGE"
        
        embedding_response = self.oci_client.embed_text(embed_text_detail)
        return embedding_response.data.embeddings[0] 