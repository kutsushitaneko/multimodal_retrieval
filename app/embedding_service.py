from io import BytesIO
import base64
from PIL import Image

class EmbeddingService:
    def __init__(self, cohere_client):
        self.cohere_client = cohere_client
        
    def get_text_embedding(self, text, input_type="search_query"):
        """クエリーテキストからCohere Embed 4.0を使用しての埋め込みベクトルを生成"""
        response = self.cohere_client.embed(
            texts=[text],
            #model="embed-v4.0",
            model="embed-multilingual-v3.0",
            input_type=input_type
        )
        
        return response.embeddings[0]
        
    def get_image_embedding(self, image):
        """アップロードされた画像からCohere Embed 4.0を使用して埋め込みベクトルを生成"""
        # 画像データをPILからBase64エンコードしてData URLに変換
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_base64}"
        
        # Cohere APIを使用して画像の埋め込みベクトルを取得
        response = self.cohere_client.embed(
            images=[data_url],
            #model="embed-v4.0",
            model="embed-multilingual-v3.0",
            input_type="image",
            embedding_types=["float"],
        )
        
        return response.embeddings.float[0] 