"""
VLMServiceファクトリーモジュール

各タブで独立したVLMServiceインスタンスを作成・管理するためのファクトリークラス。
タブ間での設定共有を完全に排除し、独立したVLM設定を実現する。
"""

from app.vlm_service import VLMService


class VLMServiceFactory:
    """VLMServiceのインスタンスを管理するファクトリークラス
    
    このクラスは各タブ・機能に特化したVLMServiceインスタンスを作成します。
    シングルトンパターンを使用せず、完全に独立したインスタンスを提供することで、
    タブ間での設定干渉を防ぎます。
    """
    
    @staticmethod
    def create_search_vlm_service():
        """検索タブ用VLMServiceインスタンスを作成
        
        検索機能と回答生成機能で使用される専用のVLMServiceインスタンス。
        独立した設定を持ち、他のタブの設定変更に影響されません。
        
        Returns:
            VLMService: 検索タブ専用のVLMServiceインスタンス
        """
        service = VLMService()
        # 検索タブ用のデフォルト設定（必要に応じて調整可能）
        service.current_vlm_settings.update({
            "temperature": 0.3,
            "max_tokens": 4096,
            "oci_region": "ap-osaka-1"
        })
        return service
    
    @staticmethod
    def create_upload_vlm_service():
        """イメージ設定タブ用VLMServiceインスタンスを作成
        
        イメージのアップロード・編集・キャプション生成で使用される専用のVLMService。
        検索タブとは完全に独立した設定を維持します。
        
        Returns:
            VLMService: イメージ設定タブ専用のVLMServiceインスタンス
        """
        service = VLMService()
        # アップロードタブ用のデフォルト設定（必要に応じて調整可能）
        service.current_vlm_settings.update({
            "temperature": 0.3,
            "max_tokens": 4096,
            "oci_region": "ap-osaka-1"
        })
        return service
    
    @staticmethod
    def create_answer_vlm_service():
        """回答生成タブ用VLMServiceインスタンスを作成（将来拡張用）
        
        将来的に回答生成機能が独立したタブになった場合に使用される専用インスタンス。
        現在は検索タブと統合されているため、create_search_vlm_service()を使用してください。
        
        Returns:
            VLMService: 回答生成タブ専用のVLMServiceインスタンス
        """
        service = VLMService()
        # 回答生成専用の設定（将来の拡張を想定）
        service.current_vlm_settings.update({
            "temperature": 0.5,  # 回答生成では若干高めの創造性
            "max_tokens": 8192,  # 長文回答に対応
            "oci_region": "ap-osaka-1"
        })
        return service
    
    @staticmethod
    def get_service_type_description():
        """各サービスタイプの説明を取得
        
        開発・デバッグ時に各VLMServiceインスタンスの用途を確認するためのメソッド。
        
        Returns:
            dict: サービスタイプと説明のマッピング
        """
        return {
            "search": "検索機能と回答生成で使用。バランスの取れた設定。",
            "upload": "画像アップロード・キャプション生成で使用。精密性重視。",
            "answer": "将来の独立回答生成タブ用。高い創造性設定。"
        } 