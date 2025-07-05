import spacy
import threading


class GlobalNLPService:
    """spaCy ja_ginzaモデルのグローバルシングルトンサービスクラス
    
    アプリケーション全体で単一のspaCyモデルインスタンスを共有し、
    メモリ使用量を削減し、初期化時間を短縮します。
    スレッドセーフな実装により、マルチセッション環境でも安全に使用できます。
    """
    
    _instance = None
    _lock = threading.Lock()
    _nlp = None
    _nlp_lock = threading.Lock()
    
    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GlobalNLPService, cls).__new__(cls)
        return cls._instance
    
    def get_nlp(self):
        """spaCyのja_ginzaモデルを取得
        
        初回呼び出し時にモデルをロードし、以降は同じインスタンスを返します。
        スレッドセーフな実装により、マルチセッション環境でも安全に使用できます。
        
        Returns:
            spacy.Language: ja_ginzaモデルのインスタンス
        """
        if self._nlp is None:
            with self._nlp_lock:
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


# モジュールレベルでのグローバルインスタンス
_global_nlp_service = None
_global_lock = threading.Lock()


def get_global_nlp_service():
    """グローバルNLPServiceインスタンスを取得
    
    Returns:
        GlobalNLPService: グローバルNLPServiceインスタンス
    """
    global _global_nlp_service
    if _global_nlp_service is None:
        with _global_lock:
            if _global_nlp_service is None:
                _global_nlp_service = GlobalNLPService()
    return _global_nlp_service 