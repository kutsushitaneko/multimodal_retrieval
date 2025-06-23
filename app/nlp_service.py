import spacy
import threading

class NLPService:
    """spaCyモデルをシングルトンパターンで管理するサービスクラス
    
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