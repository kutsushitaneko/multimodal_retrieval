import os
import shutil
import time
import threading
from datetime import datetime, timedelta

class CleanupService:
    """一時ディレクトリのクリーンアップを管理するサービスクラス
    
    マルチセッション環境で蓄積される一時ディレクトリを定期的にクリーンアップし、
    ディスク容量の枯渇を防ぎます。
    """
    
    def __init__(self, base_temp_dir, max_age_hours=24):
        """クリーンアップサービスを初期化
        
        Args:
            base_temp_dir (str): ベースの一時ディレクトリパス
            max_age_hours (int): ディレクトリの最大保持時間（時間）
        """
        self.base_temp_dir = base_temp_dir
        self.max_age_hours = max_age_hours
        self.cleanup_thread = None
        self.stop_cleanup = False
        
    def start_cleanup_thread(self, interval_minutes=60):
        """クリーンアップスレッドを開始
        
        Args:
            interval_minutes (int): クリーンアップ実行間隔（分）
        """
        if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
            return  # 既に実行中
            
        self.stop_cleanup = False
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.cleanup_thread.start()
        print(f"一時ディレクトリクリーンアップスレッドを開始しました。間隔: {interval_minutes}分")
        
    def stop_cleanup_thread(self):
        """クリーンアップスレッドを停止"""
        self.stop_cleanup = True
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            print("一時ディレクトリクリーンアップスレッドを停止しました。")
            
    def _cleanup_loop(self, interval_minutes):
        """クリーンアップループ（バックグラウンドスレッド用）"""
        while not self.stop_cleanup:
            try:
                self.cleanup_old_directories()
            except Exception as e:
                print(f"一時ディレクトリクリーンアップ中にエラーが発生しました: {e}")
            
            # 指定された間隔で待機
            for _ in range(interval_minutes * 60):  # 1秒ずつチェック
                if self.stop_cleanup:
                    break
                time.sleep(1)
                
    def cleanup_old_directories(self):
        """古い一時ディレクトリをクリーンアップ"""
        if not os.path.exists(self.base_temp_dir):
            return
            
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        removed_session_count = 0
        removed_gradio_count = 0
        
        try:
            for item in os.listdir(self.base_temp_dir):
                item_path = os.path.join(self.base_temp_dir, item)
                if not os.path.isdir(item_path):
                    continue
                
                # セッションディレクトリのクリーンアップ
                if item.startswith("session_"):
                    # ディレクトリの作成時間をチェック
                    try:
                        creation_time = datetime.fromtimestamp(os.path.getctime(item_path))
                        if creation_time < cutoff_time:
                            shutil.rmtree(item_path)
                            removed_session_count += 1
                            print(f"古いセッションディレクトリを削除しました: {item}")
                    except OSError as e:
                        print(f"セッションディレクトリ削除エラー ({item}): {e}")
                
                # Gradioの一時ファイルのクリーンアップ
                elif item == "gradio":
                    try:
                        gradio_dir = item_path
                        if os.path.exists(gradio_dir):
                            for gradio_item in os.listdir(gradio_dir):
                                gradio_item_path = os.path.join(gradio_dir, gradio_item)
                                try:
                                    if os.path.isfile(gradio_item_path):
                                        file_time = datetime.fromtimestamp(os.path.getmtime(gradio_item_path))
                                    elif os.path.isdir(gradio_item_path):
                                        file_time = datetime.fromtimestamp(os.path.getctime(gradio_item_path))
                                    else:
                                        continue
                                        
                                    if file_time < cutoff_time:
                                        if os.path.isfile(gradio_item_path):
                                            os.remove(gradio_item_path)
                                        else:
                                            shutil.rmtree(gradio_item_path)
                                        removed_gradio_count += 1
                                        
                                except OSError as e:
                                    print(f"Gradio一時ファイル削除エラー ({gradio_item}): {e}")
                    except OSError as e:
                        print(f"Gradioディレクトリアクセスエラー: {e}")
                    
        except OSError as e:
            print(f"一時ディレクトリ一覧取得エラー: {e}")
            
        if removed_session_count > 0 or removed_gradio_count > 0:
            print(f"クリーンアップ完了: セッション {removed_session_count}個、Gradio一時ファイル {removed_gradio_count}個を削除しました。")
            
    def force_cleanup_all_sessions(self):
        """全てのセッションディレクトリを強制削除（デバッグ用）"""
        if not os.path.exists(self.base_temp_dir):
            return
            
        removed_count = 0
        try:
            for item in os.listdir(self.base_temp_dir):
                if not item.startswith("session_"):
                    continue
                    
                item_path = os.path.join(self.base_temp_dir, item)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        removed_count += 1
                    except OSError as e:
                        print(f"ディレクトリ削除エラー ({item}): {e}")
                        
        except OSError as e:
            print(f"一時ディレクトリ一覧取得エラー: {e}")
            
        print(f"強制クリーンアップ完了: {removed_count}個のセッションディレクトリを削除しました。") 