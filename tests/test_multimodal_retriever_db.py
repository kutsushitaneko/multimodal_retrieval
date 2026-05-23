"""Step2: multimodal_retriever.py の DB 起動・終了処理のユニットテスト"""
import importlib
import inspect


class TestMultimodalRetrieverDbSetup:
    def test_check_db_connection_removed(self):
        module = importlib.import_module("multimodal_retriever")
        assert not hasattr(module, "check_db_connection")

    def test_main_uses_shared_pool_and_atexit_without_monitor_thread(self):
        module = importlib.import_module("multimodal_retriever")
        source = inspect.getsource(module.main)

        assert "db_pool = config.get_db_pool()" in source
        assert "atexit.register(Config.close_db_pool)" in source
        assert "check_db_connection" not in source
        assert "db_monitor_thread" not in source
        assert "threading.Thread" not in source
