"""案C: generate_answer が UI の VLM 設定を直接参照するユニットテスト"""
import inspect


class TestGenerateAnswerUsesUiVlmSettings:
    def test_generate_answer_accepts_vlm_ui_parameters(self):
        from app.ui.events import UIEvents

        sig = inspect.signature(UIEvents.generate_answer)
        param_names = list(sig.parameters.keys())
        assert "vlm_model" in param_names
        assert "vlm_temperature" in param_names
        assert "vlm_max_tokens" in param_names
        assert "vlm_oci_region" in param_names

    def test_generate_answer_does_not_use_get_current_vlm_settings(self):
        from app.ui.events import UIEvents

        source = inspect.getsource(UIEvents.generate_answer)
        assert "get_current_vlm_settings" not in source
        assert "vlm_model" in source
        assert "vlm_oci_region" in source

    def test_answer_generation_events_wires_vlm_inputs(self):
        from app.ui.events import UIEvents

        source = inspect.getsource(UIEvents.register_answer_generation_events)
        assert "search_vlm_model" in source
        assert "search_vlm_oci_region" in source

    def test_search_and_answer_passes_vlm_inputs(self):
        from app.ui.events import UIEvents

        source = inspect.getsource(UIEvents.execute_search_and_answer)
        assert "search_vlm_model" in source
        assert "search_vlm_oci_region" in source
