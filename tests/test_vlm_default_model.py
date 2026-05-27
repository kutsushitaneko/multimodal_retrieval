"""VLM デフォルトモデル（MLLM_MODEL_ID）のユニットテスト"""
import os
from unittest.mock import patch

from app.vlm_service import VLMService, resolve_default_vlm_display_name, build_vlm_ui_initialization
from app.nlp_service import NLPService


VLM_MODELS = {
    "meta.llama-4-scout-17b-16e-instruct(OCI)": {
        "model_name": "meta.llama-4-scout-17b-16e-instruct",
        "api_type": "oci.llama.chat",
        "vision": True,
    },
    "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)": {
        "model_name": "meta.llama-4-maverick-17b-128e-instruct-fp8",
        "api_type": "oci.llama.chat",
        "vision": True,
    },
    "google.gemini-2.5-flash(OCI)": {
        "model_name": "google.gemini-2.5-flash",
        "api_type": "oci.gemini.chat",
        "vision": True,
    },
}


class TestResolveDefaultVlmDisplayName:
    def test_returns_first_when_env_not_set(self):
        result = resolve_default_vlm_display_name(VLM_MODELS, None)
        assert result == "meta.llama-4-scout-17b-16e-instruct(OCI)"

    def test_matches_by_model_name(self):
        result = resolve_default_vlm_display_name(
            VLM_MODELS, "meta.llama-4-maverick-17b-128e-instruct-fp8"
        )
        assert result == "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)"

    def test_matches_by_display_name(self):
        result = resolve_default_vlm_display_name(
            VLM_MODELS, "meta.llama-4-scout-17b-16e-instruct(OCI)"
        )
        assert result == "meta.llama-4-scout-17b-16e-instruct(OCI)"

    def test_matches_oci_gemini_by_model_name(self):
        result = resolve_default_vlm_display_name(
            VLM_MODELS, "google.gemini-2.5-flash"
        )
        assert result == "google.gemini-2.5-flash(OCI)"


class TestBuildVlmUiInitialization:
    @patch.dict(os.environ, {"MLLM_MODEL_ID": "meta.llama-4-maverick-17b-128e-instruct-fp8"})
    @patch.object(VLMService, "get_vlm_models", return_value=VLM_MODELS)
    @patch.object(
        VLMService,
        "get_available_service_providers",
        return_value=["すべて", "OCI"],
    )
    def test_uses_mllm_model_id_from_env(self, _mock_providers, _mock_vlm_models):
        _choices, default_vlm, _providers, _models, _service = build_vlm_ui_initialization()
        assert default_vlm == "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)"


class TestVlmServiceFactoryDefaults:
    @patch.dict(os.environ, {"MLLM_MODEL_ID": "meta.llama-4-maverick-17b-128e-instruct-fp8"})
    @patch.object(VLMService, "get_vlm_models", return_value=VLM_MODELS)
    def test_search_and_upload_services_get_independent_defaults(self, _mock_vlm_models):
        from app.vlm_service_factory import VLMServiceFactory

        search_service = VLMServiceFactory.create_search_vlm_service()
        upload_service = VLMServiceFactory.create_upload_vlm_service()

        expected = "meta.llama-4-maverick-17b-128e-instruct-fp8(OCI)"
        assert search_service.current_vlm_settings["model"] == expected
        assert upload_service.current_vlm_settings["model"] == expected

        search_service.update_current_vlm_settings(model="meta.llama-4-scout-17b-16e-instruct(OCI)")
        assert search_service.current_vlm_settings["model"] == "meta.llama-4-scout-17b-16e-instruct(OCI)"
        assert upload_service.current_vlm_settings["model"] == expected


class TestVlmModelUiSettings:
    def test_model_ui_settings_hide_temperature_and_resolve_region(self):
        service = VLMService()
        service.model_settings = {
            "gpt-5.5(OpenAI)": {
                "model_name": "gpt-5.5",
                "api_type": "openai.reasoning",
                "max_tokens": 128000,
                "default_tokens": 4096,
                "default_temperature": 1.0,
                "supports_temperature": False,
                "vision": True,
            },
            "google.gemini-2.5-pro(OCI)": {
                "model_name": "google.gemini-2.5-pro",
                "api_type": "oci.gemini.chat",
                "max_tokens": 65536,
                "default_tokens": 8192,
                "default_temperature": 0.2,
                "default_region": "us-chicago-1",
                "vision": True,
            },
        }

        reasoning_settings = service.get_model_ui_settings("gpt-5.5(OpenAI)")
        assert reasoning_settings["temperature_visible"] is False
        assert reasoning_settings["temperature_interactive"] is False
        assert reasoning_settings["default_tokens"] == 4096
        assert reasoning_settings["is_oci_model"] is False

        gemini_settings = service.get_model_ui_settings("google.gemini-2.5-pro(OCI)")
        assert gemini_settings["temperature_visible"] is True
        assert gemini_settings["max_tokens_limit"] == 65536
        assert gemini_settings["default_tokens"] == 8192
        assert gemini_settings["region_name"] == "US Midwest (Chicago)"
        assert gemini_settings["is_oci_model"] is True

    def test_model_changed_returns_temperature_visibility_from_model_capability(self):
        service = VLMService()
        service.model_settings = {
            "gpt-5.5(OpenAI)": {
                "model_name": "gpt-5.5",
                "api_type": "openai.reasoning",
                "max_tokens": 128000,
                "default_tokens": 4096,
                "default_temperature": 1.0,
                "supports_temperature": False,
                "vision": True,
            },
        }

        temperature_slider, max_tokens_slider, oci_region_dropdown = service.model_changed("gpt-5.5(OpenAI)")

        assert temperature_slider.visible is False
        assert temperature_slider.interactive is False
        assert max_tokens_slider.maximum == 128000
        assert max_tokens_slider.value == 4096
        assert oci_region_dropdown.visible is False


class TestNlpServiceTemperatureSupport:
    def test_generate_text_omits_temperature_for_unsupported_model(self):
        class FakeVLMService:
            def supports_temperature(self, _model_display_name):
                return False

            def get_api_type(self, _model_display_name):
                return "anthropic.message"

            def get_model_name(self, _model_display_name):
                return "claude-opus-4-7"

        service = NLPService(FakeVLMService())

        with patch.object(service, "_generate_text_anthropic", return_value="ok") as mock_generate:
            assert service.generate_text("claude-opus-4-7(Anthropic)", "prompt", temperature=0.7) == "ok"

        assert mock_generate.call_args.args[2] is None
