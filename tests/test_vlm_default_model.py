"""VLM デフォルトモデル（MLLM_MODEL_ID）のユニットテスト"""
import os
from unittest.mock import patch

from app.vlm_service import VLMService, resolve_default_vlm_display_name, build_vlm_ui_initialization


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
