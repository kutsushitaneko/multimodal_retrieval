"""OCI Gemini VLM 対応のユニットテスト"""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.nlp_service import NLPService
from app.vlm_service import VLMService


GEMINI_DISPLAY_NAMES = [
    "google.gemini-2.5-pro(OCI)",
    "google.gemini-2.5-flash(OCI)",
    "google.gemini-2.5-flash-lite(OCI)",
]


def test_oci_gemini_models_are_available_as_vision_oci_models():
    service = VLMService()

    vlm_models = service.get_vlm_models()
    oci_models = service.filter_vlm_models_by_provider("OCI")

    for display_name in GEMINI_DISPLAY_NAMES:
        assert display_name in vlm_models
        assert display_name in oci_models
        assert service.get_api_type(display_name) == "oci.gemini.chat"
        assert service.get_model_vision_support(display_name) is True
        assert service.get_model_default_region(display_name) == "us-chicago-1"


def test_oci_gemini_caption_request_uses_generic_chat():
    class FakeVLMService:
        OCI_REGIONS = {"US Midwest (Chicago)": "us-chicago-1"}

        def get_api_type(self, _model_display_name):
            return "oci.gemini.chat"

    client = Mock()
    client.chat.return_value = SimpleNamespace(
        data=SimpleNamespace(
            chat_response=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=[SimpleNamespace(text="Gemini response")]
                        )
                    )
                ]
            )
        )
    )

    service = NLPService(FakeVLMService())

    with (
        patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..test"}),
        patch("oci.config.from_file", return_value={}),
        patch("oci.generative_ai_inference.GenerativeAiInferenceClient", return_value=client),
    ):
        result = service._generate_caption_oci(
            model_display_name="google.gemini-2.5-flash(OCI)",
            model_name="google.gemini-2.5-flash",
            image_data_url="data:image/png;base64,AAAA",
            prompt_text="画像を説明してください",
            temperature=1.0,
            max_tokens=4096,
            oci_region="US Midwest (Chicago)",
        )

    assert result == "Gemini response"

    chat_details = client.chat.call_args.args[0]
    chat_request = chat_details.chat_request

    assert chat_details.compartment_id == "ocid1.compartment.oc1..test"
    assert chat_details.serving_mode.model_id == "google.gemini-2.5-flash"
    assert chat_request.api_format == "GENERIC"
    assert chat_request.max_tokens == 4096
    assert chat_request.temperature == 1.0
    assert chat_request.top_p == 1.0
    assert chat_request.top_k is None
    assert chat_request.num_generations == 1
    assert chat_request.messages[0].content[0].text == "画像を説明してください"
    assert chat_request.messages[0].content[1].image_url.url == "data:image/png;base64,AAAA"
