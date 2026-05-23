"""Anthropic/OpenAI VLM モデル定義のユニットテスト"""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.nlp_service import NLPService
from app.vlm_service import VLMService


ANTHROPIC_MODELS = {
    "claude-opus-4-7(Anthropic)": ("claude-opus-4-7", 128000),
    "claude-opus-4-6(Anthropic)": ("claude-opus-4-6", 64000),
    "claude-opus-4-5-20251101(Anthropic)": ("claude-opus-4-5-20251101", 64000),
    "claude-sonnet-4-6(Anthropic)": ("claude-sonnet-4-6", 64000),
    "claude-sonnet-4-5-20250929(Anthropic)": ("claude-sonnet-4-5-20250929", 64000),
    "claude-haiku-4-5-20251001(Anthropic)": ("claude-haiku-4-5-20251001", 64000),
}

OPENAI_MODELS = {
    "gpt-5.5(OpenAI)": "gpt-5.5",
    "gpt-5.5-pro(OpenAI)": "gpt-5.5-pro",
    "gpt-5.4(OpenAI)": "gpt-5.4",
    "gpt-5.4-mini(OpenAI)": "gpt-5.4-mini",
    "gpt-5.4-nano(OpenAI)": "gpt-5.4-nano",
}

REMOVED_DIRECT_API_MODELS = {
    "claude-opus-4-0(Anthropic)",
    "claude-sonnet-4-0(Anthropic)",
    "claude-3-7-sonnet-latest(Anthropic)",
    "claude-3-5-sonnet-latest(Anthropic)",
    "claude-3-5-haiku-latest(Anthropic)",
    "claude-3-opus-latest(Anthropic)",
    "gpt-5-nano(OpenAI)",
    "gpt-5-mini(OpenAI)",
    "gpt-5(OpnAI)",
    "o4-mini-2025-04-16(OpenAI)",
    "o3-pro-2025-06-10(OpenAI)",
    "o3-2025-04-16(OpenAI)",
    "o3-mini-2025-01-31(OpenAI)",
    "gpt-4.1-2025-04-14(OpenAI)",
    "gpt-4.1-mini-2025-04-14(OpenAI)",
    "gpt-4.1-nano(OpenAI)",
    "gpt-4o(OpenAI)",
    "gpt-4o mini(OpenAI)",
    "chatgpt-4o-latest(OpenAI)",
}


def test_anthropic_models_are_current_vision_models():
    service = VLMService()
    vlm_models = service.get_vlm_models()
    anthropic_models = service.filter_vlm_models_by_provider("Anthropic")

    for display_name, (model_id, max_tokens) in ANTHROPIC_MODELS.items():
        assert display_name in vlm_models
        assert display_name in anthropic_models
        assert service.get_model_name(display_name) == model_id
        assert service.get_api_type(display_name) == "anthropic.message"
        assert service.get_model_max_tokens(display_name) == max_tokens
        assert service.get_model_vision_support(display_name) is True


def test_openai_models_are_current_reasoning_vision_models():
    service = VLMService()
    vlm_models = service.get_vlm_models()
    openai_models = service.filter_vlm_models_by_provider("OpenAI")

    for display_name, model_id in OPENAI_MODELS.items():
        assert display_name in vlm_models
        assert display_name in openai_models
        assert service.get_model_name(display_name) == model_id
        assert service.get_api_type(display_name) == "openai.reasoning"
        assert service.get_model_max_tokens(display_name) == 128000
        assert service.get_model_default_temperature(display_name) == 1.0
        assert service.get_model_vision_support(display_name) is True


def test_removed_direct_api_models_are_not_registered():
    service = VLMService()

    for display_name in REMOVED_DIRECT_API_MODELS:
        assert display_name not in service.model_settings


def test_anthropic_opus_47_omits_temperature_parameter():
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="Anthropic response")]
    )
    anthropic_module = SimpleNamespace(Anthropic=Mock(return_value=client))
    service = NLPService()

    with patch.dict(sys.modules, {"anthropic": anthropic_module}):
        result = service._generate_caption_anthropic(
            model_name="claude-opus-4-7",
            image_data_url="data:image/png;base64,AAAA",
            prompt_text="画像を説明してください",
            temperature=0.3,
            max_tokens=4096,
        )

    assert result == "Anthropic response"
    params = client.messages.create.call_args.kwargs
    assert params["model"] == "claude-opus-4-7"
    assert params["max_tokens"] == 4096
    assert "temperature" not in params


def test_openai_reasoning_models_use_responses_api():
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text="OpenAI response")
    openai_module = SimpleNamespace(OpenAI=Mock(return_value=client))
    service = NLPService()

    with patch.dict(sys.modules, {"openai": openai_module}):
        result = service._generate_caption_openai(
            model_name="gpt-5.5-pro",
            api_type="openai.reasoning",
            image_data_url="data:image/png;base64,AAAA",
            prompt_text="画像を説明してください",
            temperature=1.0,
            max_tokens=4096,
        )

    assert result == "OpenAI response"
    params = client.responses.create.call_args.kwargs
    assert params["model"] == "gpt-5.5-pro"
    assert params["reasoning"] == {"effort": "medium"}
    assert params["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "画像を説明してください",
    }
    assert params["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,AAAA",
    }
