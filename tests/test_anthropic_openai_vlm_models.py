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


def _invoke_anthropic_caption_via_public_api(display_name, temperature=0.3, max_tokens=4096):
    """公開API generate_caption_with_vlm 経由で Anthropic キャプション生成を実行し、
    Anthropic クライアントへ渡された create() の kwargs を返すヘルパー。

    画像エンコードはファイルシステムに依存しないようスタブ化し、temperature の
    扱い（モデル設定 -> APIパラメータ）という不変条件のみを観測対象にする。
    """
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="Anthropic response")]
    )
    anthropic_module = SimpleNamespace(Anthropic=Mock(return_value=client))
    service = NLPService()

    with patch.dict(sys.modules, {"anthropic": anthropic_module}), patch.object(
        service, "_image_to_base64_data_url", return_value="data:image/png;base64,AAAA"
    ):
        result = service.generate_caption_with_vlm(
            image_path="dummy.png",
            vlm_model=display_name,
            prompt_text="画像を説明してください",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    assert result == "Anthropic response"
    return client.messages.create.call_args.kwargs


def test_anthropic_temperature_follows_supports_temperature_setting():
    """supports_temperature 設定に応じて Anthropic へ temperature が渡る/渡らないことの対照テスト。

    内部の private メソッドのシグネチャではなく、公開API（generate_caption_with_vlm）を通した
    「モデル設定 -> Anthropic API パラメータ」という不変条件を検証する。
    """
    # supports_temperature:false のモデルでは temperature を渡さない
    params_unsupported = _invoke_anthropic_caption_via_public_api("claude-opus-4-7(Anthropic)")
    assert params_unsupported["model"] == "claude-opus-4-7"
    assert params_unsupported["max_tokens"] == 4096
    assert "temperature" not in params_unsupported

    # supports_temperature 未指定（=デフォルト true）のモデルでは temperature を渡す
    params_supported = _invoke_anthropic_caption_via_public_api("claude-opus-4-6(Anthropic)")
    assert params_supported["model"] == "claude-opus-4-6"
    assert params_supported["max_tokens"] == 4096
    assert params_supported["temperature"] == 0.3


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
