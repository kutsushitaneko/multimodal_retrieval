"""案C: generate_answer が UI の VLM 設定を直接参照するユニットテスト"""
import inspect
from unittest.mock import MagicMock, patch


def make_events():
    with patch("app.global_nlp_service.get_global_nlp_service", return_value=MagicMock()), \
         patch("app.vlm_service_factory.VLMServiceFactory.create_upload_vlm_service", return_value=MagicMock()), \
         patch("app.vlm_service_factory.VLMServiceFactory.create_search_vlm_service", return_value=MagicMock()):
        from app.ui.events import UIEvents

        search_service = MagicMock()
        search_service.normalize_newlines.side_effect = lambda text: text or ""
        return UIEvents(search_service)


class TestGenerateAnswerUsesUiVlmSettings:
    def test_generate_answer_accepts_vlm_ui_parameters(self):
        from app.ui.events import UIEvents

        sig = inspect.signature(UIEvents.generate_answer)
        param_names = list(sig.parameters.keys())
        assert "vlm_model" in param_names
        assert "vlm_temperature" in param_names
        assert "vlm_max_tokens" in param_names
        assert "vlm_oci_region" in param_names
        assert "answer_generation_mode" in param_names

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
        assert "answer_generation_mode_radio" in source
        assert "referenced_images_gallery" in source
        assert "listwise_reason_text" in source

    def test_search_and_answer_passes_vlm_inputs(self):
        from app.ui.events import UIEvents

        source = inspect.getsource(UIEvents.execute_search_and_answer)
        assert "search_vlm_model" in source
        assert "search_vlm_oci_region" in source
        assert "answer_generation_mode" in source
        assert "referenced_images_update" in source
        assert "listwise_reason_update" in source

    def test_search_section_defines_answer_generation_mode_radio(self):
        from app.ui.components import UIComponents

        source = inspect.getsource(UIComponents.create_search_section)
        assert "検索" in source
        assert "検索結果の選別・並べ替え設定" in source
        assert "answer_generation_mode_radio" in source
        assert "ANSWER_MODE_SINGLE_IMAGE" in source
        assert "ANSWER_MODE_LISTWISE" in source


class TestSearchAndAnswerGalleryDisplay:
    def test_search_and_answer_event_chain_updates_gallery_labels(self):
        from app.ui.events import UIEvents

        source = inspect.getsource(UIEvents.register_search_and_answer_button_events)

        assert "update_gallery_labels" in source
        assert "outputs=[vector_gallery, keyword_gallery]" in source

    def test_execute_search_and_answer_does_not_update_gallery_labels_directly(self):
        from app.ui.events import UIEvents

        source = inspect.getsource(UIEvents.execute_search_and_answer)

        assert "_apply_search_result_gallery_labels" not in source
        assert "update_gallery_labels" not in source

    def test_caption_search_gallery_labels_match_search_button(self):
        events = make_events()

        vector_gallery, keyword_gallery = events.update_gallery_labels(
            "質問",
            "テキスト",
            "キャプション（テキストベクトルと全文）",
        )

        assert vector_gallery.label == "ベクトル検索"
        assert vector_gallery.visible is True
        assert keyword_gallery.label == "全文検索"
        assert keyword_gallery.visible is True

    def test_text_image_search_gallery_label_matches_search_button(self):
        events = make_events()

        vector_gallery, keyword_gallery = events.update_gallery_labels(
            "質問",
            "テキスト",
            "画像ベクトル",
        )

        assert vector_gallery.label == "テキストベクトルによる画像検索"
        assert vector_gallery.label != "全件表示"
        assert vector_gallery.visible is True
        assert keyword_gallery.visible is False
