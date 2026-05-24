from unittest.mock import MagicMock, patch

from PIL import Image

from app.ui.events import (
    ANSWER_MODE_LISTWISE,
    ANSWER_MODE_SINGLE_IMAGE,
    REFERENCE_TYPE_ALL,
    REFERENCE_TYPE_CAPTION_ONLY,
    REFERENCE_TYPE_IMAGE_ONLY,
    UIEvents,
)


def make_events():
    with patch("app.global_nlp_service.get_global_nlp_service", return_value=MagicMock()), \
         patch("app.vlm_service_factory.VLMServiceFactory.create_upload_vlm_service", return_value=MagicMock()), \
         patch("app.vlm_service_factory.VLMServiceFactory.create_search_vlm_service", return_value=MagicMock()):
        search_service = MagicMock()
        search_service.normalize_newlines.side_effect = lambda text: text or ""
        return UIEvents(search_service)


def make_result(image_id, file_name, caption, search_mode="ベクトル検索"):
    return {
        "image_id": image_id,
        "file_name": file_name,
        "caption": caption,
        "search_mode": search_mode,
        "distance": 0.1,
        "image": Image.new("RGB", (8, 8), color="white"),
    }


def choice_values(radio):
    return [choice[1] if isinstance(choice, tuple) else choice for choice in radio.choices]


class TestAnswerGenerationMode:
    def test_image_query_forces_single_image_mode(self):
        events = make_events()

        update = events.update_answer_generation_mode_choices(
            "画像ベクトル",
            "画像",
            ANSWER_MODE_LISTWISE,
        )

        assert update.value == ANSWER_MODE_SINGLE_IMAGE
        assert choice_values(update) == [ANSWER_MODE_SINGLE_IMAGE]

    def test_text_query_keeps_listwise_mode_available(self):
        events = make_events()

        update = events.update_answer_generation_mode_choices(
            "画像ベクトル",
            "テキスト",
            ANSWER_MODE_LISTWISE,
        )

        assert update.value == ANSWER_MODE_LISTWISE
        assert ANSWER_MODE_SINGLE_IMAGE in choice_values(update)
        assert ANSWER_MODE_LISTWISE in choice_values(update)

    def test_text_query_defaults_to_listwise_mode(self):
        events = make_events()

        update = events.update_answer_generation_mode_choices(
            "画像ベクトル",
            "テキスト",
            None,
        )

        assert update.value == ANSWER_MODE_LISTWISE
        assert ANSWER_MODE_SINGLE_IMAGE in choice_values(update)
        assert ANSWER_MODE_LISTWISE in choice_values(update)


class TestListwiseCandidates:
    def test_build_candidates_prefers_all_combined_results(self):
        events = make_events()
        combined = [make_result(1, "combined.png", "combined")]
        vector = [make_result(2, "vector.png", "vector")]

        candidates = events._build_listwise_candidates({
            "all_combined_results": combined,
            "all_vector_results": vector,
            "all_keyword_results": [],
        })

        assert [candidate["file_name"] for candidate in candidates] == ["combined.png"]

    def test_build_candidates_deduplicates_vector_and_keyword_results(self):
        events = make_events()
        duplicate_vector = make_result(1, "same.png", "vector")
        duplicate_keyword = make_result(1, "same.png", "keyword", "全文検索")
        unique_keyword = make_result(2, "unique.png", "keyword", "全文検索")

        candidates = events._build_listwise_candidates({
            "all_vector_results": [duplicate_vector],
            "all_keyword_results": [duplicate_keyword, unique_keyword],
        })

        assert [candidate["id"] for candidate in candidates] == ["1", "2"]


class TestListwiseSelectionParsing:
    def test_parse_normal_json(self):
        events = make_events()
        candidates = [
            {"id": "1", "file_name": "a.png"},
            {"id": "2", "file_name": "b.png"},
        ]

        selected, reason, error = events._parse_listwise_selection(
            '{"selected_image_ids": ["2", "1"], "reason": "順序"}',
            candidates,
        )

        assert error is None
        assert reason == "順序"
        assert [candidate["id"] for candidate in selected] == ["2", "1"]

    def test_parse_fenced_json_and_drop_duplicates(self):
        events = make_events()
        candidates = [
            {"id": "1", "file_name": "a.png"},
            {"id": "2", "file_name": "b.png"},
        ]

        selected, reason, error = events._parse_listwise_selection(
            '```json\n{"selected_image_ids": ["1", "1", "2"]}\n```',
            candidates,
        )

        assert error is None
        assert reason == ""
        assert [candidate["id"] for candidate in selected] == ["1", "2"]

    def test_parse_invalid_json_returns_error(self):
        events = make_events()

        selected, reason, error = events._parse_listwise_selection("not json", [{"id": "1"}])

        assert selected == []
        assert reason == ""
        assert "JSON解析に失敗" in error

    def test_unknown_ids_return_error(self):
        events = make_events()

        selected, reason, error = events._parse_listwise_selection(
            '{"selected_image_ids": ["missing"]}',
            [{"id": "1", "file_name": "a.png"}],
        )

        assert selected == []
        assert reason == ""
        assert "存在しません" in error


class TestListwiseAnswerGeneration:
    def test_listwise_answer_uses_selected_order_and_multiple_images(self):
        events = make_events()
        events.get_current_answer_prompt = MagicMock(return_value="質問:{query_text}\n資料:{documents}")
        events._call_text_vlm = MagicMock(return_value='{"selected_image_ids": ["2", "1"]}')

        nlp_instance = MagicMock()
        nlp_instance.generate_answer_with_vlm_images.return_value = "回答本文"

        with patch("app.nlp_service.NLPService", return_value=nlp_instance):
            answer, referenced_gallery, reason_text = events.generate_answer(
                "質問",
                "デフォルト",
                {
                    "all_combined_results": [
                        make_result(1, "a.png", "A caption"),
                        make_result(2, "b.png", "B caption"),
                    ]
                },
                REFERENCE_TYPE_ALL,
                "model",
                0.0,
                1024,
                "Japan Central (Osaka)",
                ANSWER_MODE_LISTWISE,
            )

        assert "「b.png」「a.png」" in answer
        assert "回答本文" in answer
        assert referenced_gallery.visible is True
        assert len(referenced_gallery.value) == 2
        assert reason_text.visible is True
        assert nlp_instance.generate_answer_with_vlm_images.call_count == 1

    def test_caption_only_does_not_send_images_to_final_answer(self):
        events = make_events()
        events.get_current_answer_prompt = MagicMock(return_value="質問:{query_text}\n資料:{documents}")
        events._call_text_vlm = MagicMock(side_effect=[
            '{"selected_image_ids": ["1"]}',
            "caption only answer",
        ])

        answer, referenced_gallery, reason_text = events.generate_answer(
            "質問",
            "デフォルト",
            {"all_combined_results": [make_result(1, "a.png", "A caption")]},
            REFERENCE_TYPE_CAPTION_ONLY,
            "model",
            0.0,
            1024,
            "Japan Central (Osaka)",
            ANSWER_MODE_LISTWISE,
        )

        assert "caption only answer" in answer
        assert referenced_gallery.visible is True
        assert referenced_gallery.value == []
        assert reason_text.visible is True
        assert events._call_text_vlm.call_count == 2

    def test_image_only_replaces_documents_with_image_notice(self):
        events = make_events()
        events.get_current_answer_prompt = MagicMock(return_value="{documents}")
        events._call_text_vlm = MagicMock(return_value='{"selected_image_ids": ["1"]}')

        nlp_instance = MagicMock()
        nlp_instance.generate_answer_with_vlm_images.return_value = "image answer"

        with patch("app.nlp_service.NLPService", return_value=nlp_instance):
            answer, referenced_gallery, reason_text = events.generate_answer(
                "質問",
                "デフォルト",
                {"all_combined_results": [make_result(1, "a.png", "A caption")]},
                REFERENCE_TYPE_IMAGE_ONLY,
                "model",
                0.0,
                1024,
                "Japan Central (Osaka)",
                ANSWER_MODE_LISTWISE,
            )

        prompt_text = nlp_instance.generate_answer_with_vlm_images.call_args.kwargs["prompt_text"]
        assert "A caption" not in prompt_text
        assert "image answer" in answer
        assert referenced_gallery.visible is True
        assert len(referenced_gallery.value) == 1
        assert reason_text.visible is True

    def test_single_image_mode_hides_reference_display(self):
        events = make_events()
        events._generate_answer_from_single_selected_image = MagicMock(return_value="single answer")

        answer, referenced_gallery, reason_text = events.generate_answer(
            "質問",
            "デフォルト",
            {},
            REFERENCE_TYPE_ALL,
            "model",
            0.0,
            1024,
            "Japan Central (Osaka)",
            ANSWER_MODE_SINGLE_IMAGE,
        )

        assert answer == "single answer"
        assert referenced_gallery.visible is False
        assert reason_text.visible is False
