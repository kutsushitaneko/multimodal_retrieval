from app.agentic_search_strategy import (
    classify_question_strategy,
    count_regex_detected_entities,
    format_lead_tool_recommendation,
    fulltext_query_mixes_identifier_and_natural_language,
    query_has_fulltext_friendly_tokens,
)


def test_classify_identifier_for_error_code():
    hint = classify_question_strategy("ORA-00923 とは何ですか？")

    assert hint.strategy == "identifier"
    assert "plan_and_execute_search" in hint.hint_text


def test_classify_identifier_for_url():
    hint = classify_question_strategy("https://example.com/docs の内容は？")

    assert hint.strategy == "identifier"


def test_classify_none_for_nested_attribute_question():
    hint = classify_question_strategy("対象Aの属性Bの定義は？")

    assert hint.strategy == "none"
    assert hint.hint_text == ""


def test_classify_none_for_simple_question():
    hint = classify_question_strategy("猫の特徴について教えてください")

    assert hint.strategy == "none"
    assert hint.hint_text == ""


def test_identifier_takes_priority_for_mixed_question():
    hint = classify_question_strategy("ORA-00923 の原因と意味の定義は？")

    assert hint.strategy == "identifier"


def test_count_regex_detected_entities():
    assert count_regex_detected_entities("ORA-00923") >= 1
    assert count_regex_detected_entities("猫の特徴") == 0


def test_query_has_fulltext_friendly_tokens_for_paper_id_and_error_code():
    assert query_has_fulltext_friendly_tokens("2312.10997")
    assert query_has_fulltext_friendly_tokens("ORA-00923")


def test_query_has_fulltext_friendly_tokens_false_for_natural_language_title():
    long_title = "Retrieval-Augmented Generation for Large Language Models: A Survey"
    assert not query_has_fulltext_friendly_tokens(long_title)
    assert not query_has_fulltext_friendly_tokens(f"{long_title} abstract")


def test_format_lead_tool_recommendation_prefers_vector_for_natural_language():
    lead = "Retrieval-Augmented Generation for Large Language Models: A Survey"
    assert "caption_vector_search" in format_lead_tool_recommendation(lead)


def test_fulltext_query_mixes_identifier_and_natural_language():
    assert fulltext_query_mixes_identifier_and_natural_language("OCI Enterprise AI Agents マネージド デプロイ")
    assert not fulltext_query_mixes_identifier_and_natural_language("2312.10997")
    assert not fulltext_query_mixes_identifier_and_natural_language("Retrieval-Augmented Generation Survey")
    assert not fulltext_query_mixes_identifier_and_natural_language("ORA-00923")
