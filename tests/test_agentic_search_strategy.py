from app.agentic_search_strategy import (
    classify_question_strategy,
    format_lead_tool_recommendation,
    query_has_fulltext_friendly_tokens,
)


def test_classify_identifier_for_error_code():
    hint = classify_question_strategy("ORA-00923 とは何ですか？")

    assert hint.strategy == "identifier"
    assert "caption_fulltext_search" in hint.hint_text


def test_classify_identifier_for_url():
    hint = classify_question_strategy("https://example.com/docs の内容は？")

    assert hint.strategy == "identifier"


def test_classify_transitive_for_nested_no_and_attribute():
    hint = classify_question_strategy("対象Aの属性Bの定義は？")

    assert hint.strategy == "transitive"
    assert "第1ホップ" in hint.hint_text
    assert "query_variants" in hint.hint_text


def test_classify_transitive_for_location_and_coordinates():
    hint = classify_question_strategy("看板の設置場所の緯度経度は？")

    assert hint.strategy == "transitive"


def test_classify_none_for_simple_question():
    hint = classify_question_strategy("猫の特徴について教えてください")

    assert hint.strategy == "none"
    assert hint.hint_text == ""


def test_identifier_takes_priority_over_transitive():
    hint = classify_question_strategy("ORA-00923 の原因と意味の定義は？")

    assert hint.strategy == "identifier"


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


def test_identifier_hint_mentions_second_hop_vector_for_natural_language():
    hint = classify_question_strategy("2312.10997 の要約は？")
    assert "2ホップ目以降" in hint.hint_text
    assert "caption_vector_search" in hint.hint_text
