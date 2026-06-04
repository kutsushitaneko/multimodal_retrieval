from unittest.mock import MagicMock

from app.fulltext_entity_extractor import FulltextEntityExtractor
from app.search_query_generator import SearchQueryGenerator
from app.search_service import SearchService


def test_rule_entities_generate_or_exact_query_for_error_ip_url_and_paper_id():
    generator = SearchQueryGenerator()

    query = generator.generate("ORA-00923 と 192.168.0.1 と https://example.com と 2312.10997 を説明して")

    assert "{ORA-00923}" in query
    assert "{192.168.0.1}" in query
    assert "{https://example.com}" in query
    assert "{2312.10997}" in query
    assert " OR " in query
    assert " AND " not in query


def test_general_question_falls_back_to_legacy_and_query():
    generator = SearchQueryGenerator()

    query = generator.generate("赤い猫と寺院")

    assert " OR " not in query
    assert " AND " in query


def test_or_exact_query_escapes_braces_and_deduplicates_entities():
    generator = SearchQueryGenerator()

    query = generator.build_or_exact_query([
        {"text": "OCI{Agent}", "type": "product_name"},
        {"text": "OCI{Agent}", "type": "product_name"},
        {"text": "意味", "type": "product_name"},
        {"text": "", "type": "product_name"},
    ])

    assert query == "{OCI\\{Agent\\}}"


def test_llm_entity_extractor_parses_fenced_json_and_filters_invalid_entities():
    llm = MagicMock(return_value="""```json
{"entities": [
  {"text": "OCI Generative AI Agents", "type": "product_name"},
  {"text": "意味", "type": "product_name"},
  {"text": "Unknown", "type": "unknown_type"},
  {"text": "OCI Generative AI Agents", "type": "product_name"}
]}
```""")
    extractor = FulltextEntityExtractor(llm)

    entities = extractor.extract_entities("OCI Generative AI Agents の意味は？")

    assert entities == [{"text": "OCI Generative AI Agents", "type": "product_name"}]


def test_llm_entity_extractor_returns_empty_for_broken_json_or_unavailable_llm():
    broken = FulltextEntityExtractor(MagicMock(return_value="not json"))
    unavailable = FulltextEntityExtractor()

    assert broken.extract_entities("ORA-00923") == []
    assert unavailable.extract_entities("ORA-00923") == []


def test_llm_entities_are_used_as_or_fulltext_query():
    extractor = FulltextEntityExtractor(
        MagicMock(return_value='{"entities": [{"text": "OCI Generative AI Agents", "type": "product_name"}]}')
    )
    generator = SearchQueryGenerator(fulltext_entity_extractor=extractor)

    query = generator.generate("OCI Generative AI Agents の特徴は？")

    assert query == "{OCI Generative AI Agents}"


def test_search_by_caption_fulltext_returns_entity_or_query_and_details():
    query_generator = MagicMock()
    query_generator.generate.return_value = "{ORA-00923}"
    query_generator.get_morphological_analysis_details.return_value = "固有表現OR検索"
    database_service = MagicMock()
    database_service.search_by_fulltext.return_value = ([], "SQL")
    service = SearchService(MagicMock(), database_service, query_generator)

    _, executed_query, executed_sql, details = service.search_by_caption("ORA-00923 とは？", "全文検索", 8, 0.25, 0)

    assert executed_query == "{ORA-00923}"
    assert executed_sql == "SQL"
    assert details == "固有表現OR検索"
    database_service.search_by_fulltext.assert_called_once_with("{ORA-00923}", 8, 0)


def test_morphological_details_show_entity_or_search():
    generator = SearchQueryGenerator()

    details = generator.get_morphological_analysis_details("ORA-00923 とは何ですか？")

    assert "固有表現OR検索" in details
    assert "`ORA-00923`" in details
    assert "{ORA-00923}" in details


def test_resolve_agentic_fulltext_query_url_and_error_code():
    generator = SearchQueryGenerator()
    url = "https://qiita.com/yuji-arakawa/items/28f30a5434ba429f3f16"

    assert generator.resolve_agentic_fulltext_query(url) == f"{{{url}}}"
    assert generator.resolve_agentic_fulltext_query("ORA-00923") == "{ORA-00923}"


def test_resolve_agentic_fulltext_query_item_id_whole_brace():
    generator = SearchQueryGenerator()
    item_id = "28f30a5434ba429f3f16"

    assert generator.resolve_agentic_fulltext_query(item_id) == f"{{{item_id}}}"


def test_resolve_agentic_fulltext_query_passthrough_preformatted():
    generator = SearchQueryGenerator()
    prebuilt = "{ORA-00923} OR {ORA-00924}"

    assert generator.resolve_agentic_fulltext_query(prebuilt) == prebuilt


def test_resolve_agentic_fulltext_query_long_title_whole_brace_when_no_entity():
    generator = SearchQueryGenerator()
    title = "Retrieval-Augmented Generation for Large Language Models: A Survey abstract"

    resolved = generator.resolve_agentic_fulltext_query(title)
    assert resolved == f"{{{title}}}"


def test_agentic_morphological_details_section():
    generator = SearchQueryGenerator()
    from app.search_query_generator import FULLTEXT_QUERY_MODE_AGENTIC_EXACT

    details = generator.get_morphological_analysis_details(
        "28f30a5434ba429f3f16",
        fulltext_query_mode=FULLTEXT_QUERY_MODE_AGENTIC_EXACT,
    )
    assert "Agentic 完全一致検索" in details
    assert "{28f30a5434ba429f3f16}" in details
