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
