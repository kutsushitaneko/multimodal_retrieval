from unittest.mock import MagicMock

from app.search_query_generator import FULLTEXT_QUERY_MODE_AGENTIC_EXACT, SearchQueryGenerator
from app.search_service import SearchService


def test_search_by_caption_agentic_exact_uses_resolve_not_generate():
    query_generator = SearchQueryGenerator()
    database_service = MagicMock()
    database_service.search_by_fulltext.return_value = ([], "SQL")
    service = SearchService(MagicMock(), database_service, query_generator)

    _, executed_query, _, details = service.search_by_caption(
        "28f30a5434ba429f3f16",
        "全文検索",
        8,
        0.25,
        0,
        fulltext_query_mode=FULLTEXT_QUERY_MODE_AGENTIC_EXACT,
    )

    assert executed_query == "{28f30a5434ba429f3f16}"
    database_service.search_by_fulltext.assert_called_once_with("{28f30a5434ba429f3f16}", 8, 0)
    assert "Agentic 完全一致検索" in details


def test_search_by_caption_legacy_still_uses_generate_for_natural_language():
    query_generator = SearchQueryGenerator()
    database_service = MagicMock()
    database_service.search_by_fulltext.return_value = ([], "SQL")
    service = SearchService(MagicMock(), database_service, query_generator)

    _, executed_query, _, _ = service.search_by_caption(
        "赤い猫と寺院",
        "全文検索",
        8,
        0.25,
        0,
    )

    assert " AND " in executed_query
    assert " OR " not in executed_query or "{" not in executed_query
