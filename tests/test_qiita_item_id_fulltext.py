"""Qiita URL vs 記事ID単体の Agentic fulltext クエリ差分（調査用）。"""

from app.entity_patterns import extract_entities_from_text
from app.search_query_generator import SearchQueryGenerator

QIITA_URL = "https://qiita.com/yuji-arakawa/items/28f30a5434ba429f3f16"
QIITA_ITEM_ID = "28f30a5434ba429f3f16"


def test_url_is_extracted_as_entity_and_wrapped_in_braces():
    entities = extract_entities_from_text(QIITA_URL)
    assert any(e["text"] == QIITA_URL and e["type"] == "url" for e in entities)

    query = SearchQueryGenerator().resolve_agentic_fulltext_query(QIITA_URL)
    assert query == f"{{{QIITA_URL}}}"


def test_item_id_alone_is_not_extracted_as_entity():
    entities = extract_entities_from_text(QIITA_ITEM_ID)
    assert entities == []


def test_item_id_alone_uses_whole_brace_not_spacy_split():
    query = SearchQueryGenerator().resolve_agentic_fulltext_query(QIITA_ITEM_ID)

    assert query == f"{{{QIITA_ITEM_ID}}}"
    legacy = SearchQueryGenerator().generate(QIITA_ITEM_ID)
    assert QIITA_ITEM_ID not in legacy
    assert " " in legacy
