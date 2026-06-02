from app.entity_patterns import extract_entities_from_text, load_entity_pattern_specs


def test_load_entity_pattern_specs_reads_config():
    specs = load_entity_pattern_specs()
    assert specs
    assert any(spec.get("type") == "error_code" for spec in specs)


def test_extract_entities_finds_error_code():
    entities = extract_entities_from_text("ORA-00923 とは何ですか？")
    texts = [entity["text"] for entity in entities]
    assert "ORA-00923" in texts
