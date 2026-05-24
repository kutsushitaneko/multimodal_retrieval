from pathlib import Path


def test_batch_ingestion_uses_ui_vlm_service():
    source = Path("batch_injestion.py").read_text(encoding="utf-8")

    assert "VLMServiceFactory.create_upload_vlm_service()" in source
    assert "NLPService(upload_vlm_service)" in source
    assert "generate_caption_with_vlm" in source


def test_batch_ingestion_does_not_build_own_top_k_chat_request():
    source = Path("batch_injestion.py").read_text(encoding="utf-8")

    assert "chat_request.top_k = -1" not in source
    assert "EmbeddingService(" in source
