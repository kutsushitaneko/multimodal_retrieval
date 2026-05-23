"""Step4: app/ui/events.py が共有 DatabaseService を使うことのユニットテスト"""
import inspect
from unittest.mock import MagicMock, patch

import gradio as gr

from app.ui.events import UIEvents


class TestUpdateDatabaseUsesSharedService:
  @patch("app.global_nlp_service.get_global_nlp_service", return_value=MagicMock())
  @patch("app.vlm_service_factory.VLMServiceFactory.create_upload_vlm_service", return_value=MagicMock())
  @patch("app.vlm_service_factory.VLMServiceFactory.create_search_vlm_service", return_value=MagicMock())
  def test_update_database_uses_search_service_database_service(
    self, _mock_search_vlm, _mock_upload_vlm, _mock_nlp
  ):
    shared_db = MagicMock(name="shared_database_service")
    shared_db.is_image_registered.return_value = True

    search_service = MagicMock()
    search_service.database_service = shared_db

    events = UIEvents(search_service)

    uploaded_file = MagicMock()
    uploaded_file.name = "/tmp/sample.jpg"

    result = events.update_database_with_registration_and_update(
      "generated",
      "edited",
      None,
      uploaded_file,
      "sample.jpg",
      "original",
    )

    shared_db.is_image_registered.assert_called_once_with("sample.jpg")
    assert "既に登録されています" in result[2]

  def test_update_database_does_not_create_new_config_or_pool(self):
    source = inspect.getsource(UIEvents.update_database_with_registration_and_update)
    assert "Config()" not in source
    assert "get_db_pool()" not in source
    assert "DatabaseService(db_pool)" not in source
    assert "self.search_service.database_service" in source
