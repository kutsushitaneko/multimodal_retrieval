"""Step3: app/database_service.py のリトライ・接続破棄のユニットテスト"""
from unittest.mock import MagicMock, patch

import oracledb
import pytest

from app.database_service import CONNECTION_ERROR_CODES, DatabaseService


def _make_connection_error(code=3113):
    error = MagicMock()
    error.code = code
    return oracledb.DatabaseError(error)


class TestConnectionErrorDetection:
    def test_is_connection_error_true_for_known_codes(self):
        for code in CONNECTION_ERROR_CODES:
            assert DatabaseService._is_connection_error(_make_connection_error(code))

    def test_is_connection_error_false_for_other_codes(self):
        assert not DatabaseService._is_connection_error(_make_connection_error(1))

    def test_is_connection_error_false_for_non_database_error(self):
        assert not DatabaseService._is_connection_error(ValueError("x"))


class TestExecuteWithRetry:
    @pytest.fixture
    def service(self):
        pool = MagicMock()
        pool.acquire.return_value = MagicMock(name="conn")
        return DatabaseService(pool)

    def test_success_on_first_attempt(self, service):
        conn = service.db_pool.acquire.return_value

        def operation(received_conn):
            assert received_conn is conn
            return "ok"

        assert service._execute_with_retry(operation) == "ok"
        service.db_pool.acquire.assert_called_once()
        conn.close.assert_called_once()
        service.db_pool.drop.assert_not_called()

    def test_retries_and_drops_connection_on_connection_error(self, service):
        conn1 = MagicMock(name="conn1")
        conn2 = MagicMock(name="conn2")
        service.db_pool.acquire.side_effect = [conn1, conn2]

        calls = {"count": 0}

        def operation(_conn):
            calls["count"] += 1
            if calls["count"] == 1:
                raise _make_connection_error()
            return "recovered"

        with patch.object(service, "retry_delay", 0):
            result = service._execute_with_retry(operation)

        assert result == "recovered"
        service.db_pool.drop.assert_called_once_with(conn1)
        conn1.close.assert_not_called()
        conn2.close.assert_called_once()

    def test_raises_non_connection_database_error_without_retry(self, service):
        def operation(_conn):
            raise _make_connection_error(code=1)

        with pytest.raises(oracledb.DatabaseError):
            service._execute_with_retry(operation)

        service.db_pool.acquire.assert_called_once()
        service.db_pool.drop.assert_not_called()

    def test_raises_after_max_retries(self, service):
        service.max_retries = 2
        conn = MagicMock()
        service.db_pool.acquire.return_value = conn

        def operation(_conn):
            raise _make_connection_error()

        with patch.object(service, "retry_delay", 0):
            with pytest.raises(oracledb.DatabaseError):
                service._execute_with_retry(operation)

        assert service.db_pool.acquire.call_count == 2
        assert service.db_pool.drop.call_count == 2
