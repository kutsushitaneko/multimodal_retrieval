"""Step1: app/config.py の DB プール管理のユニットテスト"""
from unittest.mock import MagicMock, patch

import oracledb
import pytest

from app.config import (
    POOL_ALIAS,
    DEFAULT_PING_INTERVAL,
    DEFAULT_PING_TIMEOUT,
    Config,
)


@pytest.fixture
def config_instance():
    with patch.object(Config, "_validate_env_vars"), patch.object(Config, "_init_config"):
        config = Config()
        config.db_user = "test_user"
        config.db_password = "test_pass"
        config.db_dsn = "test_dsn"
        yield config


@pytest.fixture
def mock_oracledb():
    with patch("app.config.oracledb") as mocked:
        mocked.POOL_GETMODE_WAIT = oracledb.POOL_GETMODE_WAIT
        mocked.get_pool.return_value = None
        mocked.create_pool.return_value = MagicMock(name="new_pool")
        yield mocked


class TestGetDbPool:
    def test_creates_pool_with_ping_and_alias(self, config_instance, mock_oracledb):
        pool = config_instance.get_db_pool()

        mock_oracledb.create_pool.assert_called_once_with(
            user="test_user",
            password="test_pass",
            dsn="test_dsn",
            min=2,
            max=10,
            increment=1,
            timeout=60,
            getmode=oracledb.POOL_GETMODE_WAIT,
            ping_interval=DEFAULT_PING_INTERVAL,
            ping_timeout=DEFAULT_PING_TIMEOUT,
            pool_alias=POOL_ALIAS,
        )
        assert pool is mock_oracledb.create_pool.return_value

    def test_returns_cached_pool_without_creating_new(self, config_instance, mock_oracledb):
        cached = MagicMock(name="cached_pool")
        mock_oracledb.get_pool.return_value = cached

        pool = config_instance.get_db_pool()

        mock_oracledb.create_pool.assert_not_called()
        assert pool is cached


class TestCloseDbPool:
    def test_closes_pool_when_exists(self, mock_oracledb):
        pool = MagicMock()
        mock_oracledb.get_pool.return_value = pool

        Config.close_db_pool()

        pool.close.assert_called_once()

    def test_does_nothing_when_pool_missing(self, mock_oracledb):
        mock_oracledb.get_pool.return_value = None

        Config.close_db_pool()

        mock_oracledb.get_pool.assert_called_once_with(POOL_ALIAS)
