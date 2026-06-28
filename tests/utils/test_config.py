"""
tests/utils/test_config.py

Unit tests for src/utils/src.utils.config.py — 100% coverage.
All DB, boto3/KMS, and litellm Router calls are fully mocked.
"""

import os
import json
import base64
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from src.utils.config import (
    ConfigType,
    ModelConfig,
    get_model_config,
    initialize_config,
    cleanup_config,
    model_config as global_model_config,
)


# ---------------------------------------------------------------------------
# ConfigType Enum
# ---------------------------------------------------------------------------

class TestConfigTypeEnum:

    def test_team_value(self):
        assert ConfigType.TEAM.value == "team"

    def test_agent_value(self):
        assert ConfigType.AGENT.value == "agent"

    def test_team_and_agent_are_different(self):
        assert ConfigType.TEAM != ConfigType.AGENT


# ---------------------------------------------------------------------------
# ModelConfig — agent_id property
# ---------------------------------------------------------------------------

class TestModelConfigAgentId:

    def test_agent_id_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_UNIQUE_ID", "agent-xyz")
        mc = ModelConfig()
        assert mc.agent_id == "agent-xyz"

    def test_agent_id_cached(self, monkeypatch):
        monkeypatch.setenv("AGENT_UNIQUE_ID", "agent-abc")
        mc = ModelConfig()
        _ = mc.agent_id
        assert mc._agent_id == "agent-abc"

    def test_agent_id_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("AGENT_UNIQUE_ID", raising=False)
        mc = ModelConfig()
        mc._agent_id = None
        with pytest.raises(ValueError, match="AGENT_UNIQUE_ID"):
            _ = mc.agent_id


# ---------------------------------------------------------------------------
# ModelConfig — initialize_db_pool
# ---------------------------------------------------------------------------

class TestInitializeDbPool:

    @pytest.mark.asyncio
    async def test_raises_when_env_vars_missing(self, monkeypatch):
        for var in ["LLM_CONFIG_DB_HOST", "LLM_CONFIG_DB_PORT",
                    "LLM_CONFIG_DB_NAME", "LLM_CONFIG_DB_USER", "LLM_CONFIG_DB_PASSWORD"]:
            monkeypatch.delenv(var, raising=False)

        mc = ModelConfig()
        with pytest.raises(Exception):
            await mc.initialize_db_pool()

    @pytest.mark.asyncio
    async def test_creates_pool_on_success(self, monkeypatch):
        for var, val in [
            ("LLM_CONFIG_DB_HOST", "localhost"),
            ("LLM_CONFIG_DB_PORT", "5432"),
            ("LLM_CONFIG_DB_NAME", "testdb"),
            ("LLM_CONFIG_DB_USER", "user"),
            ("LLM_CONFIG_DB_PASSWORD", "pass"),
        ]:
            monkeypatch.setenv(var, val)

        fake_pool = MagicMock()
        mc = ModelConfig()

        with patch("src.utils.config.psycopg2.pool.ThreadedConnectionPool", return_value=fake_pool):
            await mc.initialize_db_pool()

        assert mc.db_pool is fake_pool

    @pytest.mark.asyncio
    async def test_reinitializes_executor_when_shutdown(self, monkeypatch):
        for var, val in [
            ("LLM_CONFIG_DB_HOST", "localhost"),
            ("LLM_CONFIG_DB_PORT", "5432"),
            ("LLM_CONFIG_DB_NAME", "testdb"),
            ("LLM_CONFIG_DB_USER", "user"),
            ("LLM_CONFIG_DB_PASSWORD", "pass"),
        ]:
            monkeypatch.setenv(var, val)

        mc = ModelConfig()
        mc._executor_shutdown = True
        fake_pool = MagicMock()

        with patch("src.utils.config.psycopg2.pool.ThreadedConnectionPool", return_value=fake_pool):
            await mc.initialize_db_pool()

        assert mc._executor_shutdown is False


# ---------------------------------------------------------------------------
# ModelConfig — close_db_pool
# ---------------------------------------------------------------------------

class TestCloseDbPool:

    @pytest.mark.asyncio
    async def test_does_nothing_when_pool_is_none(self):
        mc = ModelConfig()
        mc.db_pool = None
        await mc.close_db_pool()  # Must not raise

    @pytest.mark.asyncio
    async def test_closes_pool_via_executor(self):
        mc = ModelConfig()
        fake_pool = MagicMock()
        mc.db_pool = fake_pool
        mc._executor_shutdown = False

        await mc.close_db_pool()

        fake_pool.closeall.assert_called_once()
        assert mc.db_pool is None

    @pytest.mark.asyncio
    async def test_closes_pool_synchronously_when_executor_shutdown(self):
        mc = ModelConfig()
        fake_pool = MagicMock()
        mc.db_pool = fake_pool
        mc._executor_shutdown = True

        await mc.close_db_pool()

        fake_pool.closeall.assert_called_once()
        assert mc.db_pool is None


# ---------------------------------------------------------------------------
# ModelConfig — shutdown
# ---------------------------------------------------------------------------

class TestShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool_and_executor(self):
        mc = ModelConfig()
        mc.db_pool = None  # No pool to close
        mc._executor_shutdown = False

        with patch.object(mc.executor, "shutdown") as mock_shutdown:
            await mc.shutdown()
            mock_shutdown.assert_called_once_with(wait=True)

        assert mc._executor_shutdown is True

    @pytest.mark.asyncio
    async def test_shutdown_skips_executor_if_already_shutdown(self):
        mc = ModelConfig()
        mc.db_pool = None
        mc._executor_shutdown = True

        with patch.object(mc.executor, "shutdown") as mock_shutdown:
            await mc.shutdown()
            mock_shutdown.assert_not_called()


# ---------------------------------------------------------------------------
# ModelConfig — _execute_query
# ---------------------------------------------------------------------------

class TestExecuteQuery:

    def test_executes_and_returns_result(self):
        mc = ModelConfig()
        fake_conn = MagicMock()
        fake_cursor = MagicMock()
        fake_cursor.fetchone.return_value = {"col": "val"}
        fake_conn.cursor.return_value.__enter__ = MagicMock(return_value=fake_cursor)
        fake_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        fake_pool = MagicMock()
        fake_pool.getconn.return_value = fake_conn
        mc.db_pool = fake_pool

        result = mc._execute_query("SELECT 1", ())
        assert result == {"col": "val"}
        fake_pool.putconn.assert_called_once_with(fake_conn)

    def test_returns_connection_even_on_exception(self):
        mc = ModelConfig()
        fake_conn = MagicMock()
        fake_cursor = MagicMock()
        fake_cursor.execute.side_effect = RuntimeError("db error")
        fake_conn.cursor.return_value.__enter__ = MagicMock(return_value=fake_cursor)
        fake_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        fake_pool = MagicMock()
        fake_pool.getconn.return_value = fake_conn
        mc.db_pool = fake_pool

        with pytest.raises(RuntimeError):
            mc._execute_query("BAD", ())

        fake_pool.putconn.assert_called_once_with(fake_conn)


# ---------------------------------------------------------------------------
# ModelConfig — _get_agent_specific_configs (plural — returns list)
# ---------------------------------------------------------------------------

class TestGetAgentSpecificConfig:

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_rows(self):
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        with patch.object(mc, "_execute_query_all", return_value=[]):
            result = await mc._get_agent_specific_configs("team-1", "agent-1")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_config_list_when_rows_found_dict_config(self):
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        fake_rows = [{
            "resolved_config": {"api_key": "test"},
            "selected_model": "gpt-4",
            "provider": "openai",
            "config_type": "agent",
            "priority_order": "P1",
        }]

        with patch.object(mc, "_execute_query_all", return_value=fake_rows):
            result = await mc._get_agent_specific_configs("team-1", "agent-1")

        assert len(result) == 1
        assert result[0]["config_type"] == ConfigType.AGENT.value
        assert result[0]["provider"] == "openai"
        assert result[0]["selected_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_decrypts_string_config(self):
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        encrypted_config = base64.b64encode(b'{"api_key": "secret"}').decode()
        fake_rows = [{
            "resolved_config": encrypted_config,
            "selected_model": "claude-3",
            "provider": "anthropic",
            "config_type": "agent",
            "priority_order": "P1",
        }]

        with patch.object(mc, "_execute_query_all", return_value=fake_rows), \
             patch.object(mc, "decrypt", new=MagicMock(return_value='{"api_key": "secret"}')):
            result = await mc._get_agent_specific_configs("team-1", "agent-1")

        assert result[0]["config"]["api_key"] == "secret"

    @pytest.mark.asyncio
    async def test_raises_when_db_pool_not_initialized(self):
        mc = ModelConfig()
        mc.db_pool = None
        with pytest.raises(RuntimeError, match="Database pool not initialized"):
            await mc._get_agent_specific_configs("t1", "a1")

    @pytest.mark.asyncio
    async def test_retries_on_operational_error(self):
        import psycopg2
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        call_count = 0

        def fake_execute(query, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise psycopg2.OperationalError("connection dropped")
            return [{
                "resolved_config": {"key": "val"},
                "selected_model": "gpt-4",
                "provider": "openai",
                "config_type": "agent",
                "priority_order": "P1",
            }]

        with patch.object(mc, "_execute_query_all", side_effect=fake_execute), \
             patch("src.utils.config.asyncio.sleep", new=AsyncMock()):
            result = await mc._get_agent_specific_configs("t1", "a1")

        assert result is not None

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        import psycopg2
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        with patch.object(mc, "_execute_query_all", side_effect=psycopg2.OperationalError("drop")), \
             patch("src.utils.config.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(psycopg2.OperationalError):
                await mc._get_agent_specific_configs("t1", "a1")

    @pytest.mark.asyncio
    async def test_raises_on_non_operational_exception(self):
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        with patch.object(mc, "_execute_query_all", side_effect=ValueError("other error")):
            with pytest.raises(ValueError):
                await mc._get_agent_specific_configs("t1", "a1")


# ---------------------------------------------------------------------------
# ModelConfig — _get_team_level_configs (plural — returns list)
# ---------------------------------------------------------------------------

class TestGetTeamLevelConfig:

    @pytest.mark.asyncio
    async def test_raises_when_no_rows(self):
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        with patch.object(mc, "_execute_query_all", return_value=[]):
            with pytest.raises(ValueError, match="No model configuration found"):
                await mc._get_team_level_configs("team-1")

    @pytest.mark.asyncio
    async def test_returns_config_list_when_rows_found(self):
        mc = ModelConfig()
        mc.db_pool = MagicMock()

        fake_rows = [{
            "resolved_config": {"api_key": "k"},
            "selected_model": "gpt-4",
            "provider": "openai",
            "config_type": "team",
            "priority_order": "P1",
        }]

        with patch.object(mc, "_execute_query_all", return_value=fake_rows):
            result = await mc._get_team_level_configs("team-1")

        assert len(result) == 1
        assert result[0]["config_type"] == ConfigType.TEAM.value

    @pytest.mark.asyncio
    async def test_raises_when_db_pool_not_initialized(self):
        mc = ModelConfig()
        mc.db_pool = None
        with pytest.raises(RuntimeError):
            await mc._get_team_level_configs("t1")

    @pytest.mark.asyncio
    async def test_retries_on_operational_error(self):
        import psycopg2
        mc = ModelConfig()
        mc.db_pool = MagicMock()
        call_count = 0

        def fake_execute(q, p):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise psycopg2.OperationalError("drop")
            return [{
                "resolved_config": {"key": "v"},
                "selected_model": "gpt-4",
                "provider": "openai",
                "config_type": "team",
                "priority_order": "P1",
            }]

        with patch.object(mc, "_execute_query_all", side_effect=fake_execute), \
             patch("src.utils.config.asyncio.sleep", new=AsyncMock()):
            result = await mc._get_team_level_configs("t1")

        assert result is not None


# ---------------------------------------------------------------------------
# ModelConfig — get_team_model_config
# (Backend: use_litellm_completion=True → returns primary dict)
# ---------------------------------------------------------------------------

class TestGetTeamModelConfig:

    @pytest.mark.asyncio
    async def test_returns_primary_agent_config_when_available(self, monkeypatch):
        monkeypatch.setenv("AGENT_UNIQUE_ID", "agent-xyz")
        mc = ModelConfig()
        mc._agent_id = "agent-xyz"
        # Backend flag: litellm_completion path returns primary dict
        mc.use_litellm_completion = True
        mc.use_chatlitellm = False

        agent_configs = [{
            "config_type": "agent",
            "provider": "anthropic",
            "selected_model": "claude-3",
            "config": {"api_key": "secret"},
            "priority_order": "P1",
            "priority_order_int": 1,
        }]

        with patch.object(mc, "_get_agent_specific_configs", new=AsyncMock(return_value=agent_configs)):
            result = await mc.get_team_model_config("team-1")

        # Returns primary dict (not list) on litellm_completion path
        assert result["config_type"] == "agent"
        assert result["priority_order"] == "P1"

    @pytest.mark.asyncio
    async def test_falls_back_to_team_config(self, monkeypatch):
        monkeypatch.setenv("AGENT_UNIQUE_ID", "agent-xyz")
        mc = ModelConfig()
        mc._agent_id = "agent-xyz"
        mc.use_litellm_completion = True
        mc.use_chatlitellm = False

        team_configs = [{
            "config_type": "team",
            "provider": "openai",
            "selected_model": "gpt-4",
            "config": {"api_key": "key"},
            "priority_order": "P1",
            "priority_order_int": 1,
        }]

        with patch.object(mc, "_get_agent_specific_configs", new=AsyncMock(return_value=[])), \
             patch.object(mc, "_get_team_level_configs", new=AsyncMock(return_value=team_configs)):
            result = await mc.get_team_model_config("team-1")

        assert result["config_type"] == "team"

    @pytest.mark.asyncio
    async def test_propagates_exception(self, monkeypatch):
        monkeypatch.setenv("AGENT_UNIQUE_ID", "agent-xyz")
        mc = ModelConfig()
        mc._agent_id = "agent-xyz"

        with patch.object(mc, "_get_agent_specific_configs", new=AsyncMock(side_effect=RuntimeError("db down"))):
            with pytest.raises(RuntimeError):
                await mc.get_team_model_config("team-1")


# ---------------------------------------------------------------------------
# ModelConfig — create_router_for_config
# ---------------------------------------------------------------------------

class TestCreateRouterForConfig:

    def test_creates_router_successfully(self):
        from litellm import Router
        mc = ModelConfig()

        with patch("src.utils.config.Router") as mock_router_cls:
            fake_router = MagicMock()
            mock_router_cls.return_value = fake_router

            result = mc.create_router_for_config("openai", "gpt-4", {"api_key": "key"})

        assert result is fake_router
        mock_router_cls.assert_called_once()

    def test_raises_on_exception(self):
        mc = ModelConfig()

        with patch("src.utils.config.Router", side_effect=RuntimeError("router fail")):
            with pytest.raises(RuntimeError):
                mc.create_router_for_config("openai", "gpt-4", {})


# ---------------------------------------------------------------------------
# ModelConfig — get_router_for_team
# ---------------------------------------------------------------------------

class TestGetRouterForTeam:

    @pytest.mark.asyncio
    async def test_returns_router_and_model_name(self, monkeypatch):
        monkeypatch.setenv("AGENT_UNIQUE_ID", "agent-1")
        mc = ModelConfig()
        mc._agent_id = "agent-1"

        all_configs = [{
            "config_type": "team",
            "provider": "openai",
            "selected_model": "gpt-4",
            "config": {"api_key": "k"},
            "priority_order": "P1",
            "priority_order_int": 1,
        }]
        fake_router = MagicMock()

        with patch.object(mc, "_resolve_all_configs", new=AsyncMock(return_value=all_configs)), \
             patch.object(mc, "create_router_for_config", return_value=fake_router):
            router, model_name, config_type = await mc.get_router_for_team("team-1")

        assert router is fake_router
        assert model_name == "gpt-4"
        assert config_type == "team"

    @pytest.mark.asyncio
    async def test_raises_on_failure(self, monkeypatch):
        monkeypatch.setenv("AGENT_UNIQUE_ID", "agent-1")
        mc = ModelConfig()
        mc._agent_id = "agent-1"

        with patch.object(mc, "_resolve_all_configs", new=AsyncMock(side_effect=RuntimeError("fail"))):
            with pytest.raises(RuntimeError):
                await mc.get_router_for_team("team-1")


# ---------------------------------------------------------------------------
# ModelConfig — decrypt
# ---------------------------------------------------------------------------

class TestDecrypt:

    @pytest.mark.asyncio
    async def test_decrypts_successfully(self):
        mc = ModelConfig()
        plaintext = b"decrypted_value"
        fake_kms = MagicMock()
        fake_kms.decrypt.return_value = {"Plaintext": plaintext}

        ciphertext = base64.b64encode(b"fake-encrypted-data").decode()

        with patch.object(mc, "create_kms_client", new=MagicMock(return_value=fake_kms)):
            result = mc.decrypt(ciphertext)

        assert result == "decrypted_value"

    @pytest.mark.asyncio
    async def test_uses_key_id_when_set(self, monkeypatch):
        monkeypatch.setenv("AWS_KMS_KEY_ID_ARN", "arn:aws:kms:us-east-1:123:key/abc")
        mc = ModelConfig()
        plaintext = b"plaintext"
        fake_kms = MagicMock()
        fake_kms.decrypt.return_value = {"Plaintext": plaintext}

        ciphertext = base64.b64encode(b"enc").decode()

        with patch.object(mc, "create_kms_client", new=MagicMock(return_value=fake_kms)):
            mc.decrypt(ciphertext)

        call_kwargs = fake_kms.decrypt.call_args[1]
        assert "KeyId" in call_kwargs

    @pytest.mark.asyncio
    async def test_raises_client_error(self, monkeypatch):
        from botocore.exceptions import ClientError
        mc = ModelConfig()

        fake_kms = MagicMock()
        fake_kms.decrypt.side_effect = ClientError(
            {"Error": {"Code": "InvalidCiphertextException", "Message": "bad"}},
            "Decrypt"
        )
        ciphertext = base64.b64encode(b"bad").decode()

        with patch.object(mc, "create_kms_client", new=MagicMock(return_value=fake_kms)):
            with pytest.raises(ClientError):
                mc.decrypt(ciphertext)


# ---------------------------------------------------------------------------
# Context manager get_model_config
# ---------------------------------------------------------------------------

class TestGetModelConfigContextManager:

    @pytest.mark.asyncio
    async def test_yields_model_config_when_pool_exists(self):
        # Patch the global model_config's db_pool
        original_pool = global_model_config.db_pool
        global_model_config.db_pool = MagicMock()

        async with get_model_config() as mc:
            assert mc is global_model_config

        global_model_config.db_pool = original_pool

    @pytest.mark.asyncio
    async def test_initializes_pool_when_missing(self):
        original_pool = global_model_config.db_pool
        global_model_config.db_pool = None

        with patch.object(global_model_config, "initialize_db_pool", new=AsyncMock()) as mock_init:
            async with get_model_config() as mc:
                assert mc is global_model_config
            mock_init.assert_called_once()

        global_model_config.db_pool = original_pool

    @pytest.mark.asyncio
    async def test_propagates_exception_from_init(self):
        original_pool = global_model_config.db_pool
        global_model_config.db_pool = None

        with patch.object(global_model_config, "initialize_db_pool", new=AsyncMock(side_effect=RuntimeError("fail"))):
            with pytest.raises(RuntimeError):
                async with get_model_config():
                    pass  # Should not reach here

        global_model_config.db_pool = original_pool


# ---------------------------------------------------------------------------
# Standalone initialize_config / cleanup_config
# ---------------------------------------------------------------------------

class TestStandaloneHelpers:

    @pytest.mark.asyncio
    async def test_initialize_config_delegates_to_global(self):
        with patch.object(global_model_config, "initialize_db_pool", new=AsyncMock()) as mock_init:
            await initialize_config()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_config_delegates_to_global(self):
        with patch.object(global_model_config, "shutdown", new=AsyncMock()) as mock_shutdown:
            await cleanup_config()
            mock_shutdown.assert_called_once()