"""
tests/llm/test_litellm_client.py

Unit tests for src/llm/litellm_client.py → LitellmClient class.
All external LLM calls (litellm.acompletion), config lookups, and
tracing utilities are fully mocked.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client():
    """
    Import and instantiate LitellmClient with every external dependency
    patched so no real network calls are made.
    """
    with (
        patch("src.llm.litellm_client.get_model_config"),
        patch("src.llm.litellm_client.LLMUsageTracker"),
        patch("src.llm.litellm_client.extract_and_log_reasoning",
              return_value=("cleaned response", None)),
        patch("src.llm.litellm_client.acompletion"),
    ):
        from src.llm.litellm_client import LitellmClient
        return LitellmClient()


FAKE_TEAM_CONFIG = {
    "selected_model": "gpt-4o",
    "provider": "openai",
    "config": {"temperature": 0.0, "max_tokens": 1000},
}

FAKE_LLM_PARAMS = {
    "model": "openai/gpt-4o",
    "temperature": 0.0,
    "max_tokens": 1000,
}


# ===========================================================================
# __init__
# ===========================================================================

class TestLitellmClientInit:

    def test_initial_model_is_none(self):
        client = _build_client()
        assert client.model is None

    def test_initial_team_config_is_none(self):
        client = _build_client()
        assert client.team_config is None


# ===========================================================================
# get_dynamic_llm_instance
# ===========================================================================

class TestGetDynamicLlmInstance:

    @pytest.mark.asyncio
    async def test_returns_llm_params_dict(self):
        mock_config_ctx = MagicMock()
        mock_config_ctx.__aenter__ = AsyncMock(return_value=mock_config_ctx)
        mock_config_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_config_ctx.get_team_model_config = AsyncMock(
            return_value=FAKE_TEAM_CONFIG
        )

        with (
            patch(
                "src.llm.litellm_client.get_model_config",
                return_value=mock_config_ctx,
            ),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                return_value=("resp", None),
            ),
            patch("src.llm.litellm_client.acompletion"),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            result = await client.get_dynamic_llm_instance("team-123")

        assert result["model"] == "openai/gpt-4o"
        assert result["temperature"] == 0.0
        assert result["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_sets_model_and_team_config_attributes(self):
        mock_config_ctx = MagicMock()
        mock_config_ctx.__aenter__ = AsyncMock(return_value=mock_config_ctx)
        mock_config_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_config_ctx.get_team_model_config = AsyncMock(
            return_value=FAKE_TEAM_CONFIG
        )

        with (
            patch(
                "src.llm.litellm_client.get_model_config",
                return_value=mock_config_ctx,
            ),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                return_value=("resp", None),
            ),
            patch("src.llm.litellm_client.acompletion"),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            await client.get_dynamic_llm_instance("team-123")

        assert client.model == "gpt-4o"
        assert client.team_config is not None

    @pytest.mark.asyncio
    async def test_raises_value_error_on_config_failure(self):
        mock_config_ctx = MagicMock()
        mock_config_ctx.__aenter__ = AsyncMock(return_value=mock_config_ctx)
        mock_config_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_config_ctx.get_team_model_config = AsyncMock(
            side_effect=Exception("Config DB unreachable")
        )

        with (
            patch(
                "src.llm.litellm_client.get_model_config",
                return_value=mock_config_ctx,
            ),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                return_value=("resp", None),
            ),
            patch("src.llm.litellm_client.acompletion"),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            with pytest.raises(ValueError, match="team-bad"):
                await client.get_dynamic_llm_instance("team-bad")

    @pytest.mark.asyncio
    async def test_provider_model_format(self):
        team_config = {
            "selected_model": "claude-3-sonnet",
            "provider": "anthropic",
            "config": {},
        }
        mock_config_ctx = MagicMock()
        mock_config_ctx.__aenter__ = AsyncMock(return_value=mock_config_ctx)
        mock_config_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_config_ctx.get_team_model_config = AsyncMock(return_value=team_config)

        with (
            patch(
                "src.llm.litellm_client.get_model_config",
                return_value=mock_config_ctx,
            ),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                return_value=("resp", None),
            ),
            patch("src.llm.litellm_client.acompletion"),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            result = await client.get_dynamic_llm_instance("team-x")

        assert result["model"] == "anthropic/claude-3-sonnet"


# ===========================================================================
# generate_response
# ===========================================================================

class TestGenerateResponse:

    def _patched_client(self, acompletion_mock, extract_mock):
        with (
            patch("src.llm.litellm_client.get_model_config"),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                new=extract_mock,
            ),
            patch("src.llm.litellm_client.acompletion", new=acompletion_mock),
        ):
            from src.llm.litellm_client import LitellmClient

            return LitellmClient()

    @pytest.mark.asyncio
    async def test_returns_string_response(self):
        from importlib import reload
        import src.llm.litellm_client as module
        reload(module)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from LLM"

        acompletion_mock = AsyncMock(return_value=mock_response)
        extract_mock = MagicMock(return_value=("Hello from LLM", None))

        with (
            patch("src.llm.litellm_client.get_model_config"),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                side_effect=extract_mock,
            ),
            patch("src.llm.litellm_client.acompletion", acompletion_mock),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            result = await client.generate_response(
                FAKE_LLM_PARAMS,
                [{"role": "user", "content": "Hello"}],
                "auth-token",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_passes_llm_params_to_acompletion(self):
        from importlib import reload
        import src.llm.litellm_client as mod
        reload(mod)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"

        acompletion_mock = AsyncMock(return_value=mock_response)

        with (
            patch("src.llm.litellm_client.get_model_config"),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                return_value=("response", None),
            ),
            patch("src.llm.litellm_client.acompletion", acompletion_mock),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            messages = [{"role": "user", "content": "test"}]
            await client.generate_response(FAKE_LLM_PARAMS, messages, "tok")

        call_kwargs = acompletion_mock.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs["messages"] == messages

    @pytest.mark.asyncio
    async def test_tracks_token_usage(self):
        from importlib import reload
        import src.llm.litellm_client as mod
        reload(mod)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"

        acompletion_mock = AsyncMock(return_value=mock_response)
        tracker_mock = MagicMock()
        tracker_cls_mock = MagicMock(return_value=tracker_mock)

        with (
            patch("src.llm.litellm_client.get_model_config"),
            patch("src.llm.litellm_client.LLMUsageTracker", tracker_cls_mock),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                return_value=("response", None),
            ),
            patch("src.llm.litellm_client.acompletion", acompletion_mock),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            await client.generate_response(
                FAKE_LLM_PARAMS,
                [{"role": "user", "content": "x"}],
                "tok",
            )

        tracker_mock.track_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_strips_reasoning_from_response(self):
        """extract_and_log_reasoning should be called and its result returned."""
        from importlib import reload
        import src.llm.litellm_client as mod
        reload(mod)
        raw_content = "<reasoning>internal</reasoning>Clean answer."
        clean_content = "Clean answer."

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = raw_content

        acompletion_mock = AsyncMock(return_value=mock_response)
        extract_mock = MagicMock(return_value=(clean_content, "internal"))

        with (
            patch("src.llm.litellm_client.get_model_config"),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                side_effect=extract_mock,
            ),
            patch("src.llm.litellm_client.acompletion", acompletion_mock),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            result = await client.generate_response(
                FAKE_LLM_PARAMS,
                [{"role": "user", "content": "x"}],
                "tok",
            )

        assert result == clean_content

    @pytest.mark.asyncio
    async def test_raises_on_acompletion_error(self):
        from importlib import reload
        import src.llm.litellm_client as mod
        reload(mod)
        acompletion_mock = AsyncMock(side_effect=Exception("LLM unavailable"))

        with (
            patch("src.llm.litellm_client.get_model_config"),
            patch("src.llm.litellm_client.LLMUsageTracker"),
            patch(
                "src.llm.litellm_client.extract_and_log_reasoning",
                return_value=("x", None),
            ),
            patch("src.llm.litellm_client.acompletion", acompletion_mock),
        ):
            from src.llm.litellm_client import LitellmClient

            client = LitellmClient()
            with pytest.raises(Exception, match="LLM unavailable"):
                await client.generate_response(
                    FAKE_LLM_PARAMS,
                    [{"role": "user", "content": "x"}],
                    "tok",
                )
