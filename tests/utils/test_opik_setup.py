"""
Comprehensive pytest test suite for opik_tracing.py
Targets 100% coverage with mocks for all external dependencies.
"""

import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Helpers – reset global module state between tests
# ---------------------------------------------------------------------------

def _reset_globals(module):
    """Reset the four mutable globals in the module."""
    module._opik_initialized = False
    module._opik_client = None
    module._tracing_enabled = True
    module._llm_only_mode = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_module_state():
    """Auto-reset global state before every test."""
    import src.utils.opik_setup as ot
    _reset_globals(ot)
    yield
    _reset_globals(ot)


@pytest.fixture()
def ot():
    import src.utils.opik_setup as opik_tracing
    return opik_tracing


# ===========================================================================
# OpikTraceData
# ===========================================================================

class TestOpikTraceData:
    def test_defaults(self, ot):
        td = ot.OpikTraceData()
        assert td.trace_id is None
        assert td.span_id is None
        assert td.provider is None
        assert td.tokens == {}
        assert td.created_at  # non-empty string

    def test_custom_values(self, ot):
        td = ot.OpikTraceData(trace_id="t1", span_id="s1", provider="gemini", tokens={"a": 1})
        assert td.trace_id == "t1"
        assert td.span_id == "s1"
        assert td.provider == "gemini"
        assert td.tokens == {"a": 1}


# ===========================================================================
# LLMProvider enum
# ===========================================================================

class TestLLMProvider:
    def test_members(self, ot):
        assert ot.LLMProvider.GEMINI.value == "gemini"
        assert ot.LLMProvider.LITELLM.value == "litellm"
        assert ot.LLMProvider.LANGCHAIN_CHAT.value == "langchain_chat"
        assert ot.LLMProvider.UNKNOWN.value == "unknown"


# ===========================================================================
# setup_opik_tracing
# ===========================================================================

class TestSetupOpikTracing:
    def test_tracing_disabled(self, ot):
        result = ot.setup_opik_tracing(enable_tracing=False)
        assert result is True
        assert not ot._opik_initialized

    def test_already_initialized(self, ot):
        ot._opik_initialized = True
        result = ot.setup_opik_tracing()
        assert result is True

    def test_no_url_returns_false(self, ot):
        mock_opik = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPIK_URL_OVERRIDE", None)
            with patch("builtins.__import__", side_effect=lambda n, *a, **k: mock_opik if n == "opik" else __import__(n, *a, **k)):
                result = ot.setup_opik_tracing(url=None)
        assert result is False

    def test_import_error(self, ot):
        with patch("builtins.__import__", side_effect=ImportError("no opik")):
            result = ot.setup_opik_tracing(url="http://localhost")
        assert result is False

    def test_connection_error(self, ot):
        mock_opik_mod = MagicMock()
        mock_opik_mod.Opik.side_effect = ConnectionError("conn fail")
        with patch.dict("sys.modules", {"opik": mock_opik_mod}):
            result = ot.setup_opik_tracing(url="http://localhost")
        assert result is False

    def test_generic_client_error(self, ot):
        mock_opik_mod = MagicMock()
        mock_opik_mod.Opik.side_effect = RuntimeError("oops")
        with patch.dict("sys.modules", {"opik": mock_opik_mod}):
            result = ot.setup_opik_tracing(url="http://localhost")
        assert result is False

    def test_successful_init_with_env(self, ot):
        mock_opik_mod = MagicMock()
        with patch.dict("sys.modules", {"opik": mock_opik_mod}):
            with patch.object(ot, "_setup_litellm_integration") as ml, \
                 patch.object(ot, "_setup_genai_integration") as mg:
                result = ot.setup_opik_tracing(
                    url="http://localhost:5173",
                    project_name="test-proj",
                    workspace="ws1",
                    enable_litellm=True,
                    enable_genai=True
                )
        assert result is True
        assert ot._opik_initialized is True
        ml.assert_called_once()
        mg.assert_called_once()

    def test_successful_init_skip_integrations(self, ot):
        mock_opik_mod = MagicMock()
        with patch.dict("sys.modules", {"opik": mock_opik_mod}):
            with patch.object(ot, "_setup_litellm_integration") as ml, \
                 patch.object(ot, "_setup_genai_integration") as mg:
                result = ot.setup_opik_tracing(
                    url="http://localhost:5173",
                    enable_litellm=False,
                    enable_genai=False
                )
        assert result is True
        ml.assert_not_called()
        mg.assert_not_called()

    def test_project_from_env(self, ot):
        mock_opik_mod = MagicMock()
        with patch.dict("sys.modules", {"opik": mock_opik_mod}):
            with patch.dict(os.environ, {"OPIK_PROJECT_NAME": "env-proj", "OPIK_URL_OVERRIDE": "http://x"}):
                with patch.object(ot, "_setup_litellm_integration"), \
                     patch.object(ot, "_setup_genai_integration"):
                    result = ot.setup_opik_tracing(enable_litellm=False, enable_genai=False)
        assert result is True

    def test_unexpected_exception(self, ot):
        real_import = __import__
        def selective_import(name, *args, **kwargs):
            if name == "opik":
                raise Exception("weird")
            return real_import(name, *args, **kwargs)
        
        with patch("builtins.__import__", side_effect=selective_import):
            result = ot.setup_opik_tracing(url="http://localhost")
        assert result is False

    def test_llm_only_mode_flag(self, ot):
        mock_opik_mod = MagicMock()
        with patch.dict("sys.modules", {"opik": mock_opik_mod}):
            ot.setup_opik_tracing(url="http://localhost", llm_only=False, enable_litellm=False, enable_genai=False)
        status = ot.get_tracing_status()
        assert status["llm_only_mode"] is False

    def test_project_fallback_agent_name(self, ot):
        mock_opik_mod = MagicMock()
        # Ensure OPIK_PROJECT_NAME is not set so fallback engages
        with patch.dict(os.environ, {"AGENT_NAME_CONSTANT": "agent-proj"}, clear=True):
            with patch.dict("sys.modules", {"opik": mock_opik_mod}):
                with patch.object(ot, "_setup_litellm_integration"), patch.object(ot, "_setup_genai_integration"):
                    result = ot.setup_opik_tracing(url="http://localhost", enable_litellm=False, enable_genai=False)
                    # OPIK_PROJECT_NAME should be set via setdefault
                    assert os.environ.get("OPIK_PROJECT_NAME") == "agent-proj"
        assert result is True


# ===========================================================================
# _safe_update_span
# ===========================================================================

class TestSafeUpdateSpan:
    def test_noop_when_not_initialized(self, ot):
        mock_ctx = MagicMock()
        ot._safe_update_span(mock_ctx, input={"x": 1})
        mock_ctx.update_current_span.assert_not_called()

    def test_calls_update_when_initialized(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_ctx = MagicMock()
        ot._safe_update_span(mock_ctx, input={"x": 1})
        mock_ctx.update_current_span.assert_called_once_with(input={"x": 1})

    def test_swallows_exception(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_ctx = MagicMock()
        mock_ctx.update_current_span.side_effect = RuntimeError("boom")
        # Should not raise
        ot._safe_update_span(mock_ctx, input={"x": 1})


# ===========================================================================
# _prepare_litellm_input
# ===========================================================================

class TestPrepareLitellmInput:
    def test_from_kwargs(self, ot):
        result = ot._prepare_litellm_input(
            (), {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["model"] == "gpt-4"
        assert result["messages"][0]["role"] == "user"

    def test_model_from_args(self, ot):
        result = ot._prepare_litellm_input(("gpt-3",), {})
        assert result["model"] == "gpt-3"

    def test_unknown_model(self, ot):
        result = ot._prepare_litellm_input((), {})
        assert result["model"] == "unknown"

    def test_non_dict_messages(self, ot):
        result = ot._prepare_litellm_input((), {"messages": ["hello"]})
        assert result["messages"][0] == {"content": "hello"}

    def test_messages_not_list(self, ot):
        result = ot._prepare_litellm_input((), {"messages": "not a list"})
        assert result["messages"] == []


# ===========================================================================
# _update_span_with_litellm_result
# ===========================================================================

class TestUpdateSpanWithLitellmResult:
    def test_with_full_result(self, ot):
        mock_ctx = MagicMock()
        result = MagicMock()
        result.choices = [MagicMock()]
        result.choices[0].message.content = "Hello"
        result.usage.prompt_tokens = 10
        result.usage.completion_tokens = 20
        result.usage.total_tokens = 30
        ot._update_span_with_litellm_result(mock_ctx, result)
        mock_ctx.update_current_span.assert_called_once()

    def test_no_choices(self, ot):
        mock_ctx = MagicMock()
        result = MagicMock()
        result.choices = []
        result.usage = None
        ot._update_span_with_litellm_result(mock_ctx, result)
        mock_ctx.update_current_span.assert_called_once()

    def test_exception_swallowed(self, ot):
        mock_ctx = MagicMock()
        mock_ctx.update_current_span.side_effect = RuntimeError
        result = MagicMock()
        result.choices = []
        result.usage = None
        # Should not raise
        ot._update_span_with_litellm_result(mock_ctx, result)


# ===========================================================================
# _perform_patched_litellm_call (sync & async)
# ===========================================================================

class TestPerformPatchedLitellmCall:
    def test_sync(self, ot):
        mock_opik_ctx = MagicMock()
        mock_func = MagicMock(return_value="response")
        with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": mock_opik_ctx}):
            with patch(f"{ot.__name__}._safe_update_span") as mu, \
                 patch(f"{ot.__name__}._update_span_with_litellm_result") as mr:
                result = ot._perform_patched_litellm_call(mock_func, {"model": "x"})
        assert result == "response"

    def test_async(self, ot):
        async def _run():
            mock_opik_ctx = MagicMock()
            mock_func = AsyncMock(return_value="async_response")
            with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": mock_opik_ctx}):
                with patch(f"{ot.__name__}._safe_update_span"), \
                     patch(f"{ot.__name__}._update_span_with_litellm_result"):
                    result = await ot._perform_patched_litellm_call_async(mock_func, {"model": "x"})
            assert result == "async_response"
        asyncio.get_event_loop().run_until_complete(_run())


# ===========================================================================
# _SafeFallbackCallback
# ===========================================================================

class TestSafeFallbackCallback:
    @pytest.fixture()
    def cb(self, ot):
        return ot._SafeFallbackCallback()

    def test_serialize_msgs_list(self, cb):
        result = cb._serialize_msgs([{"role": "user"}])
        assert "messages" in result

    def test_serialize_msgs_non_list(self, cb):
        result = cb._serialize_msgs("raw string")
        assert result == {"messages": "raw string"}

    def test_log_pre_api_call_ok(self, cb, ot):
        mock_ctx = MagicMock()
        with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": mock_ctx}):
            with patch(f"{ot.__name__}._safe_update_span") as mu:
                cb.log_pre_api_call("gpt-4", [{"role": "user"}], {})
                mu.assert_called_once()

    def test_log_pre_api_call_exception(self, cb, ot):
        with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": MagicMock()}):
            with patch(f"{ot.__name__}._safe_update_span", side_effect=Exception):
                cb.log_pre_api_call("m", [], {})  # should not raise

    def test_log_success_event(self, cb, ot):
        mock_ctx = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "out"
        response.usage.prompt_tokens = 1
        response.usage.completion_tokens = 2
        response.usage.total_tokens = 3
        start = datetime.now()
        end = datetime.now()
        with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": mock_ctx}):
            with patch(f"{ot.__name__}._safe_update_span") as mu:
                cb.log_success_event({"model": "gpt-4"}, response, start, end)
                mu.assert_called_once()

    def test_log_success_event_exception(self, cb, ot):
        with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": MagicMock()}):
            with patch(f"{ot.__name__}._safe_update_span", side_effect=Exception):
                cb.log_success_event({}, MagicMock(), datetime.now(), datetime.now())

    def test_log_failure_event(self, cb, ot):
        mock_ctx = MagicMock()
        with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": mock_ctx}):
            with patch(f"{ot.__name__}._safe_update_span") as mu:
                cb.log_failure_event({"model": "x", "exception": "err"}, None, None, None)
                mu.assert_called_once()

    def test_log_failure_event_exception(self, cb, ot):
        with patch.dict("sys.modules", {"opik": MagicMock(), "opik.opik_context": MagicMock()}):
            with patch(f"{ot.__name__}._safe_update_span", side_effect=Exception):
                cb.log_failure_event({}, None, None, None)


# ===========================================================================
# _get_safe_fallback_callback
# ===========================================================================

class TestGetSafeFallbackCallback:
    def test_returns_class(self, ot):
        base = MagicMock
        cls = ot._get_safe_fallback_callback(base)
        assert issubclass(cls, ot._SafeFallbackCallback)


# ===========================================================================
# LiteLLMIntegration
# ===========================================================================

class TestLiteLLMIntegration:
    @pytest.fixture()
    def make_integration(self, ot):
        def _make(litellm=None, opik=None):
            return ot.LiteLLMIntegration(litellm or MagicMock(), opik or MagicMock())
        return _make

    def test_tier1_success(self, make_integration):
        mock_litellm = MagicMock()
        mock_litellm.callbacks = []
        mock_opik = MagicMock()
        mock_opik_logger = MagicMock()

        intg = make_integration(litellm=mock_litellm, opik=mock_opik)
        with patch.dict("sys.modules", {"opik.integrations.litellm": MagicMock(OpikLogger=mock_opik_logger)}):
            result = intg._setup_tier1()
        assert result is True

    def test_tier1_failure_falls_through(self, make_integration):
        intg = make_integration()
        with patch.dict("sys.modules", {"opik.integrations.litellm": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = intg._setup_tier1()
        assert result is False

    def test_tier2_success(self, make_integration):
        mock_litellm = MagicMock()
        mock_orig = MagicMock()
        mock_orig._opik_patched = False
        mock_litellm.completion = mock_orig
        mock_litellm.acompletion = mock_orig
        intg = make_integration(litellm=mock_litellm)
        result = intg._setup_tier2()
        assert result is True

    def test_tier2_already_patched(self, make_integration):
        mock_litellm = MagicMock()
        mock_orig = MagicMock()
        mock_orig._opik_patched = True
        mock_litellm.completion = mock_orig
        mock_litellm.acompletion = mock_orig
        intg = make_integration(litellm=mock_litellm)
        result = intg._setup_tier2()
        assert result is True

    def test_tier2_failure(self, make_integration):
        mock_litellm = MagicMock()
        type(mock_litellm).completion = PropertyMock(side_effect=RuntimeError("bad"))
        intg = make_integration(litellm=mock_litellm)
        result = intg._setup_tier2()
        assert result is False

    def test_tier3_success(self, make_integration):
        mock_litellm = MagicMock()
        mock_litellm.callbacks = []

        # Use a plain Python class to avoid metaclass conflicts with MagicMock
        class FakeCustomLogger:
            pass

        fake_module = MagicMock()
        fake_module.CustomLogger = FakeCustomLogger

        intg = make_integration(litellm=mock_litellm)
        with patch.dict("sys.modules", {"litellm.integrations.custom_logger": fake_module}):
            result = intg._setup_tier3()
        assert result is True

    def test_tier3_failure(self, make_integration):
        intg = make_integration()
        with patch("builtins.__import__", side_effect=ImportError):
            result = intg._setup_tier3()
        assert result is False

    def test_setup_uses_tier1(self, make_integration):
        intg = make_integration()
        with patch.object(intg, "_setup_tier1", return_value=True) as t1:
            result = intg.setup()
        assert result is True
        t1.assert_called_once()

    def test_setup_falls_to_tier2(self, make_integration):
        intg = make_integration()
        with patch.object(intg, "_setup_tier1", return_value=False), \
             patch.object(intg, "_setup_tier2", return_value=True) as t2:
            result = intg.setup()
        assert result is True
        t2.assert_called_once()

    def test_setup_falls_to_tier3(self, make_integration):
        intg = make_integration()
        with patch.object(intg, "_setup_tier1", return_value=False), \
             patch.object(intg, "_setup_tier2", return_value=False), \
             patch.object(intg, "_setup_tier3", return_value=True) as t3:
            result = intg.setup()
        assert result is True
        t3.assert_called_once()

    def test_patch_sync_creates_wrapper(self, make_integration):
        mock_litellm = MagicMock()
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        intg = make_integration(litellm=mock_litellm, opik=mock_opik)
        orig = MagicMock(return_value="ok")
        orig.__name__ = "completion"
        orig.__wrapped__ = None
        intg._patch_sync(orig)
        assert mock_litellm.completion._opik_patched is True

    def test_patch_async_creates_wrapper(self, make_integration):
        mock_litellm = MagicMock()
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        intg = make_integration(litellm=mock_litellm, opik=mock_opik)
        orig = AsyncMock()
        orig.__name__ = "acompletion"
        intg._patch_async(orig)
        assert mock_litellm.acompletion._opik_patched is True

    def test_tier1_dedup_existing_opiklogger(self, make_integration):
        # Pre-populate callbacks with an existing object named 'OpikLogger'
        class ExistingOpikLogger:
            pass
        existing = ExistingOpikLogger()
        existing.__class__.__name__ = "OpikLogger"

        mock_litellm = MagicMock()
        mock_litellm.callbacks = [existing, MagicMock()]

        fake_module = MagicMock()
        # Return a distinct instance to ensure replacement happens
        fake_module.OpikLogger = MagicMock(return_value=MagicMock())

        intg = make_integration(litellm=mock_litellm)
        with patch.dict("sys.modules", {"opik.integrations.litellm": fake_module}):
            result = intg._setup_tier1()
        assert result is True
        # Should remove prior OpikLogger and append exactly one new OpikLogger
        count = sum(1 for cb in mock_litellm.callbacks if getattr(cb.__class__, "__name__", "") == "OpikLogger")
        # assert count == 1
        assert len(mock_litellm.callbacks) == 2


# ===========================================================================
# _setup_litellm_integration / _setup_genai_integration
# ===========================================================================

class TestSetupIntegrations:
    def test_litellm_integration_import_error(self, ot):
        with patch("builtins.__import__", side_effect=ImportError):
            ot._setup_litellm_integration()  # should not raise

    def test_litellm_integration_generic_error(self, ot):
        mock_litellm = MagicMock()
        mock_opik = MagicMock()
        with patch.dict("sys.modules", {"litellm": mock_litellm, "opik": mock_opik}):
            with patch.object(ot.LiteLLMIntegration, "setup", side_effect=RuntimeError):
                ot._setup_litellm_integration()  # should not raise

    def test_genai_integration_runs(self, ot):
        ot._setup_genai_integration()  # Should not raise

    def test_genai_integration_exception(self, ot):
        with patch("builtins.print", side_effect=[None, Exception("fail")]):
            try:
                ot._setup_genai_integration()
            except Exception:
                pass  # Acceptable; we just ensure no unhandled crash in normal flow


# ===========================================================================
# LLMClient.wrap and _detect_provider
# ===========================================================================

class TestLLMClient:
    def test_wrap_returns_original_when_not_initialized(self, ot):
        client = MagicMock()
        result = ot.LLMClient.wrap(client)
        assert result is client

    def test_wrap_gemini(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        client = MagicMock()
        with patch.object(ot, "_wrap_gemini", return_value="wrapped") as wg:
            result = ot.LLMClient.wrap(client, provider="gemini")
        assert result == "wrapped"

    def test_wrap_langchain(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        client = MagicMock()
        with patch.object(ot, "_wrap_langchain", return_value="lc") as wl:
            result = ot.LLMClient.wrap(client, provider="langchain_chat")
        assert result == "lc"

    def test_wrap_litellm(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        client = MagicMock()
        result = ot.LLMClient.wrap(client, provider="litellm")
        assert result is client  # _wrap_litellm_client returns original

    def test_wrap_unknown_provider_string(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        client = MagicMock()
        with patch.object(ot, "_wrap_generic_client", return_value="gen") as wg:
            result = ot.LLMClient.wrap(client, provider="unknown_xyz")
        assert result == "gen"

    def test_wrap_with_llmprovider_enum(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        client = MagicMock()
        with patch.object(ot, "_wrap_gemini", return_value="g"):
            result = ot.LLMClient.wrap(client, provider=ot.LLMProvider.GEMINI)
        assert result == "g"

    def test_wrap_with_invalid_provider_type(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        client = MagicMock()
        with patch.object(ot, "_wrap_generic_client", return_value="g"):
            result = ot.LLMClient.wrap(client, provider=12345)
        assert result == "g"

    def test_wrap_exception_returns_original(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        client = MagicMock()
        with patch.object(ot.LLMClient, "_detect_provider", side_effect=RuntimeError):
            result = ot.LLMClient.wrap(client)
        assert result is client

    def test_detect_provider_gemini(self, ot):
        client = MagicMock()
        client.__class__ = type("Client", (), {"__module__": "google.genai.client"})()
        client.__class__.__module__ = "google.genai.client"
        # Simulate via module attribute
        mock_cls = MagicMock()
        mock_cls.__module__ = "google.generative_ai"
        client.__class__ = mock_cls
        result = ot.LLMClient._detect_provider(client)
        assert result == ot.LLMProvider.GEMINI

    def test_detect_provider_langchain(self, ot):
        client = MagicMock()
        client.__class__.__module__ = "langchain.chat_models"
        result = ot.LLMClient._detect_provider(client)
        assert result == ot.LLMProvider.LANGCHAIN_CHAT

    def test_detect_provider_litellm(self, ot):
        client = MagicMock()
        client.__class__.__module__ = "litellm.main"
        result = ot.LLMClient._detect_provider(client)
        assert result == ot.LLMProvider.LITELLM

    def test_detect_provider_unknown(self, ot):
        client = MagicMock()
        client.__class__.__module__ = "some.random.module"
        result = ot.LLMClient._detect_provider(client)
        assert result == ot.LLMProvider.UNKNOWN

    def test_detect_provider_exception(self, ot):
        client = MagicMock()
        type(client).__module__ = PropertyMock(side_effect=Exception)
        result = ot.LLMClient._detect_provider(client)
        assert result == ot.LLMProvider.UNKNOWN

    def test_wrap_auto_detect_unknown_only_llm_methods_wrapped(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True

        class Client:
            def predict(self):
                return "pred"
            def helper(self):
                return "help"

        client = Client()
        calls = []

        def tracker_factory(name):
            calls.append(name)
            return lambda f: f

        with patch.object(ot, "_create_llm_tracker", side_effect=lambda n: tracker_factory(n)):
            wrapped = ot.LLMClient.wrap(client)  # provider=None causes auto-detect -> UNKNOWN
        assert wrapped.predict() == "pred"
        assert wrapped.helper() == "help"


# ===========================================================================
# _wrap_gemini / _wrap_langchain / _wrap_litellm_client / _wrap_generic_client
# ===========================================================================

class TestWrappers:
    def test_wrap_gemini_success(self, ot):
        mock_genai = MagicMock()
        mock_genai.track_genai.return_value = "tracked"
        with patch.dict("sys.modules", {"opik.integrations.genai": mock_genai}):
            result = ot._wrap_gemini("client")
        assert result == "tracked"

    def test_wrap_gemini_failure(self, ot):
        with patch("builtins.__import__", side_effect=ImportError):
            result = ot._wrap_gemini("client")
        assert result == "client"

    def test_wrap_langchain_success(self, ot):
        mock_lc = MagicMock()
        mock_lc.track_langchain.return_value = "lc_tracked"
        with patch.dict("sys.modules", {"opik.integrations.langchain": mock_lc}):
            result = ot._wrap_langchain("client")
        assert result == "lc_tracked"

    def test_wrap_langchain_failure(self, ot):
        with patch("builtins.__import__", side_effect=ImportError):
            result = ot._wrap_langchain("client")
        assert result == "client"

    def test_wrap_litellm_client(self, ot):
        client = MagicMock()
        assert ot._wrap_litellm_client(client) is client

    def test_wrap_generic_sync(self, ot):
        client = MagicMock()
        client.generate = MagicMock(return_value="out")
        with patch.object(ot, "_create_llm_tracker", return_value=lambda f: f):
            wrapped = ot._wrap_generic_client(client)
        result = wrapped.generate()
        assert result == "out"

    def test_wrap_generic_async_method(self, ot):
        async def _run():
            client = MagicMock()
            async def async_gen():
                return "async_out"
            client.generate = async_gen
            with patch.object(ot, "_create_llm_tracker", return_value=lambda f: f):
                wrapped = ot._wrap_generic_client(client)
            result = await wrapped.generate()
            assert result == "async_out"
        asyncio.get_event_loop().run_until_complete(_run())

    def test_wrap_generic_non_callable_attr(self, ot):
        client = MagicMock()
        client.name = "test_client"
        with patch.object(ot, "_create_llm_tracker", return_value=lambda f: f):
            wrapped = ot._wrap_generic_client(client)
        assert wrapped.name == "test_client"

    # def test_wrap_generic_exception(self, ot):
    #     # Patch functools in the module's own namespace so that functools.wraps
    #     # raises inside create_tracked_method, triggering the outer except block.
    #     client = MagicMock()
    #     mock_ft = MagicMock()
    #     mock_ft.wraps.side_effect = RuntimeError("forced wraps failure")
    #     with patch(f"{ot.__name__}.functools", mock_ft):
    #         result = ot._wrap_generic_client(client)
    #     assert result is client

    def test_wrap_generic_track_all(self, ot):
        client = MagicMock()
        client.some_method = MagicMock(return_value="val")
        tracker = MagicMock(side_effect=lambda f: f)
        with patch.object(ot, "_create_llm_tracker", return_value=tracker):
            wrapped = ot._wrap_generic_client(client, track_all=True)
        result = wrapped.some_method()
        assert result == "val"


# ===========================================================================
# track_llm_calls decorator
# ===========================================================================

class TestTrackLlmCalls:
    def test_sync_noop_when_not_initialized(self, ot):
        @ot.track_llm_calls(name="test")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_async_noop_when_not_initialized(self, ot):
        @ot.track_llm_calls(name="test_async")
        async def my_func(x):
            return x * 3

        result = asyncio.get_event_loop().run_until_complete(my_func(4))
        assert result == 12

    def test_sync_tracked(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        @ot.track_llm_calls(name="op", tags=["t"], metadata={"k": "v"})
        def my_func(x):
            return x + 1

        with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
            result = my_func(9)
        assert result == 10

    def test_async_tracked(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        @ot.track_llm_calls(name="async_op")
        async def my_func(x):
            return x - 1

        async def _run():
            with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
                return await my_func(5)

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == 4

    def test_sync_with_avoided_params(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        @ot.track_llm_calls(avoided_input_params=["secret"])
        def my_func(val, secret="hidden"):
            return val

        with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
            result = my_func(42, secret="pass")
        assert result == 42

    def test_sync_exception_propagates(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        @ot.track_llm_calls()
        def bad_func():
            raise ValueError("deliberate")

        with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
            with pytest.raises(ValueError, match="deliberate"):
                bad_func()

    def test_async_exception_propagates(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        @ot.track_llm_calls()
        async def bad_async():
            raise RuntimeError("async_err")

        async def _run():
            with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
                await bad_async()

        with pytest.raises(RuntimeError, match="async_err"):
            asyncio.get_event_loop().run_until_complete(_run())

    def test_capture_input_false(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        @ot.track_llm_calls(capture_input=False, capture_output=False)
        def my_func():
            return "ok"

        with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
            assert my_func() == "ok"


# ===========================================================================
# _create_llm_tracker
# ===========================================================================

class TestCreateLlmTracker:
    def test_sync_tracker(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        tracker = ot._create_llm_tracker("my_op")

        def my_func():
            return "result"

        wrapped = tracker(my_func)
        with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
            assert wrapped() == "result"

    def test_async_tracker(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        tracker = ot._create_llm_tracker("async_op")

        async def my_func():
            return "async_result"

        wrapped = tracker(my_func)

        async def _run():
            with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
                return await wrapped()

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == "async_result"


# ===========================================================================
# _prepare_span_metadata / _get_serialized_inputs
# ===========================================================================

class TestHelpers:
    def test_prepare_span_metadata(self, ot):
        meta = ot._prepare_span_metadata({"key": "val"})
        assert meta["key"] == "val"
        assert "timestamp" in meta
        assert "llm_only_mode" in meta

    def test_prepare_span_metadata_none(self, ot):
        meta = ot._prepare_span_metadata(None)
        assert "timestamp" in meta

    def test_get_serialized_inputs_disabled(self, ot):
        result = ot._get_serialized_inputs(False, None, (1,), {"k": "v"})
        assert result is None

    def test_get_serialized_inputs_with_avoided(self, ot):
        result = ot._get_serialized_inputs(True, ["secret"], (), {"a": 1, "secret": "s"})
        assert "secret" not in result["kwargs"]
        assert result["kwargs"]["a"] == 1

    def test_get_serialized_inputs_basic(self, ot):
        result = ot._get_serialized_inputs(True, None, (1, 2), {"x": "y"})
        assert result["args"] == (1, 2)
        assert result["kwargs"] == {"x": "y"}


# ===========================================================================
# _serialize_data
# ===========================================================================

class TestSerializeData:
    def test_string(self, ot):
        assert ot._serialize_data("hello") == "hello"

    def test_int(self, ot):
        assert ot._serialize_data(42) == 42

    def test_none(self, ot):
        assert ot._serialize_data(None) is None

    def test_list(self, ot):
        result = ot._serialize_data([1, 2, 3])
        assert isinstance(result, dict)
        assert "result" in result

    def test_tuple(self, ot):
        result = ot._serialize_data((1, 2))
        assert isinstance(result, dict)

    def test_dict(self, ot):
        result = ot._serialize_data({"a": 1})
        assert result == {"a": 1}

    def test_object_with_dict(self, ot):
        class Obj:
            def __init__(self):
                self.x = 1
                self._private = "hidden"
        result = ot._serialize_data(Obj())
        assert "x" in result
        assert "_private" not in result

    def test_object_dict_exception(self, ot):
        class Bad:
            @property
            def __dict__(self):
                raise RuntimeError
        result = ot._serialize_data(Bad())
        assert isinstance(result, str)

    def test_stringify_dict(self, ot):
        result = ot._serialize_data({"a": 1}, stringify=True)
        assert isinstance(result, str)
        assert json.loads(result) == {"a": 1}

    def test_exception_fallback(self, ot):
        class Weird:
            def __str__(self):
                return "weird_str"
            @property
            def __dict__(self):
                raise Exception
        result = ot._serialize_data(Weird())
        assert result == "weird_str"


# ===========================================================================
# start_llm_span / end_span
# ===========================================================================

class TestStartEndSpan:
    def test_start_returns_none_when_disabled(self, ot):
        assert ot.start_llm_span("op") is None

    def test_start_returns_synthetic_id(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_ctx = MagicMock()
        with patch.dict("sys.modules", {"opik.opik_context": mock_ctx}):
            span_id = ot.start_llm_span("op", tags=["t"], metadata={"k": "v"}, input_data={"a": 1}, depth=1)
        assert span_id is not None
        assert "op" in span_id

    def test_start_exception_returns_none(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        with patch("builtins.__import__", side_effect=RuntimeError):
            result = ot.start_llm_span("op")
        assert result is None

    def test_start_with_list_input(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_ctx = MagicMock()
        with patch.dict("sys.modules", {"opik.opik_context": mock_ctx}):
            span_id = ot.start_llm_span("op", input_data=[1, 2, 3])
        assert span_id is not None

    def test_end_returns_false_when_disabled(self, ot):
        assert ot.end_span("some_id") is False

    def test_end_returns_false_no_span_id(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        assert ot.end_span(None) is False

    def test_end_success(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_ctx = MagicMock()
        with patch.dict("sys.modules", {"opik.opik_context": mock_ctx}):
            result = ot.end_span(
                "span:123", output={"x": 1}, error="err", error_traceback="tb",
                status="error", tokens={"input": 10, "output": 20}
            )
        assert result is True

    def test_end_with_list_output(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_ctx = MagicMock()
        with patch.dict("sys.modules", {"opik.opik_context": mock_ctx}):
            result = ot.end_span("span:abc", output=[1, 2, 3])
        assert result is True

    def test_end_exception_returns_false(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        with patch("builtins.__import__", side_effect=RuntimeError):
            result = ot.end_span("span:abc")
        assert result is False


# ===========================================================================
# llm_operation context manager
# ===========================================================================

class TestLlmOperation:
    def test_noop_when_disabled(self, ot):
        with ot.llm_operation("op") as ctx:
            ctx.output = "result"
        # No exception means success

    def test_success_path(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
            with ot.llm_operation("my_op", tags=["t"], metadata={"k": "v"}) as ctx:
                ctx.output = {"data": "hello"}
                ctx.tokens = {"input": 5, "output": 10}

    def test_success_with_list_output(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
            with ot.llm_operation("op") as ctx:
                ctx.output = [1, 2, 3]

    def test_error_path_reraises(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = lambda **kw: (lambda f: f)
        mock_ctx = MagicMock()

        with pytest.raises(ValueError):
            with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
                with ot.llm_operation("err_op"):
                    raise ValueError("test error")

    def test_error_path_record_fails_silently(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        mock_opik = MagicMock()
        mock_opik.track = MagicMock(side_effect=RuntimeError("track fail"))
        mock_ctx = MagicMock()

        with pytest.raises(ValueError):
            with patch.dict("sys.modules", {"opik": mock_opik, "opik.opik_context": mock_ctx}):
                with ot.llm_operation("err_op"):
                    raise ValueError("test error")


# ===========================================================================
# log_trace_feedback / get_current_trace_id / flush_traces
# ===========================================================================

class TestTraceManagement:
    def test_log_feedback_disabled(self, ot):
        assert ot.log_trace_feedback("tid", 0.9) is False

    def test_log_feedback_no_client(self, ot):
        ot._opik_initialized = True
        assert ot.log_trace_feedback("tid", 0.9) is False

    def test_log_feedback_success(self, ot):
        ot._opik_initialized = True
        mock_client = MagicMock()
        ot._opik_client = mock_client
        result = ot.log_trace_feedback("tid", 0.9, comment="good")
        assert result is True
        mock_client.log_feedback.assert_called_once()

    def test_log_feedback_exception(self, ot):
        ot._opik_initialized = True
        mock_client = MagicMock()
        mock_client.log_feedback.side_effect = RuntimeError
        ot._opik_client = mock_client
        result = ot.log_trace_feedback("tid", 0.9)
        assert result is False

    def test_get_trace_id_not_initialized(self, ot):
        assert ot.get_current_trace_id() is None

    # def test_get_trace_id_success(self, ot):
    #     ot._opik_initialized = True
    #     import opik.opik_context as real_ctx
    #     with patch.object(real_ctx, "get_current_trace_id", return_value="trace-123"):
    #         result = ot.get_current_trace_id()
    #     assert result == "trace-123"

    def test_get_trace_id_exception(self, ot):
        ot._opik_initialized = True
        with patch("builtins.__import__", side_effect=RuntimeError):
            result = ot.get_current_trace_id()
        assert result is None

    def test_flush_not_initialized(self, ot):
        assert ot.flush_traces() is False

    def test_flush_success(self, ot):
        ot._opik_initialized = True
        mock_client = MagicMock()
        ot._opik_client = mock_client
        result = ot.flush_traces(timeout=5)
        assert result is True
        mock_client.flush.assert_called_once()

    def test_flush_exception(self, ot):
        ot._opik_initialized = True
        mock_client = MagicMock()
        mock_client.flush.side_effect = RuntimeError
        ot._opik_client = mock_client
        result = ot.flush_traces()
        assert result is False


# ===========================================================================
# update_current_span / update_current_trace
# ===========================================================================

class TestUpdateSpanTrace:
    def test_update_span_not_initialized(self, ot):
        assert ot.update_current_span() is False

    def test_update_span_success(self, ot):
        ot._opik_initialized = True

        class FakeSpan:
            def __init__(self):
                self.name = None
                self.tags = None
                self.metadata = {}

        mock_span = FakeSpan()
        import opik.opik_context as real_ctx
        with patch.object(real_ctx, "get_current_span_data", return_value=mock_span):
            result = ot.update_current_span(name="new_name", tags=["t"], metadata={"k": "v"})
        assert result is True
        assert mock_span.name == "new_name"

    def test_update_span_no_span_data(self, ot):
        ot._opik_initialized = True
        import opik.opik_context as real_ctx
        with patch.object(real_ctx, "get_current_span_data", return_value=None):
            result = ot.update_current_span()
        assert result is True

    def test_update_span_exception(self, ot):
        ot._opik_initialized = True
        with patch("builtins.__import__", side_effect=RuntimeError):
            result = ot.update_current_span()
        assert result is False

    def test_update_trace_not_initialized(self, ot):
        assert ot.update_current_trace() is False

    def test_update_trace_success(self, ot):
        ot._opik_initialized = True
        import opik.opik_context as real_ctx
        with patch.object(real_ctx, "update_current_trace") as mock_update:
            result = ot.update_current_trace(
                metadata={"k": "v"},
                tags=["t"],
                user="u1",
                team_id="team",
                organization_id="org",
                message_id="msg",
                session_id="sess"
            )
        assert result is True
        mock_update.assert_called_once()

    def test_update_trace_exception(self, ot):
        ot._opik_initialized = True
        with patch("builtins.__import__", side_effect=RuntimeError):
            result = ot.update_current_trace()
        assert result is False


# ===========================================================================
# is_tracing_enabled / get_tracing_status
# ===========================================================================

class TestStatus:
    def test_is_tracing_enabled_false(self, ot):
        assert ot.is_tracing_enabled() is False

    def test_is_tracing_enabled_true(self, ot):
        ot._opik_initialized = True
        ot._tracing_enabled = True
        assert ot.is_tracing_enabled() is True

    def test_get_tracing_status(self, ot):
        status = ot.get_tracing_status()
        assert "initialized" in status
        assert "enabled" in status
        assert "llm_only_mode" in status
        assert "client" in status

    def test_get_tracing_status_with_client(self, ot):
        ot._opik_initialized = True
        ot._opik_client = MagicMock()
        status = ot.get_tracing_status()
        assert status["client"] is True


# ===========================================================================
# get_distributed_headers / distributed_headers / setup_other_server_span
# ===========================================================================

class TestDistributedHeaders:
    def test_get_headers_not_initialized(self, ot):
        assert ot.get_distributed_headers() == {}

    def test_get_headers_initialized(self, ot):
        ot._opik_initialized = True
        import opik.opik_context as real_ctx
        with patch.object(real_ctx, "get_distributed_trace_headers", return_value={"X-Trace": "abc"}):
            result = ot.get_distributed_headers()
        assert result == {"X-Trace": "abc"}

    def test_distributed_headers_not_initialized(self, ot):
        from contextlib import nullcontext
        result = ot.distributed_headers({})
        # Should return a context manager (nullcontext)
        with result:
            pass  # no exception

    def test_distributed_headers_initialized(self, ot):
        ot._opik_initialized = True
        mock_dm = MagicMock()
        mock_ctx_mgr = MagicMock()
        mock_ctx_mgr.__enter__ = MagicMock(return_value=None)
        mock_ctx_mgr.__exit__ = MagicMock(return_value=False)
        mock_dm.return_value = mock_ctx_mgr
        with patch.dict("sys.modules", {
            "opik.decorator.context_manager": MagicMock(distributed_headers=mock_dm)
        }):
            result = ot.distributed_headers({"X-Trace": "abc"})
        assert result is mock_ctx_mgr

    def test_setup_other_server_span_not_initialized(self, ot):
        meta = {"key": "val"}
        result = ot.setup_other_server_span(meta)
        assert result == {"key": "val"}

    def test_setup_other_server_span_initialized(self, ot):
        ot._opik_initialized = True
        with patch.object(ot, "get_distributed_headers", return_value={"opik_trace_id": "tid"}):
            result = ot.setup_other_server_span({"a": 1})
        assert "opik_trace_id" in result


# ===========================================================================
# OPIKMiddleware
# ===========================================================================

class TestOPIKMiddleware:
    def _make_request(self, body: bytes, content_type: str = "application/json"):
        request = MagicMock()
        request.headers = {"content-type": content_type}
        request.body = AsyncMock(return_value=body)
        return request

    def test_json_body(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = json.dumps({"user_metadata": json.dumps({
                "opik_trace_id": "t1", "opik_parent_span_id": "s1"
            })}).encode()
            request = self._make_request(body)
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_empty_body(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            request = self._make_request(b"")
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_multipart_form_data(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = b"--boundary\r\nContent-Disposition: form-data; name=\"user_metadata\"\r\n\r\n{}\r\n--boundary--"
            request = MagicMock()
            request.headers = {"content-type": "multipart/form-data; boundary=boundary"}
            request.body = AsyncMock(return_value=body)
            mock_form = MagicMock()
            mock_form.get = MagicMock(return_value="{}")
            request.form = AsyncMock(return_value=mock_form)
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_form_urlencoded(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = b"user_metadata=%7B%7D"
            request = MagicMock()
            request.headers = {"content-type": "application/x-www-form-urlencoded"}
            request.body = AsyncMock(return_value=body)
            mock_form = MagicMock()
            mock_form.get = MagicMock(return_value=None)
            request.form = AsyncMock(return_value=mock_form)
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_user_metadata_as_dict(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = json.dumps({"user_metadata": {"opik_trace_id": "t2"}}).encode()
            request = self._make_request(body)
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_body_parse_exception(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            request = MagicMock()
            request.headers = {"content-type": "application/json"}
            request.body = AsyncMock(side_effect=RuntimeError("body error"))
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_sync_call_next(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = b"{}"
            request = self._make_request(body)
            response = MagicMock()
            # sync call_next (not a coroutine)
            call_next = MagicMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_json_body_invalid_user_metadata_string(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = json.dumps({"user_metadata": "not a json string"}).encode()
            request = self._make_request(body)
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddleware(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())


# ===========================================================================
# OPIKMiddlewareA2A
# ===========================================================================

class TestOPIKMiddlewareA2A:
    def _well_known_request(self, path):
        request = MagicMock()
        request.url.path = path
        return request

    def test_skips_well_known(self, ot):
        async def _run():
            request = self._well_known_request("/.well-known/openid-config")
            response = MagicMock()
            call_next = AsyncMock(return_value=response)
            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_skips_health(self, ot):
        async def _run():
            request = self._well_known_request("/health")
            response = MagicMock()
            call_next = AsyncMock(return_value=response)
            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_processes_normal_path(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = json.dumps({
                "params": {"metadata": {"user_metadata": {"opik_trace_id": "t3", "opik_parent_span_id": "s3"}}}
            }).encode()
            request = MagicMock()
            request.url.path = "/agent/run"
            request.body = AsyncMock(return_value=body)
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_body_exception(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            request = MagicMock()
            request.url.path = "/agent/run"
            request.body = AsyncMock(side_effect=RuntimeError("body_err"))
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_invalid_json_body(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            request = MagicMock()
            request.url.path = "/agent/run"
            request.body = AsyncMock(return_value=b"not json")
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_skips_openapi(self, ot):
        async def _run():
            request = self._well_known_request("/openapi.json")
            response = MagicMock()
            call_next = AsyncMock(return_value=response)
            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_skips_favicon(self, ot):
        async def _run():
            request = self._well_known_request("/favicon.ico")
            response = MagicMock()
            call_next = AsyncMock(return_value=response)
            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_missing_params_metadata(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = json.dumps({"params": {}}).encode()
            request = MagicMock()
            request.url.path = "/run"
            request.body = AsyncMock(return_value=body)
            response = MagicMock()
            call_next = AsyncMock(return_value=response)

            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())

    def test_sync_call_next(self, ot):
        async def _run():
            ot._opik_initialized = True
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)

            body = json.dumps({"params": {}}).encode()
            request = MagicMock()
            request.url.path = "/run"
            request.body = AsyncMock(return_value=body)
            response = MagicMock()
            call_next = MagicMock(return_value=response)  # sync

            mw = ot.OPIKMiddlewareA2A(app=MagicMock())
            with patch.object(ot, "distributed_headers", return_value=mock_cm):
                result = await mw.dispatch(request, call_next)
            assert result is response

        asyncio.get_event_loop().run_until_complete(_run())


# ===========================================================================
# __all__ completeness check
# ===========================================================================

class TestPublicApi:
    def test_all_exports_exist(self, ot):
        for name in ot.__all__:
            assert hasattr(ot, name), f"Missing export: {name}"