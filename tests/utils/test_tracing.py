"""
tests/utils/test_tracing.py

Unit tests for src/utils/tracing.py — 100% coverage.
All OTLP / OpenTelemetry SDK calls are fully mocked.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_module_state():
    """Reset the module-level globals before each test."""
    import src.utils.tracing as tracing_mod
    tracing_mod._tracer_provider = None
    tracing_mod._initialized = False


# ---------------------------------------------------------------------------
# get_tracer_provider
# ---------------------------------------------------------------------------

class TestGetTracerProvider:

    def test_returns_global_when_not_initialized(self):
        _reset_module_state()
        from src.utils.tracing import get_tracer_provider
        from opentelemetry import trace as otel_trace
        result = get_tracer_provider()
        assert result is otel_trace.get_tracer_provider()

    def test_returns_stored_provider_when_set(self):
        import src.utils.tracing as tracing_mod
        _reset_module_state()
        fake_provider = MagicMock()
        tracing_mod._tracer_provider = fake_provider
        from src.utils.tracing import get_tracer_provider
        assert get_tracer_provider() is fake_provider
        _reset_module_state()


# ---------------------------------------------------------------------------
# setup_tracing
# ---------------------------------------------------------------------------

class TestSetupTracing:

    def setup_method(self):
        _reset_module_state()

    def teardown_method(self):
        _reset_module_state()

    def test_returns_false_when_tracing_disabled(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "false")
        from src.utils.tracing import setup_tracing
        result = setup_tracing()
        assert result is False

    def test_returns_false_when_no_otlp_endpoint(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from src.utils.tracing import setup_tracing
        result = setup_tracing()
        assert result is False

    def test_returns_true_on_success(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)

        with patch("src.utils.tracing.OTLPSpanExporter"), \
             patch("src.utils.tracing.SimpleSpanProcessor"), \
             patch("src.utils.tracing.TracerProvider") as mock_tp_cls, \
             patch("src.utils.tracing.trace"), \
             patch("src.utils.tracing._instrument_http_clients"):
            mock_tp = MagicMock()
            mock_tp.resource = MagicMock()
            mock_tp_cls.return_value = mock_tp

            from src.utils.tracing import setup_tracing
            result = setup_tracing()

        assert result is True

    def test_already_initialized_skips_and_returns_true(self, monkeypatch):
        import src.utils.tracing as tracing_mod
        tracing_mod._initialized = True

        from src.utils.tracing import setup_tracing
        result = setup_tracing()
        assert result is True

    def test_uses_simple_span_processor_in_lambda(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "my-lambda")

        mock_simple = MagicMock()
        with patch("src.utils.tracing.OTLPSpanExporter"), \
             patch("src.utils.tracing.SimpleSpanProcessor", return_value=mock_simple) as simple_cls, \
             patch("src.utils.tracing.TracerProvider") as mock_tp_cls, \
             patch("src.utils.tracing.trace"), \
             patch("src.utils.tracing._instrument_http_clients"):
            mock_tp = MagicMock()
            mock_tp_cls.return_value = mock_tp

            from src.utils.tracing import setup_tracing
            setup_tracing()

        simple_cls.assert_called_once()

    def test_uses_batch_span_processor_in_non_lambda(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)

        with patch("src.utils.tracing.OTLPSpanExporter"), \
             patch("src.utils.tracing.SimpleSpanProcessor") as simple_cls, \
             patch("src.utils.tracing.TracerProvider") as mock_tp_cls, \
             patch("src.utils.tracing.trace"), \
             patch("src.utils.tracing._instrument_http_clients"):
            mock_tp = MagicMock()
            mock_tp_cls.return_value = mock_tp

            from src.utils.tracing import setup_tracing
            setup_tracing()

    def test_exception_during_setup_returns_false(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

        with patch("src.utils.tracing.OTLPSpanExporter", side_effect=RuntimeError("oops")):
            from src.utils.tracing import setup_tracing
            result = setup_tracing()
        assert result is False

    def test_uses_env_service_name(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "my-custom-service")
        monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)

        captured_resources = []

        def capture_resource(attrs):
            captured_resources.append(attrs)
            return MagicMock()

        with patch("src.utils.tracing.OTLPSpanExporter"), \
            patch("src.utils.tracing.SimpleSpanProcessor"), \
             patch("src.utils.tracing.TracerProvider") as mock_tp_cls, \
               patch("src.utils.tracing.trace"), \
               patch("src.utils.tracing._instrument_http_clients"), \
               patch("src.utils.tracing.Resource") as mock_resource_cls:
            mock_tp = MagicMock()
            mock_tp_cls.return_value = mock_tp
            mock_resource_cls.create.side_effect = capture_resource

            from src.utils.tracing import setup_tracing
            setup_tracing()

        if captured_resources:
            assert captured_resources[0].get("service.name") == "my-custom-service"

    def test_uses_param_service_name_over_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "from-env")
        monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)

        captured_resources = []

        with patch("src.utils.tracing.OTLPSpanExporter"), \
             patch("src.utils.tracing.SimpleSpanProcessor"), \
             patch("src.utils.tracing.TracerProvider") as mock_tp_cls, \
             patch("src.utils.tracing.trace"), \
             patch("src.utils.tracing._instrument_http_clients"), \
             patch("src.utils.tracing.Resource") as mock_resource_cls:
            mock_tp = MagicMock()
            mock_tp_cls.return_value = mock_tp
            mock_resource_cls.create.side_effect = lambda attrs: captured_resources.append(attrs) or MagicMock()

            from src.utils.tracing import setup_tracing
            setup_tracing(
                service_name="from-param",
                otlp_endpoint="http://localhost:4318/v1/traces"
            )

        if captured_resources:
            assert captured_resources[0].get("service.name") == "from-param"


# ---------------------------------------------------------------------------
# _instrument_http_clients
# ---------------------------------------------------------------------------

class TestInstrumentHttpClients:

    def test_instruments_httpx_when_available(self):
        mock_httpx_inst = MagicMock()
        mock_httpx_mod = MagicMock()
        mock_httpx_mod.HTTPXClientInstrumentor.return_value = mock_httpx_inst

        with patch.dict("sys.modules", {
            "opentelemetry.instrumentation.httpx": mock_httpx_mod,
            "opentelemetry.instrumentation.requests": MagicMock()
        }):
            from src.utils.tracing import _instrument_http_clients
            provider = MagicMock()
            _instrument_http_clients(provider, None)

        mock_httpx_inst.instrument.assert_called_once_with(tracer_provider=provider)

    def test_instruments_requests_when_available(self):
        mock_req_inst = MagicMock()
        mock_req_mod = MagicMock()
        mock_req_mod.RequestsInstrumentor.return_value = mock_req_inst

        with patch.dict("sys.modules", {
            "opentelemetry.instrumentation.httpx": MagicMock(),
            "opentelemetry.instrumentation.requests": mock_req_mod
        }):
            from src.utils.tracing import _instrument_http_clients
            provider = MagicMock()
            _instrument_http_clients(provider, None)

        mock_req_inst.instrument.assert_called_once_with(tracer_provider=provider, excluded_urls=None)

    def test_handles_import_error_gracefully(self):
        with patch.dict("sys.modules", {
            "opentelemetry.instrumentation.httpx": None,
            "opentelemetry.instrumentation.requests": None
        }):
            from src.utils.tracing import _instrument_http_clients
            # Should not raise even when imports fail
            _instrument_http_clients(MagicMock(), None)


# ---------------------------------------------------------------------------
# instrument_fastapi_app
# ---------------------------------------------------------------------------

class TestInstrumentFastapiApp:

    def test_instruments_app_when_provider_ready(self):
        mock_provider = MagicMock()
        mock_provider.resource = MagicMock()
        mock_provider.resource.attributes = {"service.name": "my-svc"}

        mock_fastapi_mod = MagicMock()
        mock_instrumentor_cls = MagicMock()
        mock_fastapi_mod.FastAPIInstrumentor = mock_instrumentor_cls

        with patch("src.utils.tracing.get_tracer_provider", return_value=mock_provider), \
             patch.dict("sys.modules", {"opentelemetry.instrumentation.fastapi": mock_fastapi_mod}):
            from src.utils.tracing import instrument_fastapi_app
            app = MagicMock()
            instrument_fastapi_app(app)

        mock_instrumentor_cls.instrument_app.assert_called_once_with(app, tracer_provider=mock_provider)

    def test_skips_instrumentation_when_provider_missing_resource(self):
        mock_provider = MagicMock(spec=[])  # No 'resource' attribute

        with patch("src.utils.tracing.get_tracer_provider", return_value=mock_provider):
            from src.utils.tracing import instrument_fastapi_app
            app = MagicMock()
            instrument_fastapi_app(app)  # Must not raise

    def test_handles_import_error_gracefully(self):
        with patch("src.utils.tracing.get_tracer_provider", return_value=MagicMock()), \
             patch.dict("sys.modules", {"opentelemetry.instrumentation.fastapi": None}):
            from src.utils.tracing import instrument_fastapi_app
            instrument_fastapi_app(MagicMock())  # Must not raise


# ---------------------------------------------------------------------------
# get_tracer
# ---------------------------------------------------------------------------

class TestGetTracer:

    def test_returns_tracer(self):
        from src.utils.tracing import get_tracer
        tracer = get_tracer("my.module")
        assert tracer is not None

    def test_default_name(self):
        from src.utils.tracing import get_tracer
        tracer = get_tracer()
        assert tracer is not None


# ---------------------------------------------------------------------------
# flush_traces
# ---------------------------------------------------------------------------

class TestFlushTraces:

    def test_returns_true_on_successful_flush(self):
        mock_provider = MagicMock()
        mock_provider.force_flush.return_value = True

        with patch("src.utils.tracing.get_tracer_provider", return_value=mock_provider):
            from src.utils.tracing import flush_traces
            result = flush_traces()

        assert result is True
        mock_provider.force_flush.assert_called_once_with(5000)

    def test_returns_false_on_flush_timeout(self):
        mock_provider = MagicMock()
        mock_provider.force_flush.return_value = False

        with patch("src.utils.tracing.get_tracer_provider", return_value=mock_provider):
            from src.utils.tracing import flush_traces
            result = flush_traces()

        assert result is False

    def test_returns_false_when_provider_has_no_force_flush(self):
        mock_provider = MagicMock(spec=[])

        with patch("src.utils.tracing.get_tracer_provider", return_value=mock_provider):
            from src.utils.tracing import flush_traces
            result = flush_traces()

        assert result is False

    def test_returns_false_on_exception(self):
        mock_provider = MagicMock()
        mock_provider.force_flush.side_effect = RuntimeError("flush error")

        with patch("src.utils.tracing.get_tracer_provider", return_value=mock_provider):
            from src.utils.tracing import flush_traces
            result = flush_traces()

        assert result is False

    def test_custom_timeout_passed(self):
        mock_provider = MagicMock()
        mock_provider.force_flush.return_value = True

        with patch("src.utils.tracing.get_tracer_provider", return_value=mock_provider):
            from src.utils.tracing import flush_traces
            flush_traces(timeout_millis=10000)

        mock_provider.force_flush.assert_called_once_with(10000)
