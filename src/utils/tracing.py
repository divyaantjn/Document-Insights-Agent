"""
OpenTelemetry Tracing Configuration Module for yash-unified-idp-backend.

This module provides tracing setup using OpenTelemetry with configurable
OTLP endpoints via environment variables.

CRITICAL: This module MUST be initialized BEFORE phoenix_setup.py to ensure
correct service.name is used for all traces.
"""
import os
import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

# Store our TracerProvider for use by other modules (like phoenix_setup.py)
_tracer_provider = None
_initialized = False


def get_tracer_provider():
    """Get the TracerProvider we created.

    Use this instead of trace.get_tracer_provider() to ensure you get OUR
    provider with the correct service.name.
    """
    global _tracer_provider
    if _tracer_provider is None:
        # Fall back to global if our setup hasn't run yet
        return trace.get_tracer_provider()
    return _tracer_provider


def setup_tracing(
    service_name: Optional[str] = None,
    service_environment: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    otlp_timeout: Optional[int] = None,
) -> bool:
    """
    Initialize OpenTelemetry tracing with configurable settings.

    Environment variables:
    - OTEL_SERVICE_NAME: Name of the service (default: "yash-unified-idp-backend")
    - OTEL_ENVIRONMENT: Deployment environment (default: "dev")
    - OTLP_ENDPOINT_OLD1: OTLP collector endpoint (required)
    - OTEL_EXPORTER_OTLP_ENDPOINT: Alternative OTLP endpoint variable
    - OTEL_EXPORTER_OTLP_TIMEOUT: Export timeout in seconds (default: 10)
    - OTEL_TRACING_ENABLED: Enable/disable tracing (default: "true")

    Returns:
        True if tracing was successfully initialized, False otherwise.
    """
    global _tracer_provider, _initialized

    if _initialized:
        logger.debug("Tracing already initialized, skipping")
        return True

    tracing_enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true"
    if not tracing_enabled:
        logger.info("OpenTelemetry tracing is disabled via OTEL_TRACING_ENABLED")
        return False

    _service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "yash-unified-idp-backend")
    _service_environment = service_environment or os.getenv("OTEL_ENVIRONMENT", "dev")
    _otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    _otlp_timeout = otlp_timeout or int(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "10"))

    if not _otlp_endpoint:
        logger.warning(
            "OTLP_ENDPOINT_OLD1 (or OTEL_EXPORTER_OTLP_ENDPOINT) is not set. "
            "OpenTelemetry tracing is disabled. "
            "Set this environment variable to enable distributed tracing."
        )
        return False

    try:
        resource = Resource.create({
            "service.name": _service_name,
            "service.environment": _service_environment,
            "service.type": "backend",
        })

        # Create OTLP exporter targeting your OTEL collector
        otlp_exporter = OTLPSpanExporter(
            endpoint=_otlp_endpoint,
            timeout=_otlp_timeout
        )

        # Detect if running in Lambda
        is_lambda = os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None

        if is_lambda and hasattr(otlp_exporter, '_session'):
            from requests.adapters import HTTPAdapter
            otlp_exporter._session.headers.update({"Connection": "close"})
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=5, max_retries=0)
            otlp_exporter._session.mount("http://", adapter)
            otlp_exporter._session.mount("https://", adapter)


        # SimpleSpanProcessor: exports each span synchronously inline
        # BatchSpanProcessor is incompatible with Lambda freeze/thaw — its background
        # thread freezes mid-export and fires through dead TCP connections on thaw
        span_processor = SimpleSpanProcessor(otlp_exporter)
        
        # Create our TracerProvider with the correct resource (service.name)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(span_processor)

        # Set as the global provider - overrides any existing provider
        trace.set_tracer_provider(tracer_provider)

        # Store reference for use by phoenix_setup.py and other modules
        _tracer_provider = tracer_provider
        _initialized = True

        # Instrument HTTP clients with our TracerProvider
        _instrument_http_clients(tracer_provider, _otlp_endpoint)

        logger.info(
            f"✅ OpenTelemetry tracing initialized: service={_service_name}, "
            f"environment={_service_environment}, endpoint={_otlp_endpoint}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        return False


def _instrument_http_clients(tracer_provider,_otlp_endpoint):
    """Instrument HTTP client libraries."""
    # Instrument HTTPX (used for HTTP calls)
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.debug("HTTPX instrumented for tracing")
    except Exception as e:
        logger.debug(f"HTTPX instrumentation skipped: {e}")

    # Instrument requests (fallback HTTP library)
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        # Exclude OTLP endpoint to prevent self-instrumentation loop
        RequestsInstrumentor().instrument(
            tracer_provider=tracer_provider,
            excluded_urls=_otlp_endpoint,
        )
    except Exception as e:
        logger.debug(f"Requests instrumentation skipped: {e}")


def instrument_fastapi_app(app):
    """Instrument a FastAPI application for tracing.

    Uses FastAPIInstrumentor to add automatic span creation for all
    HTTP requests handled by the FastAPI app.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        tracer_provider = get_tracer_provider()

        if tracer_provider and hasattr(tracer_provider, 'resource'):
            FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)

            resource = tracer_provider.resource
            service_name = resource.attributes.get("service.name", "unknown")
            logger.info(f"✅ FastAPI app instrumented for tracing (service.name={service_name})")
        else:
            logger.debug("Tracing not initialized, skipping FastAPI instrumentation")
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI app: {e}")


def get_tracer(name: str = __name__):
    """Get a tracer instance for creating custom spans."""
    return trace.get_tracer(name)

def refresh_otlp_session():
    """Call at the START of every Lambda invocation to close stale TCP connections."""
    if not _tracer_provider or not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return
    try:
        processor = _tracer_provider._active_span_processor
        # Collect all candidate processors — handles both:
        # - SimpleSpanProcessor set directly as _active_span_processor
        # - SynchronousMultiSpanProcessor wrapping multiple processors (added by Phoenix instrumentors)
        candidates = getattr(processor, '_span_processors', None)
        if candidates is None:
            candidates = [processor]
        refreshed = False
        for p in candidates:
            exporter = getattr(p, 'span_exporter', None)
            if exporter and hasattr(exporter, '_session'):
                exporter._session.close()
                refreshed = True
        if refreshed:
            logger.debug("OTLP session refreshed")
        else:
            logger.warning(f"refresh_otlp_session: exporter not found, processor type={type(processor)}")
    except Exception as e:
        logger.warning(f"refresh_otlp_session failed: {e}")


def flush_traces(timeout_millis: int = 5000) -> bool:
    """
    Force flush all pending spans to the OTLP endpoint.

    CRITICAL FOR LAMBDA: BatchSpanProcessor queues spans in memory and exports
    them periodically. In Lambda, the function may freeze before spans are flushed.
    Call this function before returning from the Lambda handler to ensure traces
    are exported.

    Args:
        timeout_millis: Maximum time in milliseconds to wait for flush (default: 5000)

    Returns:
        True if flush was successful, False otherwise
    """
    try:
        tracer_provider = get_tracer_provider()

        if tracer_provider and hasattr(tracer_provider, 'force_flush'):
            success = tracer_provider.force_flush(timeout_millis)
            if success:
                logger.debug("✅ Traces flushed successfully to OTLP endpoint")
            else:
                logger.warning("Trace flush timed out - some traces may not be exported")
            return success
        else:
            logger.warning("TracerProvider does not support force_flush")
            return False

    except Exception as e:
        logger.error(f"Failed to flush traces: {e}")
        return False
