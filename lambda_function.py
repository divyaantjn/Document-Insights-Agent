import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lambda-safe writable paths
if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    os.environ.setdefault("HOME", "/tmp")

# =============================================================================
# TRACING INITIALIZATION - MUST happen before importing the FastAPI app
# =============================================================================

# Initialize OpenTelemetry tracing FIRST!
# This MUST run before setup_phoenix() so our TracerProvider with correct
# service.name is set before Phoenix instruments LLM libraries
from src.utils.tracing import setup_tracing, flush_traces, refresh_otlp_session
setup_tracing()

# =============================================================================
# APPLICATION IMPORT
# =============================================================================

from mangum import Mangum

# Import the FastAPI app from main.py
# main.py's module-level code runs here, but setup_tracing() above ensures
# our TracerProvider is already initialized before main.py's setup_tracing() call
from main import app

# Create Mangum handler
_mangum_handler = Mangum(app)


def handler(event, context):
    """
    Lambda handler that wraps Mangum with trace flushing.

    CRITICAL: BatchSpanProcessor queues spans in memory. Without flushing,
    Lambda may freeze before spans are exported to the collector.
    """
    refresh_otlp_session()  # close stale TCP connections from previous frozen container
    try:
        # Process the request
        response = _mangum_handler(event, context)
        return response
    finally:
        # ALWAYS flush traces before Lambda freezes
        flush_traces(timeout_millis=5000)
