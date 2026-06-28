"""
error_capture.py - Error capture utilities for yash-unified-idp-backend tracing.

This module provides utilities for capturing errors into both X-Ray annotations
and OpenTelemetry span attributes for searchability in Grafana/Tempo and X-Ray.
"""

import logging
from typing import Dict, Any, List
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)


def capture_http_error_details(status_code: int, error_details: Dict[str, Any]):
    """
    Capture HTTP error details on the current span and X-Ray segment.

    Args:
        status_code: The HTTP status code
        error_details: Dictionary containing error information
    """
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.status_code", status_code)
            span.set_attribute("error.type", "http_error")

            message = error_details.get("message", "Unknown error")
            span.set_attribute("error.message", str(message)[:500])

            if status_code >= 500:
                span.set_status(Status(StatusCode.ERROR, str(message)[:128]))

            logger.debug(f"✅ HTTP error captured: {status_code}")
    except Exception as e:
        logger.warning(f"Failed to capture HTTP error: {e}")

    # Also set X-Ray annotations
    try:
        from aws_xray_sdk.core import xray_recorder
        segment = xray_recorder.current_segment()
        if segment:
            segment.put_annotation("error", True)
            segment.put_annotation("error_status_code", status_code)
            segment.put_annotation("error_type", "http_error")
    except Exception:
        pass


def capture_validation_error(validation_errors: List[Dict[str, Any]]):
    """
    Capture validation errors on the current span.

    Args:
        validation_errors: List of validation error details (from RequestValidationError)
    """
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.type", "validation_error")
            span.set_attribute("validation.error_count", len(validation_errors))

            # Extract first few error messages
            error_messages = []
            for err in validation_errors[:3]:
                if isinstance(err, dict):
                    msg = err.get("msg", str(err))[:100]
                else:
                    msg = str(err)[:100]
                error_messages.append(msg)

            if error_messages:
                span.set_attribute("validation.messages", ", ".join(error_messages))

            logger.debug(f"✅ Validation error captured: {len(validation_errors)} errors")
    except Exception as e:
        logger.warning(f"Failed to capture validation error: {e}")

    # Also set X-Ray annotations
    try:
        from aws_xray_sdk.core import xray_recorder
        from src.utils.otel_utils import set_xray_annotation
        set_xray_annotation("validation_error", True)
        set_xray_annotation("error_type", "validation_error")
    except Exception:
        pass


def capture_external_api_error(service_name: str, error: Exception,
                                endpoint: str = None, status_code: int = None):
    """
    Capture external API call errors on the current span.

    Args:
        service_name: Name of the external service that failed
        error: The exception that occurred
        endpoint: The endpoint that was called
        status_code: The HTTP status code (if applicable)
    """
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.type", "external_api_error")
            span.set_attribute("error.external_service", service_name)
            span.set_attribute("error.message", str(error)[:500])

            if endpoint:
                span.set_attribute("error.endpoint", endpoint)
            if status_code:
                span.set_attribute("error.external_status_code", status_code)

            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, f"External API error: {service_name}"))

            logger.debug(f"✅ External API error captured for {service_name}")
    except Exception as e:
        logger.warning(f"Failed to capture external API error: {e}")


def capture_processing_error(document_name: str, error: Exception, stage: str = None):
    """
    Capture document processing errors (OCR, NER, etc.) on the current span.

    Args:
        document_name: The document being processed (truncated)
        error: The exception that occurred
        stage: The processing stage where the error occurred (e.g., 'ocr', 'ner')
    """
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.type", "processing_error")
            span.set_attribute("error.message", str(error)[:500])
            span.set_attribute("document.name", str(document_name)[:200])

            if stage:
                span.set_attribute("error.stage", stage)
                span.set_attribute("processing.stage", stage)

            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, f"Processing error at {stage}: {type(error).__name__}"))

            logger.debug(f"✅ Processing error captured at stage={stage}")
    except Exception as e:
        logger.warning(f"Failed to capture processing error: {e}")
