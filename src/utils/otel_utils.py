"""
otel_utils.py - OpenTelemetry span annotation utilities for yash-unified-idp-backend.

This module provides utilities for setting span attributes that are
searchable in Grafana/Tempo. It also includes X-Ray annotation helpers
for when running in Lambda with X-Ray enabled.
"""

import logging
import os
from opentelemetry import trace

logger = logging.getLogger(__name__)


def set_span_attribute(key: str, value):
    """
    Set an attribute on the current OpenTelemetry span.

    Args:
        key: The attribute key
        value: The attribute value (str, int, float, bool, or list of these)
    """
    if value is None:
        return

    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            # Ensure value is a supported type
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            elif isinstance(value, list):
                # OTEL supports list of homogeneous primitive types
                span.set_attribute(key, value)
            else:
                # Convert to string for unsupported types
                span.set_attribute(key, str(value)[:500])

            logger.debug(f"✅ Span attribute set: {key}={value}")
    except Exception as e:
        logger.warning(f"Failed to set span attribute {key}: {e}")


def set_xray_annotation(key: str, value):
    """
    Set an annotation on the current X-Ray segment/subsegment.
    Only works when running in Lambda with X-Ray enabled.

    Args:
        key: The annotation key
        value: The annotation value (str, int, float, bool)
    """
    if value is None:
        return

    try:
        from aws_xray_sdk.core import xray_recorder
        segment = xray_recorder.current_segment()
        if segment:
            segment.put_annotation(key, value)
            logger.debug(f"✅ X-Ray annotation set: {key}={value}")
    except Exception as e:
        logger.debug(f"X-Ray annotation skipped: {e}")


def set_xray_metadata(namespace: str, key: str, value):
    """
    Set metadata on the current X-Ray segment/subsegment.

    Args:
        namespace: The metadata namespace
        key: The metadata key
        value: The metadata value (any JSON-serializable type)
    """
    if value is None:
        return

    try:
        from aws_xray_sdk.core import xray_recorder
        segment = xray_recorder.current_segment()
        if segment:
            segment.put_metadata(key, value, namespace)
            logger.debug(f"✅ X-Ray metadata set: {namespace}/{key}")
    except Exception as e:
        logger.debug(f"X-Ray metadata skipped: {e}")


def set_user_context(user_id: str = None, user_email: str = None, auth_mode: str = None):
    """
    Set user context on the current span.

    Args:
        user_id: The user's ID (from JWT sub claim)
        user_email: The user's email
        auth_mode: The authentication mode (e.g., 'keycloak')
    """
    if user_id:
        set_span_attribute("user.id", user_id)
    if user_email:
        set_span_attribute("user.email", user_email)
    if auth_mode:
        set_span_attribute("auth.mode", auth_mode)


def set_error_context(error_type: str, error_message: str, is_critical: bool = False):
    """
    Set error context on the current span.

    Args:
        error_type: Type of error (e.g., 'ValidationError', 'AuthenticationError')
        error_message: Error message (truncated to 500 chars)
        is_critical: Whether this is a critical error
    """
    from opentelemetry.trace.status import Status, StatusCode

    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.type", error_type)
            span.set_attribute("error.message", str(error_message)[:500])
            span.set_attribute("error.critical", is_critical)

            if is_critical:
                span.set_status(Status(StatusCode.ERROR, str(error_message)[:128]))

            logger.debug(f"✅ Error context set: {error_type}")
    except Exception as e:
        logger.warning(f"Failed to set error context: {e}")


def set_message_id(message_id: str):
    """
    Set message ID on the current span for tracing correlation.

    This is used to correlate traces with frontend message IDs passed
    via user_metadata. The message_id allows you to trace a user's request
    from the frontend all the way through backend processing.

    Args:
        message_id: The message ID from user_metadata (usually from frontend)
    """
    if message_id:
        set_span_attribute("message.id", message_id)
        set_span_attribute("message_id", message_id)  # Also set without dot for compatibility


def force_user_context_to_xray(user_id: str = None, user_email: str = None,
                                username: str = None, realm: str = None):
    """
    Force user context to X-Ray annotations (for searchability in X-Ray console).

    Args:
        user_id: The user's ID
        user_email: The user's email
        username: The user's username
        realm: The authentication realm
    """
    if user_id:
        set_xray_annotation("user_id", str(user_id))
    if user_email:
        set_xray_annotation("user_email", user_email)
    if username:
        set_xray_annotation("username", username)
    if realm:
        set_xray_annotation("realm", realm)
