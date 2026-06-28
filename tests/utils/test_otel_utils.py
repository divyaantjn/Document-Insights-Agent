"""
tests/utils/test_otel_utils.py

Unit tests for src/utils/otel_utils.py — 100% coverage.
"""

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# set_span_attribute
# ---------------------------------------------------------------------------

class TestSetSpanAttribute:

    def _run(self, key, value, span=None):
        """Helper to call set_span_attribute with a mocked current span."""
        from src.utils.otel_utils import set_span_attribute
        if span is None:
            span = MagicMock()
            span.is_recording.return_value = True

        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_span_attribute(key, value)
        return span

    def test_none_value_returns_early(self):
        from src.utils.otel_utils import set_span_attribute
        span = MagicMock()
        span.is_recording.return_value = True
        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_span_attribute("key", None)
        span.set_attribute.assert_not_called()

    def test_string_value(self):
        span = self._run("mykey", "myvalue")
        span.set_attribute.assert_called_once_with("mykey", "myvalue")

    def test_int_value(self):
        span = self._run("count", 42)
        span.set_attribute.assert_called_once_with("count", 42)

    def test_float_value(self):
        span = self._run("score", 0.95)
        span.set_attribute.assert_called_once_with("score", 0.95)

    def test_bool_value(self):
        span = self._run("flag", True)
        span.set_attribute.assert_called_once_with("flag", True)

    def test_list_value(self):
        span = self._run("items", [1, 2, 3])
        span.set_attribute.assert_called_once_with("items", [1, 2, 3])

    def test_unsupported_type_converts_to_string(self):
        span = self._run("obj", {"nested": "dict"})
        call_args = span.set_attribute.call_args[0]
        assert isinstance(call_args[1], str)

    def test_span_not_recording_skips_set(self):
        from src.utils.otel_utils import set_span_attribute
        span = MagicMock()
        span.is_recording.return_value = False
        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_span_attribute("k", "v")
        span.set_attribute.assert_not_called()

    def test_exception_during_set_attribute_is_swallowed(self):
        from src.utils.otel_utils import set_span_attribute
        span = MagicMock()
        span.is_recording.return_value = True
        span.set_attribute.side_effect = RuntimeError("span error")
        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_span_attribute("k", "v")  # Must not raise

    def test_no_span_does_not_raise(self):
        from src.utils.otel_utils import set_span_attribute
        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = None
            set_span_attribute("k", "v")  # Must not raise


# ---------------------------------------------------------------------------
# set_xray_annotation
# ---------------------------------------------------------------------------

class TestSetXrayAnnotation:

    def test_none_value_returns_early(self):
        from src.utils.otel_utils import set_xray_annotation
        set_xray_annotation("key", None)  # Must not raise

    def test_sets_annotation_on_segment(self):
        from src.utils.otel_utils import set_xray_annotation
        mock_segment = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.current_segment.return_value = mock_segment

        with patch.dict("sys.modules", {"aws_xray_sdk.core": MagicMock(xray_recorder=mock_recorder)}):
            set_xray_annotation("user_id", "u-123")

        mock_segment.put_annotation.assert_called_once_with("user_id", "u-123")

    def test_no_segment_does_not_raise(self):
        from src.utils.otel_utils import set_xray_annotation
        mock_recorder = MagicMock()
        mock_recorder.current_segment.return_value = None

        with patch.dict("sys.modules", {"aws_xray_sdk.core": MagicMock(xray_recorder=mock_recorder)}):
            set_xray_annotation("key", "val")  # Must not raise

    def test_import_error_is_swallowed(self):
        from src.utils.otel_utils import set_xray_annotation
        with patch.dict("sys.modules", {"aws_xray_sdk": None, "aws_xray_sdk.core": None}):
            set_xray_annotation("k", "v")  # Must not raise


# ---------------------------------------------------------------------------
# set_xray_metadata
# ---------------------------------------------------------------------------

class TestSetXrayMetadata:

    def test_none_value_returns_early(self):
        from src.utils.otel_utils import set_xray_metadata
        set_xray_metadata("ns", "key", None)  # Must not raise

    def test_sets_metadata_on_segment(self):
        from src.utils.otel_utils import set_xray_metadata
        mock_segment = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.current_segment.return_value = mock_segment

        with patch.dict("sys.modules", {"aws_xray_sdk.core": MagicMock(xray_recorder=mock_recorder)}):
            set_xray_metadata("app", "payload", {"data": 1})

        mock_segment.put_metadata.assert_called_once_with("payload", {"data": 1}, "app")

    def test_no_segment_does_not_raise(self):
        from src.utils.otel_utils import set_xray_metadata
        mock_recorder = MagicMock()
        mock_recorder.current_segment.return_value = None

        with patch.dict("sys.modules", {"aws_xray_sdk.core": MagicMock(xray_recorder=mock_recorder)}):
            set_xray_metadata("ns", "k", "v")

    def test_import_error_is_swallowed(self):
        from src.utils.otel_utils import set_xray_metadata
        with patch.dict("sys.modules", {"aws_xray_sdk": None, "aws_xray_sdk.core": None}):
            set_xray_metadata("ns", "k", "v")


# ---------------------------------------------------------------------------
# set_user_context
# ---------------------------------------------------------------------------

class TestSetUserContext:

    def _call(self, user_id=None, user_email=None, auth_mode=None):
        from src.utils.otel_utils import set_user_context
        with patch("src.utils.otel_utils.set_span_attribute") as mock_set:
            set_user_context(user_id=user_id, user_email=user_email, auth_mode=auth_mode)
            return mock_set.call_args_list

    def test_all_params_set(self):
        calls = self._call(user_id="u1", user_email="u@x.com", auth_mode="keycloak")
        keys = [c[0][0] for c in calls]
        assert "user.id" in keys
        assert "user.email" in keys
        assert "auth.mode" in keys

    def test_no_params_no_calls(self):
        calls = self._call()
        assert calls == []

    def test_only_user_id(self):
        calls = self._call(user_id="u1")
        assert any(c[0][0] == "user.id" for c in calls)
        assert not any(c[0][0] == "user.email" for c in calls)

    def test_only_email(self):
        calls = self._call(user_email="e@x.com")
        assert any(c[0][0] == "user.email" for c in calls)

    def test_only_auth_mode(self):
        calls = self._call(auth_mode="oauth2")
        assert any(c[0][0] == "auth.mode" for c in calls)


# ---------------------------------------------------------------------------
# set_error_context
# ---------------------------------------------------------------------------

class TestSetErrorContext:

    def test_sets_error_attributes(self):
        from src.utils.otel_utils import set_error_context
        span = MagicMock()
        span.is_recording.return_value = True

        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_error_context("ValidationError", "field required")

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert calls.get("error") is True
        assert calls.get("error.type") == "ValidationError"
        assert "field required" in calls.get("error.message", "")

    def test_critical_sets_status(self):
        from src.utils.otel_utils import set_error_context
        span = MagicMock()
        span.is_recording.return_value = True

        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_error_context("CriticalError", "boom", is_critical=True)

        span.set_status.assert_called_once()

    def test_non_critical_does_not_set_status(self):
        from src.utils.otel_utils import set_error_context
        span = MagicMock()
        span.is_recording.return_value = True

        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_error_context("MinorError", "oops", is_critical=False)

        span.set_status.assert_not_called()

    def test_exception_in_span_set_is_swallowed(self):
        from src.utils.otel_utils import set_error_context
        span = MagicMock()
        span.is_recording.return_value = True
        span.set_attribute.side_effect = RuntimeError("span fail")

        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            set_error_context("E", "msg")  # Must not raise

    def test_long_message_truncated(self):
        from src.utils.otel_utils import set_error_context
        span = MagicMock()
        span.is_recording.return_value = True

        with patch("src.utils.otel_utils.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            long_msg = "x" * 1000
            set_error_context("E", long_msg)

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert len(calls.get("error.message", "")) <= 500


# ---------------------------------------------------------------------------
# set_message_id
# ---------------------------------------------------------------------------

class TestSetMessageId:

    def test_sets_both_attributes(self):
        from src.utils.otel_utils import set_message_id
        with patch("src.utils.otel_utils.set_span_attribute") as mock_set:
            set_message_id("msg-001")
        keys = [c[0][0] for c in mock_set.call_args_list]
        assert "message.id" in keys
        assert "message_id" in keys

    def test_falsy_value_does_nothing(self):
        from src.utils.otel_utils import set_message_id
        with patch("src.utils.otel_utils.set_span_attribute") as mock_set:
            set_message_id("")
        mock_set.assert_not_called()

    def test_none_value_does_nothing(self):
        from src.utils.otel_utils import set_message_id
        with patch("src.utils.otel_utils.set_span_attribute") as mock_set:
            set_message_id(None)
        mock_set.assert_not_called()


# ---------------------------------------------------------------------------
# force_user_context_to_xray
# ---------------------------------------------------------------------------

class TestForceUserContextToXray:

    def _call(self, **kwargs):
        from src.utils.otel_utils import force_user_context_to_xray
        with patch("src.utils.otel_utils.set_xray_annotation") as mock_ann:
            force_user_context_to_xray(**kwargs)
            return mock_ann.call_args_list

    def test_all_params_annotated(self):
        calls = self._call(user_id="u1", user_email="e@x.com", username="uname", realm="master")
        keys = [c[0][0] for c in calls]
        assert set(keys) == {"user_id", "user_email", "username", "realm"}

    def test_no_params_no_annotations(self):
        calls = self._call()
        assert calls == []

    def test_only_user_id(self):
        calls = self._call(user_id="u1")
        assert any(c[0][0] == "user_id" for c in calls)

    def test_only_username(self):
        calls = self._call(username="admin")
        assert any(c[0][0] == "username" for c in calls)

    def test_only_realm(self):
        calls = self._call(realm="test-realm")
        assert any(c[0][0] == "realm" for c in calls)
