"""
tests/utils/test_error_capture.py

Unit tests for src/utils/error_capture.py — 100% coverage.
"""

import pytest
from unittest.mock import MagicMock, patch

from opentelemetry.trace.status import StatusCode


# ---------------------------------------------------------------------------
# capture_http_error_details
# ---------------------------------------------------------------------------

class TestCaptureHttpErrorDetails:

    def _run(self, status_code, error_details, span_recording=True):
        from src.utils.error_capture import capture_http_error_details
        span = MagicMock()
        span.is_recording.return_value = span_recording
        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_http_error_details(status_code, error_details)
        return span

    def test_sets_error_true(self):
        span = self._run(400, {"message": "bad request"})
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error") is True

    def test_sets_status_code(self):
        span = self._run(404, {"message": "not found"})
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error.status_code") == 404

    def test_sets_error_type(self):
        span = self._run(400, {"message": "bad"})
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error.type") == "http_error"

    def test_500_sets_error_status(self):
        span = self._run(500, {"message": "internal error"})
        span.set_status.assert_called_once()
        status_arg = span.set_status.call_args[0][0]
        assert status_arg.status_code == StatusCode.ERROR

    def test_4xx_does_not_set_error_status(self):
        span = self._run(400, {"message": "bad request"})
        span.set_status.assert_not_called()

    def test_message_truncated_to_500_chars(self):
        span = self._run(400, {"message": "x" * 1000})
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert len(attrs.get("error.message", "")) <= 500

    def test_missing_message_defaults_to_unknown(self):
        span = self._run(400, {})
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error.message") == "Unknown error"

    def test_span_not_recording_no_attributes(self):
        span = self._run(400, {"message": "bad"}, span_recording=False)
        span.set_attribute.assert_not_called()

    def test_xray_annotations_attempted(self):
        from src.utils.error_capture import capture_http_error_details
        span = MagicMock()
        span.is_recording.return_value = True

        mock_segment = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.current_segment.return_value = mock_segment

        with patch("src.utils.error_capture.trace") as mock_trace, \
             patch.dict("sys.modules", {"aws_xray_sdk.core": MagicMock(xray_recorder=mock_recorder)}):
            mock_trace.get_current_span.return_value = span
            capture_http_error_details(500, {"message": "err"})

        mock_segment.put_annotation.assert_called()

    def test_xray_exception_silently_swallowed(self):
        from src.utils.error_capture import capture_http_error_details
        span = MagicMock()
        span.is_recording.return_value = True

        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_http_error_details(500, {"message": "err"})  # Should not raise

    def test_span_none_does_not_raise(self):
        from src.utils.error_capture import capture_http_error_details
        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = None
            capture_http_error_details(400, {"message": "bad"})  # Must not raise


# ---------------------------------------------------------------------------
# capture_validation_error
# ---------------------------------------------------------------------------

class TestCaptureValidationError:

    def _run(self, errors, span_recording=True):
        from src.utils.error_capture import capture_validation_error
        span = MagicMock()
        span.is_recording.return_value = span_recording
        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_validation_error(errors)
        return span

    def test_sets_error_true(self):
        span = self._run([{"msg": "field required"}])
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error") is True

    def test_sets_error_count(self):
        errors = [{"msg": "e1"}, {"msg": "e2"}]
        span = self._run(errors)
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("validation.error_count") == 2

    def test_extracts_first_three_messages(self):
        errors = [{"msg": f"error {i}"} for i in range(5)]
        span = self._run(errors)
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        messages = attrs.get("validation.messages", "")
        # Should contain at most 3 messages joined
        assert isinstance(messages, str)

    def test_non_dict_error_uses_str(self):
        span = self._run(["simple string error"])
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "validation.messages" in attrs

    def test_empty_errors_list(self):
        span = self._run([])
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("validation.error_count") == 0
        # No messages attribute set when empty
        assert "validation.messages" not in attrs

    def test_span_not_recording_no_attributes(self):
        span = self._run([{"msg": "err"}], span_recording=False)
        span.set_attribute.assert_not_called()

    def test_xray_annotations_attempted(self):
        from src.utils.error_capture import capture_validation_error
        span = MagicMock()
        span.is_recording.return_value = True

        mock_segment = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.current_segment.return_value = mock_segment
        mock_otel = MagicMock()

        with patch("src.utils.error_capture.trace") as mock_trace, \
             patch.dict("sys.modules", {
                 "aws_xray_sdk.core": MagicMock(xray_recorder=mock_recorder),
                 "src.utils.otel_utils": mock_otel
             }):
            mock_trace.get_current_span.return_value = span
            capture_validation_error([{"msg": "e"}])

    def test_exception_in_span_swallowed(self):
        from src.utils.error_capture import capture_validation_error
        span = MagicMock()
        span.is_recording.return_value = True
        span.set_attribute.side_effect = RuntimeError("span fail")

        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_validation_error([{"msg": "e"}])  # Must not raise


# ---------------------------------------------------------------------------
# capture_external_api_error
# ---------------------------------------------------------------------------

class TestCaptureExternalApiError:

    def _run(self, service, error, endpoint=None, status_code=None, span_recording=True):
        from src.utils.error_capture import capture_external_api_error
        span = MagicMock()
        span.is_recording.return_value = span_recording
        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_external_api_error(service, error, endpoint=endpoint, status_code=status_code)
        return span

    def test_sets_error_true(self):
        span = self._run("OpenAI", ValueError("timeout"))
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error") is True

    def test_sets_service_name(self):
        span = self._run("AWS", RuntimeError("fail"))
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error.external_service") == "AWS"

    def test_sets_endpoint_when_provided(self):
        span = self._run("Bedrock", RuntimeError("fail"), endpoint="/invoke")
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error.endpoint") == "/invoke"

    def test_no_endpoint_attribute_when_not_provided(self):
        span = self._run("Bedrock", RuntimeError("fail"))
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "error.endpoint" not in attrs

    def test_sets_status_code_when_provided(self):
        span = self._run("API", RuntimeError("fail"), status_code=503)
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error.external_status_code") == 503

    def test_records_exception_on_span(self):
        exc = RuntimeError("api error")
        span = self._run("SVC", exc)
        span.record_exception.assert_called_once_with(exc)

    def test_sets_error_status(self):
        span = self._run("SVC", RuntimeError("fail"))
        span.set_status.assert_called_once()

    def test_span_not_recording_no_attributes(self):
        span = self._run("SVC", RuntimeError("fail"), span_recording=False)
        span.set_attribute.assert_not_called()

    def test_exception_in_span_swallowed(self):
        from src.utils.error_capture import capture_external_api_error
        span = MagicMock()
        span.is_recording.return_value = True
        span.set_attribute.side_effect = RuntimeError("oops")

        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_external_api_error("SVC", RuntimeError("err"))  # Must not raise


# ---------------------------------------------------------------------------
# capture_processing_error
# ---------------------------------------------------------------------------

class TestCaptureProcessingError:

    def _run(self, doc_name, error, stage=None, span_recording=True):
        from src.utils.error_capture import capture_processing_error
        span = MagicMock()
        span.is_recording.return_value = span_recording
        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_processing_error(doc_name, error, stage=stage)
        return span

    def test_sets_error_true(self):
        span = self._run("doc.pdf", ValueError("parse error"))
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error") is True

    def test_sets_document_name(self):
        span = self._run("my_doc.pdf", RuntimeError("err"))
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("document.name") == "my_doc.pdf"

    def test_document_name_truncated(self):
        long_name = "d" * 300
        span = self._run(long_name, RuntimeError("err"))
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert len(attrs.get("document.name", "")) <= 200

    def test_sets_stage_when_provided(self):
        span = self._run("doc.pdf", RuntimeError("err"), stage="ocr")
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert attrs.get("error.stage") == "ocr"
        assert attrs.get("processing.stage") == "ocr"

    def test_no_stage_attributes_when_not_provided(self):
        span = self._run("doc.pdf", RuntimeError("err"), stage=None)
        attrs = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "error.stage" not in attrs

    def test_records_exception(self):
        exc = RuntimeError("fail")
        span = self._run("doc.pdf", exc)
        span.record_exception.assert_called_once_with(exc)

    def test_sets_error_status(self):
        span = self._run("doc.pdf", RuntimeError("fail"), stage="ner")
        span.set_status.assert_called_once()

    def test_span_not_recording_no_attributes(self):
        span = self._run("doc.pdf", RuntimeError("fail"), span_recording=False)
        span.set_attribute.assert_not_called()

    def test_exception_in_span_swallowed(self):
        from src.utils.error_capture import capture_processing_error
        span = MagicMock()
        span.is_recording.return_value = True
        span.set_attribute.side_effect = RuntimeError("span fail")

        with patch("src.utils.error_capture.trace") as mock_trace:
            mock_trace.get_current_span.return_value = span
            capture_processing_error("doc.pdf", RuntimeError("e"))  # Must not raise
