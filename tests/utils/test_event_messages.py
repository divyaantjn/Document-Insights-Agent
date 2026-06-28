"""
tests/utils/test_event_messages.py

Unit tests for src/utils/event_messages.py — 100% coverage.
"""

import pytest

from src.utils.event_messages import EventMessages, EventTypes, EventPriority


class TestEventMessages:
    """Verify every constant in EventMessages is a non-empty string."""

    # ---- Core workflow ----
    def test_task_received(self):
        assert isinstance(EventMessages.TASK_RECEIVED, str)
        assert EventMessages.TASK_RECEIVED

    def test_analyzing_document(self):
        assert isinstance(EventMessages.ANALYZING_DOCUMENT, str)
        assert EventMessages.ANALYZING_DOCUMENT

    def test_extracting_text(self):
        assert isinstance(EventMessages.EXTRACTING_TEXT, str)
        assert EventMessages.EXTRACTING_TEXT

    def test_processing_ocr(self):
        assert isinstance(EventMessages.PROCESSING_OCR, str)
        assert EventMessages.PROCESSING_OCR

    def test_processing_ner(self):
        assert isinstance(EventMessages.PROCESSING_NER, str)
        assert EventMessages.PROCESSING_NER

    def test_finalizing_results(self):
        assert isinstance(EventMessages.FINALIZING_RESULTS, str)
        assert EventMessages.FINALIZING_RESULTS

    def test_task_completed(self):
        assert isinstance(EventMessages.TASK_COMPLETED, str)
        assert EventMessages.TASK_COMPLETED

    # ---- Upload ----
    def test_upload_start(self):
        assert isinstance(EventMessages.UPLOAD_START, str)
        assert EventMessages.UPLOAD_START

    def test_upload_complete(self):
        assert isinstance(EventMessages.UPLOAD_COMPLETE, str)
        assert EventMessages.UPLOAD_COMPLETE

    def test_documents_ingested(self):
        assert isinstance(EventMessages.DOCUMENTS_INGESTED, str)
        assert EventMessages.DOCUMENTS_INGESTED

    # ---- Query ----
    def test_query_received(self):
        assert isinstance(EventMessages.QUERY_RECEIVED, str)
        assert EventMessages.QUERY_RECEIVED

    def test_query_completed(self):
        assert isinstance(EventMessages.QUERY_COMPLETED, str)
        assert EventMessages.QUERY_COMPLETED

    # ---- ZIP ----
    def test_zip_processing_start(self):
        assert isinstance(EventMessages.ZIP_PROCESSING_START, str)
        assert EventMessages.ZIP_PROCESSING_START

    def test_zip_files_analyzed(self):
        assert isinstance(EventMessages.ZIP_FILES_ANALYZED, str)
        assert EventMessages.ZIP_FILES_ANALYZED

    def test_zip_processing_complete(self):
        assert isinstance(EventMessages.ZIP_PROCESSING_COMPLETE, str)
        assert EventMessages.ZIP_PROCESSING_COMPLETE

    # ---- Error ----
    def test_error_invalid_request(self):
        assert isinstance(EventMessages.ERROR_INVALID_REQUEST, str)
        assert EventMessages.ERROR_INVALID_REQUEST

    def test_error_processing_failed(self):
        assert isinstance(EventMessages.ERROR_PROCESSING_FAILED, str)
        assert EventMessages.ERROR_PROCESSING_FAILED

    def test_error_ocr_failed(self):
        assert isinstance(EventMessages.ERROR_OCR_FAILED, str)
        assert EventMessages.ERROR_OCR_FAILED

    def test_error_ner_failed(self):
        assert isinstance(EventMessages.ERROR_NER_FAILED, str)
        assert EventMessages.ERROR_NER_FAILED

    def test_error_upload_failed(self):
        assert isinstance(EventMessages.ERROR_UPLOAD_FAILED, str)
        assert EventMessages.ERROR_UPLOAD_FAILED

    def test_error_query_failed(self):
        assert isinstance(EventMessages.ERROR_QUERY_FAILED, str)
        assert EventMessages.ERROR_QUERY_FAILED

    def test_error_system_unavailable(self):
        assert isinstance(EventMessages.ERROR_SYSTEM_UNAVAILABLE, str)
        assert EventMessages.ERROR_SYSTEM_UNAVAILABLE

    def test_all_messages_unique(self):
        """No two message constants should share the exact same string."""
        messages = [
            EventMessages.TASK_RECEIVED,
            EventMessages.ANALYZING_DOCUMENT,
            EventMessages.EXTRACTING_TEXT,
            EventMessages.PROCESSING_OCR,
            EventMessages.PROCESSING_NER,
            EventMessages.FINALIZING_RESULTS,
            EventMessages.TASK_COMPLETED,
            EventMessages.UPLOAD_START,
            EventMessages.UPLOAD_COMPLETE,
            EventMessages.DOCUMENTS_INGESTED,
            EventMessages.QUERY_RECEIVED,
            EventMessages.QUERY_COMPLETED,
            EventMessages.ZIP_PROCESSING_START,
            EventMessages.ZIP_FILES_ANALYZED,
            EventMessages.ZIP_PROCESSING_COMPLETE,
            EventMessages.ERROR_INVALID_REQUEST,
            EventMessages.ERROR_PROCESSING_FAILED,
            EventMessages.ERROR_OCR_FAILED,
            EventMessages.ERROR_NER_FAILED,
            EventMessages.ERROR_UPLOAD_FAILED,
            EventMessages.ERROR_QUERY_FAILED,
            EventMessages.ERROR_SYSTEM_UNAVAILABLE,
        ]
        assert len(messages) == len(set(messages))


class TestEventTypes:
    """Verify every constant in EventTypes is a non-empty string."""

    def test_system(self):
        assert EventTypes.SYSTEM == "system"

    def test_agent(self):
        assert EventTypes.AGENT == "agent"

    def test_task(self):
        assert EventTypes.TASK == "task"

    def test_document(self):
        assert EventTypes.DOCUMENT == "document"

    def test_ocr(self):
        assert EventTypes.OCR == "ocr"

    def test_ner(self):
        assert EventTypes.NER == "ner"

    def test_upload(self):
        assert EventTypes.UPLOAD == "upload"

    def test_query(self):
        assert EventTypes.QUERY == "query"

    def test_success(self):
        assert EventTypes.SUCCESS == "success"

    def test_error(self):
        assert EventTypes.ERROR == "error"

    def test_progress(self):
        assert EventTypes.PROGRESS == "progress"

    def test_all_unique(self):
        types = [
            EventTypes.SYSTEM, EventTypes.AGENT, EventTypes.TASK,
            EventTypes.DOCUMENT, EventTypes.OCR, EventTypes.NER,
            EventTypes.UPLOAD, EventTypes.QUERY, EventTypes.SUCCESS,
            EventTypes.ERROR, EventTypes.PROGRESS,
        ]
        assert len(types) == len(set(types))


class TestEventPriority:
    """Verify every constant in EventPriority is a non-empty string."""

    def test_low(self):
        assert EventPriority.LOW == "low"

    def test_normal(self):
        assert EventPriority.NORMAL == "normal"

    def test_high(self):
        assert EventPriority.HIGH == "high"

    def test_critical(self):
        assert EventPriority.CRITICAL == "critical"

    def test_all_unique(self):
        priorities = [EventPriority.LOW, EventPriority.NORMAL, EventPriority.HIGH, EventPriority.CRITICAL]
        assert len(priorities) == len(set(priorities))
