"""
tests/utils/test_logger.py

Unit tests for src/utils/logger.py — 100% coverage.
"""

import logging
import pytest

from src.utils.logger import setup_logger


class TestSetupLogger:
    """Tests for the setup_logger factory function."""

    def test_returns_logger_instance(self):
        logger = setup_logger("test_default")
        assert isinstance(logger, logging.Logger)

    def test_default_name(self):
        logger = setup_logger()
        assert logger.name == "database_manager"

    def test_custom_name(self):
        logger = setup_logger("my_custom_logger")
        assert logger.name == "my_custom_logger"

    def test_default_level_is_info(self):
        logger = setup_logger("test_level_info")
        assert logger.level == logging.INFO

    def test_custom_level_debug(self):
        logger = setup_logger("test_level_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_custom_level_warning(self):
        logger = setup_logger("test_level_warning", level=logging.WARNING)
        assert logger.level == logging.WARNING

    def test_custom_level_error(self):
        logger = setup_logger("test_level_error", level=logging.ERROR)
        assert logger.level == logging.ERROR

    def test_custom_level_critical(self):
        logger = setup_logger("test_level_critical", level=logging.CRITICAL)
        assert logger.level == logging.CRITICAL

    def test_adds_stream_handler(self):
        logger = setup_logger("test_handler_added")
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_handler_has_formatter(self):
        logger = setup_logger("test_formatter")
        for handler in logger.handlers:
            assert handler.formatter is not None

    def test_formatter_contains_expected_fields(self):
        logger = setup_logger("test_formatter_fields")
        handler = logger.handlers[0]
        fmt_str = handler.formatter._fmt
        assert "%(asctime)s" in fmt_str
        assert "%(name)s" in fmt_str
        assert "%(levelname)s" in fmt_str

    def test_no_duplicate_handlers_on_repeated_call(self):
        """Calling setup_logger twice with the same name must NOT add extra handlers."""
        name = "test_no_dup_handlers"
        logger1 = setup_logger(name)
        handler_count_first = len(logger1.handlers)

        logger2 = setup_logger(name)
        assert len(logger2.handlers) == handler_count_first

    def test_same_object_returned_for_same_name(self):
        """Python's logging.getLogger is a registry — same name → same object."""
        name = "test_same_object"
        logger1 = setup_logger(name)
        logger2 = setup_logger(name)
        assert logger1 is logger2

    def test_handler_level_is_debug(self):
        logger = setup_logger("test_handler_level")
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                assert handler.level == logging.DEBUG
                break

    def test_logger_propagate_default(self):
        """Propagation should be left at the Python default (True)."""
        logger = setup_logger("test_propagate")
        # The function does not explicitly set propagate, so it stays True
        assert logger.propagate is True

    def test_unique_loggers_independent(self):
        """Different names produce independent logger instances."""
        l1 = setup_logger("unique_a")
        l2 = setup_logger("unique_b")
        assert l1 is not l2
        assert l1.name != l2.name
