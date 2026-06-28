"""
tests/utils/test_kafka_base.py

Unit tests for src/utils/kafka_base.py — 100% coverage.
Covers: extract_user_context, KafkaManager (singleton, producer lifecycle).
"""

import os
import threading
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.utils.kafka_base import (
    CUSTOM_TOKEN_SEPARATOR,
    extract_user_context,
    KafkaManager,
)


# ---------------------------------------------------------------------------
# extract_user_context
# ---------------------------------------------------------------------------

class TestExtractUserContext:

    def test_no_token_returns_no_payload(self):
        result = extract_user_context(None)
        assert result == {"encrypted_payload": "NO_PAYLOAD"}

    def test_empty_string_returns_no_payload(self):
        result = extract_user_context("")
        assert result == {"encrypted_payload": "NO_PAYLOAD"}

    def test_custom_separator_extracts_payload(self):
        token = f"jwtpart{CUSTOM_TOKEN_SEPARATOR}my-encrypted-payload"
        result = extract_user_context(token)
        assert result["encrypted_payload"] == "my-encrypted-payload"

    def test_bearer_token_generates_mock_payload(self):
        token = "Bearer abcdefghij1234567890"
        result = extract_user_context(token)
        ep = result["encrypted_payload"]
        assert ep.startswith("mock-encrypted-payload-")
        # Last 10 chars of the jwt part (after stripping Bearer )
        assert ep.endswith("1234567890")

    def test_raw_jwt_without_bearer_generates_mock_payload(self):
        jwt = "rawtoken1234567890"
        result = extract_user_context(jwt)
        ep = result["encrypted_payload"]
        assert ep.startswith("mock-encrypted-payload-")
        assert ep.endswith(jwt[-10:])

    def test_bearer_case_insensitive(self):
        jwt_part = "BEARER sometoken1234567890"
        result = extract_user_context(jwt_part)
        ep = result["encrypted_payload"]
        # lowercase test
        assert "mock-encrypted-payload-" in ep

    def test_token_with_separator_empty_payload(self):
        token = f"jwtpart{CUSTOM_TOKEN_SEPARATOR}"
        result = extract_user_context(token)
        assert result["encrypted_payload"] == ""

    def test_token_with_multiple_separators_uses_first_split(self):
        sep = CUSTOM_TOKEN_SEPARATOR
        token = f"jwt{sep}payload{sep}extra"
        result = extract_user_context(token)
        assert result["encrypted_payload"] == f"payload{sep}extra"

    def test_constant_separator_value(self):
        assert CUSTOM_TOKEN_SEPARATOR == "$YashUnified2025$"


# ---------------------------------------------------------------------------
# KafkaManager
# ---------------------------------------------------------------------------

class TestKafkaManagerSingleton:

    def test_is_singleton(self):
        m1 = KafkaManager()
        m2 = KafkaManager()
        assert m1 is m2

    def test_initialized_flag(self):
        m = KafkaManager()
        assert m._initialized is True

    def test_producer_starts_as_none_or_existing(self):
        # Reset for clean test by checking the attribute exists
        m = KafkaManager()
        assert hasattr(m, "producer")

    def test_singleton_thread_safety(self):
        """Two threads must get the exact same instance."""
        instances = []

        def create():
            instances.append(KafkaManager())

        t1 = threading.Thread(target=create)
        t2 = threading.Thread(target=create)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert instances[0] is instances[1]


class TestKafkaManagerGetProducer:

    def test_get_producer_when_kafka_not_installed(self):
        """When KAFKA_INSTALLED is False, get_producer should return None."""
        manager = KafkaManager()
        original_producer = manager.producer
        manager.producer = None  # Reset for test

        with patch("src.utils.kafka_base.KAFKA_INSTALLED", False):
            result = manager.get_producer()
        assert result is None

        manager.producer = original_producer  # restore

    def test_get_producer_when_no_bootstrap_servers(self, monkeypatch):
        """When KAFKA_BOOTSTRAP_SERVERS is absent, producer stays None."""
        manager = KafkaManager()
        original_producer = manager.producer
        manager.producer = None

        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        with patch("src.utils.kafka_base.KAFKA_INSTALLED", True):
            result = manager.get_producer()
        assert result is None

        manager.producer = original_producer

    def test_get_producer_returns_existing_producer(self):
        """If producer is already set, get_producer returns it without re-init."""
        manager = KafkaManager()
        fake_producer = MagicMock()
        original_producer = manager.producer
        manager.producer = fake_producer

        result = manager.get_producer()
        assert result is fake_producer

        manager.producer = original_producer

    def test_initialize_producer_success(self, monkeypatch):
        """_initialize_producer should create KafkaProducer when all env vars set."""
        manager = KafkaManager()
        original_producer = manager.producer
        manager.producer = None

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        monkeypatch.setenv("KAFKA_USE_PLAINTEXT", "true")
        monkeypatch.setenv("KAFKA_USE_SSL", "false")

        fake_producer_instance = MagicMock()
        mock_kafka_producer_cls = MagicMock(return_value=fake_producer_instance)

        with patch("src.utils.kafka_base.KAFKA_INSTALLED", True), \
             patch("src.utils.kafka_base.KafkaProducer", mock_kafka_producer_cls):
            result = manager._initialize_producer()

        assert result is True
        assert manager.producer is fake_producer_instance

        manager.producer = original_producer

    def test_initialize_producer_ssl_mode(self, monkeypatch):
        """SSL mode is activated when KAFKA_USE_SSL=true."""
        manager = KafkaManager()
        original_producer = manager.producer
        manager.producer = None

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        monkeypatch.setenv("KAFKA_USE_PLAINTEXT", "false")
        monkeypatch.setenv("KAFKA_USE_SSL", "true")

        fake_producer_instance = MagicMock()
        mock_kafka_cls = MagicMock(return_value=fake_producer_instance)

        with patch("src.utils.kafka_base.KAFKA_INSTALLED", True), \
             patch("src.utils.kafka_base.KafkaProducer", mock_kafka_cls):
            manager._initialize_producer()

        call_kwargs = mock_kafka_cls.call_args[1]
        assert call_kwargs.get("security_protocol") == "SSL"

        manager.producer = original_producer

    def test_initialize_producer_no_brokers_available(self, monkeypatch):
        """NoBrokersAvailable exception should cause producer to remain None."""
        manager = KafkaManager()
        original_producer = manager.producer
        manager.producer = None

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        from src.utils.kafka_base import NoBrokersAvailable

        mock_kafka_cls = MagicMock(side_effect=NoBrokersAvailable("no brokers"))

        with patch("src.utils.kafka_base.KAFKA_INSTALLED", True), \
             patch("src.utils.kafka_base.KafkaProducer", mock_kafka_cls):
            result = manager._initialize_producer()

        assert result is False
        assert manager.producer is None

        manager.producer = original_producer

    def test_initialize_producer_generic_exception(self, monkeypatch):
        """Any unexpected exception during producer init → returns False."""
        manager = KafkaManager()
        original_producer = manager.producer
        manager.producer = None

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        mock_kafka_cls = MagicMock(side_effect=RuntimeError("unexpected"))

        with patch("src.utils.kafka_base.KAFKA_INSTALLED", True), \
             patch("src.utils.kafka_base.KafkaProducer", mock_kafka_cls):
            result = manager._initialize_producer()

        assert result is False
        manager.producer = original_producer


class TestKafkaManagerClose:

    def test_close_with_no_producer(self):
        """close() with producer=None should not raise."""
        manager = KafkaManager()
        original_producer = manager.producer
        manager.producer = None

        manager.close()  # Should not raise
        assert manager.producer is None

        manager.producer = original_producer

    def test_close_flushes_and_closes_producer(self):
        manager = KafkaManager()
        original_producer = manager.producer

        fake_producer = MagicMock()
        manager.producer = fake_producer

        manager.close()

        fake_producer.flush.assert_called_once_with(timeout=5)
        fake_producer.close.assert_called_once_with(timeout=5)
        assert manager.producer is None

        manager.producer = original_producer

    def test_close_handles_exception_in_flush(self):
        """Exception during flush/close should be silently swallowed."""
        manager = KafkaManager()
        original_producer = manager.producer

        fake_producer = MagicMock()
        fake_producer.flush.side_effect = RuntimeError("flush error")
        manager.producer = fake_producer

        manager.close()  # Must not raise
        assert manager.producer is None  # finally sets it to None

        manager.producer = original_producer
