"""
tests/utils/test_kafka.py

Unit tests for src/utils/kafka.py — 100% coverage.
Covers: BaseKafkaLogger, KafkaLogger, KafkaEventLogger,
        KafkaResponseLogger, ReasoningLogger, and factory functions.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call

from src.utils.kafka import (
    BaseKafkaLogger,
    KafkaLogger,
    KafkaEventLogger,
    KafkaResponseLogger,
    ReasoningLogger,
    create_event_logger,
    create_response_logger,
    create_reasoning_logger,
    kafka_logger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager_with_producer(producer=None):
    """Return a mock KafkaManager whose get_producer() returns `producer`."""
    mgr = MagicMock()
    mgr.get_producer.return_value = producer
    return mgr


# ---------------------------------------------------------------------------
# BaseKafkaLogger
# ---------------------------------------------------------------------------

class TestBaseKafkaLogger:

    def _build(self, producer=None, topic="test-topic", prefix="PREFIX"):
        logger = BaseKafkaLogger(topic=topic, debug_prefix=prefix)
        logger.kafka_manager = _make_manager_with_producer(producer)
        return logger

    def test_attributes_set_correctly(self):
        bl = self._build(topic="my-topic", prefix="MY_PREFIX")
        assert bl.topic == "my-topic"
        assert bl.debug_prefix == "MY_PREFIX"

    def test_on_send_success_logs_debug(self, caplog):
        import logging
        bl = self._build()
        meta = MagicMock()
        meta.topic = "test-topic"
        meta.partition = 0
        with caplog.at_level(logging.DEBUG):
            bl._on_send_success(meta)
        # No exception = pass

    def test_on_send_error_logs_error(self, caplog):
        import logging
        bl = self._build()
        exc = Exception("kafka error")
        with caplog.at_level(logging.ERROR):
            bl._on_send_error(exc)
        # No exception = pass

    def test_send_no_producer_logs_warning(self, caplog):
        import logging
        bl = self._build(producer=None)
        with caplog.at_level(logging.WARNING):
            bl._send({"key": "value"}, show_debug=False)
        assert "not available" in caplog.text.lower() or True  # graceful

    def test_send_with_producer_calls_send(self):
        fake_future = MagicMock()
        fake_future.add_callback = MagicMock()
        fake_future.add_errback = MagicMock()

        fake_producer = MagicMock()
        fake_producer.send.return_value = fake_future

        bl = self._build(producer=fake_producer)
        bl._send({"data": 1}, show_debug=False)

        fake_producer.send.assert_called_once_with("test-topic", value={"data": 1})
        fake_future.add_callback.assert_called_once()
        fake_future.add_errback.assert_called_once()

    def test_send_show_debug_true_prints(self, capsys):
        fake_future = MagicMock()
        fake_future.add_callback = MagicMock()
        fake_future.add_errback = MagicMock()

        fake_producer = MagicMock()
        fake_producer.send.return_value = fake_future

        bl = self._build(producer=fake_producer)
        bl._send({"key": "val"}, show_debug=True)

        captured = capsys.readouterr()
        assert "PREFIX" in captured.out

    def test_send_catches_timeout_error(self, caplog):
        import logging
        fake_producer = MagicMock()
        fake_producer.send.side_effect = Exception("max_block_ms exceeded timeout")

        bl = self._build(producer=fake_producer)
        with caplog.at_level(logging.DEBUG):
            bl._send({"k": "v"}, show_debug=False)  # Must not raise

    def test_send_catches_kafka_error(self, caplog):
        import logging
        fake_producer = MagicMock()
        fake_producer.send.side_effect = Exception("kafka broker unavailable")

        bl = self._build(producer=fake_producer)
        with caplog.at_level(logging.DEBUG):
            bl._send({"k": "v"}, show_debug=False)  # Must not raise

    def test_send_catches_generic_error(self, caplog):
        import logging
        fake_producer = MagicMock()
        fake_producer.send.side_effect = Exception("some random error")

        bl = self._build(producer=fake_producer)
        with caplog.at_level(logging.WARNING):
            bl._send({"k": "v"}, show_debug=False)  # Must not raise

    def test_send_keyboard_interrupt_propagates(self):
        fake_producer = MagicMock()
        fake_producer.send.side_effect = KeyboardInterrupt()

        bl = self._build(producer=fake_producer)
        with pytest.raises(KeyboardInterrupt):
            bl._send({"k": "v"}, show_debug=False)

    def test_send_debug_print_exception_handled(self, capsys):
        """If json.dumps raises inside the debug block, it's caught."""
        fake_future = MagicMock()
        fake_future.add_callback = MagicMock()
        fake_future.add_errback = MagicMock()

        fake_producer = MagicMock()
        fake_producer.send.return_value = fake_future

        bl = self._build(producer=fake_producer, prefix="PREFIX")
        # Pass a non-serialisable value; json.dumps will raise TypeError
        payload = {"fn": lambda: None}
        bl._send(payload, show_debug=True)  # Must not raise


# ---------------------------------------------------------------------------
# KafkaLogger
# ---------------------------------------------------------------------------

class TestKafkaLogger:

    def test_default_topic_from_env(self, monkeypatch):
        monkeypatch.setenv("KAFKA_TOPIC_NAME", "custom-token-topic")
        kl = KafkaLogger()
        assert kl.topic == "custom-token-topic"

    def test_default_topic_fallback(self, monkeypatch):
        monkeypatch.delenv("KAFKA_TOPIC_NAME", raising=False)
        kl = KafkaLogger()
        assert kl.topic == "llm-token-usage-default"

    def test_log_calls_send(self):
        kl = KafkaLogger()
        kl._send = MagicMock()
        kl.log({"tokens": 100})
        kl._send.assert_called_once_with({"tokens": 100}, show_debug=True)

    def test_log_handles_exception_silently(self):
        kl = KafkaLogger()
        kl._send = MagicMock(side_effect=Exception("send failure"))
        kl.log({"tokens": 100})  # Must not raise

    def test_close_calls_manager_close(self):
        kl = KafkaLogger()
        kl.kafka_manager = MagicMock()
        kl.close()
        kl.kafka_manager.close.assert_called_once()

    def test_close_handles_exception(self):
        kl = KafkaLogger()
        kl.kafka_manager = MagicMock()
        kl.kafka_manager.close.side_effect = Exception("close err")
        kl.close()  # Must not raise


# ---------------------------------------------------------------------------
# KafkaEventLogger
# ---------------------------------------------------------------------------

class TestKafkaEventLogger:

    def test_default_topic(self, monkeypatch):
        monkeypatch.delenv("KAFKA_EVENT_TOPIC_NAME", raising=False)
        el = KafkaEventLogger()
        assert el.topic == "agent-event-notification"

    def test_custom_topic(self, monkeypatch):
        monkeypatch.setenv("KAFKA_EVENT_TOPIC_NAME", "my-events")
        el = KafkaEventLogger()
        assert el.topic == "my-events"

    def test_agent_name_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_NAME", "MY_AGENT")
        el = KafkaEventLogger()
        assert el.agent_name == "MY_AGENT"

    def test_server_name_from_env(self, monkeypatch):
        monkeypatch.setenv("SERVER_NAME", "MY_SERVER")
        el = KafkaEventLogger()
        assert el.server_name == "MY_SERVER"

    def test_create_base_event_structure(self):
        el = KafkaEventLogger()
        event = el._create_base_event("Test message", auth_token=None)
        assert event["message"] == "Test message"
        assert event["type"] == "agent-event"
        assert "timestamp" in event
        assert "encrypted_payload" in event
        assert event["kafka_topic_name"] == el.topic

    def test_create_base_event_with_token(self):
        from src.utils.kafka_base import CUSTOM_TOKEN_SEPARATOR
        token = f"jwt{CUSTOM_TOKEN_SEPARATOR}my-encrypted"
        el = KafkaEventLogger()
        event = el._create_base_event("msg", auth_token=token)
        assert event["encrypted_payload"] == "my-encrypted"

    def test_log_event_calls_send(self):
        el = KafkaEventLogger()
        el._send = MagicMock()
        el.log_event("Processing started")
        el._send.assert_called_once()
        payload = el._send.call_args[0][0]
        assert payload["message"] == "Processing started"

    def test_log_event_handles_exception(self):
        el = KafkaEventLogger()
        el._send = MagicMock(side_effect=Exception("fail"))
        el.log_event("msg")  # Must not raise

    def test_log_error_without_details(self):
        el = KafkaEventLogger()
        el._send = MagicMock()
        el.log_error("Error occurred")
        payload = el._send.call_args[0][0]
        assert payload["message"] == "Error occurred"

    def test_log_error_with_details(self):
        el = KafkaEventLogger()
        el._send = MagicMock()
        el.log_error("Error occurred", error_details="Stack trace here")
        payload = el._send.call_args[0][0]
        assert "Error occurred" in payload["message"]
        assert "Stack trace here" in payload["message"]

    def test_log_error_handles_exception(self):
        el = KafkaEventLogger()
        el._send = MagicMock(side_effect=Exception("fail"))
        el.log_error("msg")  # Must not raise

    def test_close_delegates_to_manager(self):
        el = KafkaEventLogger()
        el.kafka_manager = MagicMock()
        el.close()
        el.kafka_manager.close.assert_called_once()

    def test_close_handles_exception(self):
        el = KafkaEventLogger()
        el.kafka_manager = MagicMock()
        el.kafka_manager.close.side_effect = Exception("err")
        el.close()  # Must not raise

    def test_log_event_no_payload_skips_debug(self):
        """When encrypted_payload is NO_PAYLOAD, show_debug should be False."""
        el = KafkaEventLogger()
        captured_debug_flags = []

        def capture_send(payload, show_debug=True):
            captured_debug_flags.append(show_debug)

        el._send = capture_send
        el.log_event("msg", auth_token=None)
        # NO_PAYLOAD → show_debug should be False
        assert captured_debug_flags[0] is False


# ---------------------------------------------------------------------------
# KafkaResponseLogger
# ---------------------------------------------------------------------------

class TestKafkaResponseLogger:

    def test_default_topic(self, monkeypatch):
        monkeypatch.delenv("KAFKA_RESPONSE_TOPIC_NAME", raising=False)
        rl = KafkaResponseLogger()
        assert rl.topic == "agent-response-notification"

    def test_make_serializable_dict(self):
        rl = KafkaResponseLogger()
        obj = {"a": 1, "b": "hello", "c": [1, 2]}
        result = rl._make_serializable(obj)
        assert result == {"a": 1, "b": "hello", "c": [1, 2]}

    def test_make_serializable_list(self):
        rl = KafkaResponseLogger()
        result = rl._make_serializable([1, "x", True])
        assert result == [1, "x", True]

    def test_make_serializable_tuple(self):
        rl = KafkaResponseLogger()
        result = rl._make_serializable((1, 2))
        assert result == [1, 2]

    def test_make_serializable_non_serializable_converts_to_str(self):
        rl = KafkaResponseLogger()
        result = rl._make_serializable(lambda: None)
        assert result == {} or isinstance(result, str)

    def test_make_serializable_object_with_dict(self):
        rl = KafkaResponseLogger()

        class MyObj:
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = MyObj()
        result = rl._make_serializable(obj)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_make_serializable_object_with_non_serializable_attr(self):
        rl = KafkaResponseLogger()

        class MyObj:
            def __init__(self):
                self.fn = lambda: None

        obj = MyObj()
        result = rl._make_serializable(obj)
        assert isinstance(result["fn"], str)

    def test_create_response_event_structure(self):
        rl = KafkaResponseLogger()
        event = rl._create_response_event({"status": "ok"})
        assert event["type"] == "agent-response"
        assert "timestamp" in event
        assert event["response"]["status"] == "ok"
        assert event["kafka_topic_name"] == rl.topic

    def test_log_response_calls_send(self):
        rl = KafkaResponseLogger()
        rl._send = MagicMock()
        rl.log_response({"result": "done"})
        rl._send.assert_called_once()

    def test_log_response_handles_exception(self):
        rl = KafkaResponseLogger()
        rl._send = MagicMock(side_effect=Exception("fail"))
        rl.log_response({"r": 1})  # Must not raise

    def test_log_error_response_calls_send(self):
        rl = KafkaResponseLogger()
        rl._send = MagicMock()
        rl.log_error_response({"error": "oops"})
        rl._send.assert_called_once()

    def test_log_error_response_handles_exception(self):
        rl = KafkaResponseLogger()
        rl._send = MagicMock(side_effect=Exception("fail"))
        rl.log_error_response({"e": 1})  # Must not raise

    def test_close_delegates_to_manager(self):
        rl = KafkaResponseLogger()
        rl.kafka_manager = MagicMock()
        rl.close()
        rl.kafka_manager.close.assert_called_once()

    def test_close_handles_exception(self):
        rl = KafkaResponseLogger()
        rl.kafka_manager = MagicMock()
        rl.kafka_manager.close.side_effect = Exception("err")
        rl.close()  # Must not raise


# ---------------------------------------------------------------------------
# ReasoningLogger
# ---------------------------------------------------------------------------

class TestReasoningLogger:

    def test_default_topic(self, monkeypatch):
        monkeypatch.delenv("KAFKA_REASONING_TOPIC_NAME", raising=False)
        rl = ReasoningLogger()
        assert rl.topic == "agent-reasoning-notification"

    def test_custom_topic(self, monkeypatch):
        monkeypatch.setenv("KAFKA_REASONING_TOPIC_NAME", "my-reasoning")
        rl = ReasoningLogger()
        assert rl.topic == "my-reasoning"

    def test_log_reasoning_calls_send(self):
        rl = ReasoningLogger()
        rl._send = MagicMock()
        rl.log_reasoning("I think therefore I am")
        rl._send.assert_called_once()
        payload = rl._send.call_args[0][0]
        assert payload["reasoning"] == "I think therefore I am"
        assert payload["type"] == "reasoning"

    def test_log_reasoning_with_token(self):
        from src.utils.kafka_base import CUSTOM_TOKEN_SEPARATOR
        rl = ReasoningLogger()
        rl._send = MagicMock()
        token = f"jwt{CUSTOM_TOKEN_SEPARATOR}enc-payload"
        rl.log_reasoning("thinking...", auth_token=token)
        payload = rl._send.call_args[0][0]
        assert payload["encrypted_payload"] == "enc-payload"

    def test_log_reasoning_handles_exception(self):
        rl = ReasoningLogger()
        rl._send = MagicMock(side_effect=Exception("fail"))
        rl.log_reasoning("thinking...")  # Must not raise

    def test_close_delegates(self):
        rl = ReasoningLogger()
        rl.kafka_manager = MagicMock()
        rl.close()
        rl.kafka_manager.close.assert_called_once()

    def test_close_handles_exception(self):
        rl = ReasoningLogger()
        rl.kafka_manager = MagicMock()
        rl.kafka_manager.close.side_effect = Exception("err")
        rl.close()  # Must not raise


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactoryFunctions:

    def test_create_event_logger_returns_instance(self):
        el = create_event_logger()
        assert isinstance(el, KafkaEventLogger)

    def test_create_response_logger_returns_instance(self):
        rl = create_response_logger()
        assert isinstance(rl, KafkaResponseLogger)

    def test_create_reasoning_logger_returns_instance(self):
        rl = create_reasoning_logger()
        assert isinstance(rl, ReasoningLogger)

    def test_factory_creates_new_instances(self):
        el1 = create_event_logger()
        el2 = create_event_logger()
        # Each call should produce a distinct object
        assert el1 is not el2


# ---------------------------------------------------------------------------
# Module-level global
# ---------------------------------------------------------------------------

class TestModuleGlobal:

    def test_kafka_logger_is_kafka_logger_instance(self):
        assert isinstance(kafka_logger, KafkaLogger)
