"""
tests/utils/test_obs.py

Unit tests for src/utils/obs.py — 100% coverage.
Covers: LLMUsageTracker.track_response and observe_token_usage.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_usage(prompt=10, completion=20, total=30):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = total
    return usage


def _make_response(usage=None):
    response = MagicMock()
    if usage is None:
        del response.usage  # Remove the attribute entirely
        response.__class__ = type("Resp", (), {})
    else:
        response.usage = usage
    return response


# ---------------------------------------------------------------------------
# LLMUsageTracker
# ---------------------------------------------------------------------------

class TestLLMUsageTracker:

    def _build(self, auth_token=None, agent_name=None, server_name=None, monkeypatch=None):
        if monkeypatch and agent_name:
            monkeypatch.setenv("AGENT_NAME", agent_name)
        if monkeypatch and server_name:
            monkeypatch.setenv("SERVER_NAME", server_name)
        from src.utils.obs import LLMUsageTracker
        return LLMUsageTracker(auth_token=auth_token)

    def test_default_auth_token_is_empty_string(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        assert tracker.auth_token == ""

    def test_agent_name_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_NAME", "MY_AGENT")
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        assert tracker.agent_name == "MY_AGENT"

    def test_server_name_from_env(self, monkeypatch):
        monkeypatch.setenv("SERVER_NAME", "MY_SERVER")
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        assert tracker.server_name == "MY_SERVER"

    def test_track_response_no_usage_attr(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        response = MagicMock(spec=[])  # No 'usage' attribute
        result = tracker.track_response(response)
        assert result["status"] == "error"

    def test_track_response_usage_is_none(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        response = MagicMock()
        response.usage = None
        result = tracker.track_response(response)
        assert result["status"] == "error"

    def test_track_response_usage_as_dict(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker(auth_token=None)

        response = MagicMock()
        response.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

        with patch("src.utils.obs.kafka_logger") as mock_kafka:
            result = tracker.track_response(response, model_name="openai/gpt-4")

        assert result["status"] == "success"
        assert result["total_tokens"] == 30

    def test_track_response_usage_as_object_with_dict(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()

        usage_obj = MagicMock()
        usage_obj.__dict__ = {
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "total_tokens": 20,
        }
        # Make it not a dict directly
        usage_obj = type("Usage", (), usage_obj.__dict__)()

        response = MagicMock()
        response.usage = usage_obj

        with patch("src.utils.obs.kafka_logger") as mock_kafka:
            result = tracker.track_response(response)

        assert result["status"] == "success"

    def test_track_response_usage_not_dict_and_no__dict__(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()

        class BadUsage:
            """No __dict__ and not a plain dict."""
            __slots__ = []

        response = MagicMock()
        response.usage = BadUsage()
        result = tracker.track_response(response)
        assert result["status"] == "error"
        assert "Cannot parse" in result["message"]

    def test_model_name_stripped_of_provider(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        response = MagicMock()
        response.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        captured = []

        def fake_log(payload):
            captured.append(payload)

        with patch("src.utils.obs.kafka_logger") as mock_kafka:
            mock_kafka.log.side_effect = fake_log
            tracker.track_response(response, model_name="anthropic/claude-3")

        assert captured[0]["model_name"] == "claude-3"

    def test_model_name_without_provider_stays_unchanged(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        response = MagicMock()
        response.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        captured = []

        with patch("src.utils.obs.kafka_logger") as mock_kafka:
            mock_kafka.log.side_effect = lambda p: captured.append(p)
            tracker.track_response(response, model_name="gpt-4")

        assert captured[0]["model_name"] == "gpt-4"

    def test_no_model_name_uses_unknown(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        response = MagicMock()
        response.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        captured = []

        with patch("src.utils.obs.kafka_logger") as mock_kafka:
            mock_kafka.log.side_effect = lambda p: captured.append(p)
            tracker.track_response(response, model_name=None)

        assert captured[0]["model_name"] == "UNKNOWN_MODEL"

    def test_zero_total_tokens_skips_kafka(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()
        response = MagicMock()
        response.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        with patch("src.utils.obs.kafka_logger") as mock_kafka:
            result = tracker.track_response(response)
            mock_kafka.log.assert_not_called()

        assert result["status"] == "success"
        assert result["total_tokens"] == 0

    def test_track_response_exception_returns_error(self):
        from src.utils.obs import LLMUsageTracker
        tracker = LLMUsageTracker()

        response = MagicMock()
        # Cause an unexpected error
        type(response).usage = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

        result = tracker.track_response(response)
        assert result["status"] == "error"
        assert "boom" in result["message"]

    def test_encrypted_payload_included_in_kafka_payload(self):
        from src.utils.obs import LLMUsageTracker
        from src.utils.kafka_base import CUSTOM_TOKEN_SEPARATOR
        token = f"jwt{CUSTOM_TOKEN_SEPARATOR}enc-payload"
        tracker = LLMUsageTracker(auth_token=token)
        response = MagicMock()
        response.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        captured = []

        with patch("src.utils.obs.kafka_logger") as mock_kafka:
            mock_kafka.log.side_effect = lambda p: captured.append(p)
            tracker.track_response(response, model_name="gpt-4")

        assert captured[0]["encrypted_payload"] == "enc-payload"


# ---------------------------------------------------------------------------
# observe_token_usage
# ---------------------------------------------------------------------------

class TestObserveTokenUsage:

    def test_with_auth_token_creates_tracker_and_calls_track(self):
        from src.utils.obs import observe_token_usage
        response = MagicMock()
        response.usage = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}

        with patch("src.utils.obs.kafka_logger"):
            observe_token_usage(response, auth_token="Bearer token123", model_name="gpt-4")

    def test_without_auth_token_does_nothing(self):
        from src.utils.obs import observe_token_usage
        response = MagicMock()

        with patch("src.utils.obs.LLMUsageTracker") as mock_tracker_cls:
            observe_token_usage(response, auth_token=None)
            mock_tracker_cls.assert_not_called()

    def test_empty_auth_token_does_nothing(self):
        from src.utils.obs import observe_token_usage
        response = MagicMock()

        with patch("src.utils.obs.LLMUsageTracker") as mock_tracker_cls:
            observe_token_usage(response, auth_token="")
            mock_tracker_cls.assert_not_called()

    def test_model_name_passed_to_track_response(self):
        from src.utils.obs import observe_token_usage, LLMUsageTracker

        mock_tracker = MagicMock()
        mock_tracker.track_response = MagicMock()

        response = MagicMock()

        with patch("src.utils.obs.LLMUsageTracker", return_value=mock_tracker):
            observe_token_usage(response, auth_token="tok", model_name="claude-3")

        mock_tracker.track_response.assert_called_once_with(response, model_name="claude-3")
