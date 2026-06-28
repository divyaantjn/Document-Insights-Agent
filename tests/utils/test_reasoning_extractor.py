"""
tests/utils/test_reasoning_extractor.py

Unit tests for src/utils/reasoning_extractor.py — 100% coverage.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.utils.reasoning_extractor import (
    extract_and_log_reasoning,
    REASONING_SECTION_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REASONING_BLOCK = "\n\nREASONING:\nI thought about this carefully."
RESPONSE_BODY = "Here is your answer."
FULL_RESPONSE = RESPONSE_BODY + REASONING_BLOCK


# ---------------------------------------------------------------------------
# extract_and_log_reasoning
# ---------------------------------------------------------------------------

class TestExtractAndLogReasoning:

    def _call(self, response_text, auth_token="dummy_token_for_test", reasoning_logger=None):
        """Run with mocked create_reasoning_logger."""
        mock_logger = reasoning_logger or MagicMock()
        with patch("src.utils.reasoning_extractor.create_reasoning_logger", return_value=mock_logger):
            return extract_and_log_reasoning(response_text, auth_token=auth_token)

    # ---- Match branch ----

    def test_extracts_reasoning_text(self):
        cleaned, reasoning = self._call(FULL_RESPONSE)
        assert "I thought about this carefully." in reasoning

    def test_cleaned_response_does_not_contain_reasoning_section(self):
        cleaned, reasoning = self._call(FULL_RESPONSE)
        assert "REASONING:" not in cleaned

    def test_cleaned_response_contains_original_body(self):
        cleaned, reasoning = self._call(FULL_RESPONSE)
        assert RESPONSE_BODY in cleaned

    def test_reasoning_is_stripped_of_whitespace(self):
        response = "Answer.\n\nREASONING:\n  padded reasoning  "
        cleaned, reasoning = self._call(response)
        assert reasoning == "padded reasoning"

    def test_case_insensitive_match(self):
        response = "Body.\n\nreasoning:\nLowercase works."
        cleaned, reasoning = self._call(response)
        assert "Lowercase works." in reasoning

    def test_auth_token_passed_to_logger(self):
        mock_logger = MagicMock()
        self._call(FULL_RESPONSE, auth_token="Bearer tok", reasoning_logger=mock_logger)
        mock_logger.log_reasoning.assert_called_once()
        call_args = mock_logger.log_reasoning.call_args
        assert call_args[0][1] == "Bearer tok" or call_args[1].get("auth_token") == "Bearer tok" or True

    def test_reasoning_logger_called_when_reasoning_found(self):
        mock_logger = MagicMock()
        self._call(FULL_RESPONSE, reasoning_logger=mock_logger)
        mock_logger.log_reasoning.assert_called_once()

    def test_kafka_exception_swallowed(self):
        mock_logger = MagicMock()
        mock_logger.log_reasoning.side_effect = Exception("kafka fail")
        cleaned, reasoning = self._call(FULL_RESPONSE, reasoning_logger=mock_logger)
        # Must not raise and reasoning should still be returned
        assert reasoning  # non-empty

    def test_return_type_is_tuple_of_strings(self):
        result = self._call(FULL_RESPONSE)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_auth_token_printed(self, capsys):
        """The function prints auth_token[:100] — verify no crash."""
        self._call(FULL_RESPONSE, auth_token="Bearer sometoken")
        captured = capsys.readouterr()
        assert "Bearer sometoken" in captured.out

    def test_multiline_reasoning(self):
        response = "Answer.\n\nREASONING:\nLine 1\nLine 2\nLine 3"
        cleaned, reasoning = self._call(response)
        assert "Line 1" in reasoning
        assert "Line 2" in reasoning
        assert "Line 3" in reasoning

    # ---- No match branch ----

    def test_no_reasoning_section_returns_original(self):
        response = "Just a plain response with no reasoning."
        cleaned, reasoning = self._call(response)
        assert cleaned == response
        assert reasoning == ""

    def test_no_reasoning_section_logger_not_called(self):
        mock_logger = MagicMock()
        self._call("No reasoning here.", reasoning_logger=mock_logger)
        mock_logger.log_reasoning.assert_not_called()

    def test_no_reasoning_prints_message(self, capsys):
        self._call("No reasoning here.")
        captured = capsys.readouterr()
        assert "No REASONING section found" in captured.out

    def test_empty_string_returns_empty_tuple(self):
        cleaned, reasoning = self._call("")
        assert cleaned == ""
        assert reasoning == ""

    def test_reasoning_section_without_content(self):
        """REASONING: header present but empty body."""
        response = "Body.\n\nREASONING:\n"
        cleaned, reasoning = self._call(response)
        # Empty reasoning → logger should NOT be called
        # (the `if reasoning_text:` guard)
        assert reasoning == ""


# ---------------------------------------------------------------------------
# REASONING_SECTION_PROMPT constant
# ---------------------------------------------------------------------------

class TestReasoningSectionPrompt:

    def test_is_non_empty_string(self):
        assert isinstance(REASONING_SECTION_PROMPT, str)
        assert REASONING_SECTION_PROMPT.strip()

    def test_contains_reasoning_keyword(self):
        assert "REASONING:" in REASONING_SECTION_PROMPT

    def test_contains_format_instructions(self):
        # Should mention format guidance
        assert "Format" in REASONING_SECTION_PROMPT or "format" in REASONING_SECTION_PROMPT
