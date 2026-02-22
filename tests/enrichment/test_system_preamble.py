"""Tests for enrichment.system_preamble module.

Covers: preamble content, env-based toggle, injection into message lists,
and integration with TranslationConfig.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from enrichment.system_preamble import (
    get_system_preamble,
    inject_system_preamble,
    _PREAMBLE,
)


class TestGetSystemPreamble:
    """Tests for get_system_preamble()."""

    def test_returns_preamble_by_default(self) -> None:
        """Preamble is returned when PREAMBLE_ENABLED is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PREAMBLE_ENABLED", None)
            result = get_system_preamble()
        assert result == _PREAMBLE
        assert len(result) > 0

    def test_returns_preamble_when_enabled_true(self) -> None:
        """Preamble is returned when PREAMBLE_ENABLED=true."""
        with patch.dict(os.environ, {"PREAMBLE_ENABLED": "true"}):
            result = get_system_preamble()
        assert result == _PREAMBLE

    def test_returns_empty_when_disabled(self) -> None:
        """Empty string when PREAMBLE_ENABLED=false."""
        with patch.dict(os.environ, {"PREAMBLE_ENABLED": "false"}):
            result = get_system_preamble()
        assert result == ""

    def test_case_insensitive_false(self) -> None:
        """PREAMBLE_ENABLED=False (capital F) also disables."""
        with patch.dict(os.environ, {"PREAMBLE_ENABLED": "False"}):
            result = get_system_preamble()
        assert result == ""

    def test_preamble_contains_tool_hierarchy(self) -> None:
        """Preamble covers tool preference hierarchy."""
        assert "Tool Preference Hierarchy" in _PREAMBLE
        assert "Read" in _PREAMBLE
        assert "Grep" in _PREAMBLE
        assert "Glob" in _PREAMBLE
        assert "Edit" in _PREAMBLE

    def test_preamble_contains_sequencing_rules(self) -> None:
        """Preamble covers sequencing rules."""
        assert "Sequencing Rules" in _PREAMBLE
        assert "Read a file BEFORE editing" in _PREAMBLE

    def test_preamble_contains_chaining_patterns(self) -> None:
        """Preamble covers tool chaining patterns."""
        assert "Tool Chaining Patterns" in _PREAMBLE
        assert "Discovery" in _PREAMBLE
        assert "Modification" in _PREAMBLE

    def test_preamble_contains_parallel_rules(self) -> None:
        """Preamble covers parallel vs sequential execution."""
        assert "Parallel vs Sequential" in _PREAMBLE
        assert "INDEPENDENT" in _PREAMBLE
        assert "DEPENDENT" in _PREAMBLE

    def test_preamble_contains_safety_patterns(self) -> None:
        """Preamble covers safety patterns."""
        assert "Safety Patterns" in _PREAMBLE
        assert "force push" in _PREAMBLE
        assert "credentials" in _PREAMBLE

    def test_preamble_contains_output_conventions(self) -> None:
        """Preamble covers output conventions."""
        assert "Output Conventions" in _PREAMBLE
        assert "source of truth" in _PREAMBLE


class TestInjectSystemPreamble:
    """Tests for inject_system_preamble()."""

    def test_prepends_to_existing_system_message(self) -> None:
        """Preamble is prepended to existing system message content."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = inject_system_preamble(messages, "PREAMBLE")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "PREAMBLE\n\nYou are helpful."
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_inserts_system_message_when_none_exists(self) -> None:
        """System message is created when messages have no system role."""
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_system_preamble(messages, "PREAMBLE")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "PREAMBLE"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_empty_preamble_returns_unchanged(self) -> None:
        """Empty preamble string returns messages as-is."""
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_system_preamble(messages, "")
        assert result == messages

    def test_does_not_mutate_input(self) -> None:
        """Original message list is not modified."""
        messages = [
            {"role": "system", "content": "Original"},
            {"role": "user", "content": "Hello"},
        ]
        original_content = messages[0]["content"]
        inject_system_preamble(messages, "PREAMBLE")
        assert messages[0]["content"] == original_content

    def test_empty_system_content_uses_preamble_only(self) -> None:
        """System message with empty content gets preamble as content."""
        messages = [{"role": "system", "content": ""}]
        result = inject_system_preamble(messages, "PREAMBLE")
        assert result[0]["content"] == "PREAMBLE"

    def test_empty_messages_list(self) -> None:
        """Empty message list gets a system message inserted."""
        result = inject_system_preamble([], "PREAMBLE")
        assert len(result) == 1
        assert result[0] == {"role": "system", "content": "PREAMBLE"}

    def test_preserves_extra_system_fields(self) -> None:
        """Extra fields on the system message dict are preserved."""
        messages = [
            {"role": "system", "content": "Original", "name": "agent"},
        ]
        result = inject_system_preamble(messages, "PREAMBLE")
        assert result[0]["name"] == "agent"
        assert result[0]["content"] == "PREAMBLE\n\nOriginal"

    def test_non_system_first_message_preserved(self) -> None:
        """When first message is not system, it shifts to index 1."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = inject_system_preamble(messages, "PREAMBLE")
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


class TestIntegrationWithConfig:
    """Test that TranslationConfig picks up the preamble."""

    def test_config_loads_preamble_by_default(self) -> None:
        """TranslationConfig.system_prompt_preamble is populated."""
        from translation.config import TranslationConfig

        config = TranslationConfig()
        assert len(config.system_prompt_preamble) > 0
        assert "Tool Preference Hierarchy" in config.system_prompt_preamble

    def test_config_empty_when_disabled(self) -> None:
        """Config preamble is empty when PREAMBLE_ENABLED=false."""
        with patch.dict(os.environ, {"PREAMBLE_ENABLED": "false"}):
            from translation.config import TranslationConfig

            config = TranslationConfig()
        assert config.system_prompt_preamble == ""

    def test_forward_translation_includes_preamble(self) -> None:
        """anthropic_to_openai() includes preamble in system message."""
        from translation.forward import anthropic_to_openai

        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are a coding assistant.",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        }
        result = anthropic_to_openai(request)
        system_msg = result["messages"][0]
        assert system_msg["role"] == "system"
        assert "Tool Preference Hierarchy" in system_msg["content"]
        assert "You are a coding assistant." in system_msg["content"]
