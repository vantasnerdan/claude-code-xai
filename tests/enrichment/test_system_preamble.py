"""Tests for enrichment.system_preamble module.

Covers: identity preamble, behavioral preamble, env-based toggles,
Anthropic identity stripping, injection into message lists,
and integration with TranslationConfig.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from enrichment.system_preamble import (
    get_system_preamble,
    inject_system_preamble,
    strip_anthropic_identity,
    _PREAMBLE,
    _IDENTITY_PREAMBLE,
)


class TestIdentityPreambleContent:
    """Tests for identity preamble content."""

    def test_identity_declares_grok(self) -> None:
        """Identity preamble asserts Grok identity."""
        assert "You are Grok" in _IDENTITY_PREAMBLE
        assert "xAI" in _IDENTITY_PREAMBLE

    def test_identity_disclaims_claude(self) -> None:
        """Identity preamble tells model to disregard Claude claims."""
        assert "Disregard" in _IDENTITY_PREAMBLE
        assert "identity claims" in _IDENTITY_PREAMBLE

    def test_identity_preserves_tool_conventions(self) -> None:
        """Identity preamble frames tool conventions as environment rules."""
        assert "environment conventions" in _IDENTITY_PREAMBLE

    def test_identity_truthful_response(self) -> None:
        """Identity preamble directs truthful model identification."""
        assert "respond truthfully" in _IDENTITY_PREAMBLE
        assert "Grok by xAI" in _IDENTITY_PREAMBLE


class TestGetSystemPreamble:
    """Tests for get_system_preamble()."""

    def test_returns_identity_and_behavioral_by_default(self) -> None:
        """Both identity and behavioral preamble returned by default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PREAMBLE_ENABLED", None)
            os.environ.pop("IDENTITY_ENABLED", None)
            result = get_system_preamble()
        assert _IDENTITY_PREAMBLE in result
        assert _PREAMBLE in result
        assert result.index(_IDENTITY_PREAMBLE) < result.index(_PREAMBLE)

    def test_identity_before_behavioral(self) -> None:
        """Identity section appears before behavioral conventions."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PREAMBLE_ENABLED", None)
            os.environ.pop("IDENTITY_ENABLED", None)
            result = get_system_preamble()
        assert result.startswith("# Identity")

    def test_preamble_disabled_returns_identity_only(self) -> None:
        """When PREAMBLE_ENABLED=false, only identity is returned."""
        with patch.dict(os.environ, {"PREAMBLE_ENABLED": "false"}, clear=False):
            os.environ.pop("IDENTITY_ENABLED", None)
            result = get_system_preamble()
        assert "You are Grok" in result
        assert "Tool Preference Hierarchy" not in result

    def test_identity_disabled_returns_behavioral_only(self) -> None:
        """When IDENTITY_ENABLED=false, only behavioral conventions returned."""
        with patch.dict(os.environ, {"IDENTITY_ENABLED": "false"}, clear=False):
            os.environ.pop("PREAMBLE_ENABLED", None)
            result = get_system_preamble()
        assert result == _PREAMBLE
        assert "You are Grok" not in result

    def test_both_disabled_returns_empty(self) -> None:
        """Empty string when both PREAMBLE_ENABLED and IDENTITY_ENABLED are false."""
        with patch.dict(os.environ, {
            "PREAMBLE_ENABLED": "false",
            "IDENTITY_ENABLED": "false",
        }):
            result = get_system_preamble()
        assert result == ""

    def test_both_enabled_explicitly(self) -> None:
        """Explicit true for both returns full preamble."""
        with patch.dict(os.environ, {
            "PREAMBLE_ENABLED": "true",
            "IDENTITY_ENABLED": "true",
        }):
            result = get_system_preamble()
        assert "You are Grok" in result
        assert "Tool Preference Hierarchy" in result

    def test_case_insensitive_false(self) -> None:
        """PREAMBLE_ENABLED=False (capital F) also disables."""
        with patch.dict(os.environ, {
            "PREAMBLE_ENABLED": "False",
            "IDENTITY_ENABLED": "False",
        }):
            result = get_system_preamble()
        assert result == ""

    def test_preamble_contains_tool_hierarchy(self) -> None:
        """Behavioral preamble covers tool preference hierarchy."""
        assert "Tool Preference Hierarchy" in _PREAMBLE
        assert "Read" in _PREAMBLE
        assert "Grep" in _PREAMBLE
        assert "Glob" in _PREAMBLE
        assert "Edit" in _PREAMBLE

    def test_preamble_contains_sequencing_rules(self) -> None:
        """Behavioral preamble covers sequencing rules."""
        assert "Sequencing Rules" in _PREAMBLE
        assert "Read a file BEFORE editing" in _PREAMBLE

    def test_preamble_contains_chaining_patterns(self) -> None:
        """Behavioral preamble covers tool chaining patterns."""
        assert "Tool Chaining Patterns" in _PREAMBLE
        assert "Discovery" in _PREAMBLE
        assert "Modification" in _PREAMBLE

    def test_preamble_contains_parallel_rules(self) -> None:
        """Behavioral preamble covers parallel vs sequential execution."""
        assert "Parallel vs Sequential" in _PREAMBLE
        assert "INDEPENDENT" in _PREAMBLE
        assert "DEPENDENT" in _PREAMBLE

    def test_preamble_contains_safety_patterns(self) -> None:
        """Behavioral preamble covers safety patterns."""
        assert "Safety Patterns" in _PREAMBLE
        assert "force push" in _PREAMBLE
        assert "credentials" in _PREAMBLE

    def test_preamble_contains_output_conventions(self) -> None:
        """Behavioral preamble covers output conventions."""
        assert "Output Conventions" in _PREAMBLE
        assert "source of truth" in _PREAMBLE


class TestStripAnthropicIdentity:
    """Tests for strip_anthropic_identity()."""

    def test_strips_powered_by_model_named(self) -> None:
        """Strips 'You are powered by the model named ...' pattern."""
        text = (
            "Some context. "
            "You are powered by the model named Opus 4.6. "
            "The exact model ID is claude-opus-4-6. "
            "More context."
        )
        result = strip_anthropic_identity(text)
        assert "powered by the model named" not in result
        assert "model ID" not in result
        assert "Some context." in result
        assert "More context." in result

    def test_strips_powered_by_claude(self) -> None:
        """Strips 'You are powered by Claude Opus 4.6' pattern."""
        text = "Start. You are powered by Claude Opus 4.6. End."
        result = strip_anthropic_identity(text)
        assert "powered by Claude" not in result
        assert "Start." in result
        assert "End." in result

    def test_strips_standalone_model_id(self) -> None:
        """Strips standalone 'The exact model ID is claude-...' pattern."""
        text = "Before. The exact model ID is claude-opus-4-6. After."
        result = strip_anthropic_identity(text)
        assert "exact model ID" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_knowledge_cutoff(self) -> None:
        """Strips 'Assistant knowledge cutoff is ...' pattern."""
        text = "Context. Assistant knowledge cutoff is May 2025. More."
        result = strip_anthropic_identity(text)
        assert "knowledge cutoff" not in result
        assert "Context." in result
        assert "More." in result

    def test_strips_claude_background_info_block(self) -> None:
        """Strips <claude_background_info> XML blocks."""
        text = (
            "Before.\n"
            "<claude_background_info>\n"
            "The most recent frontier Claude model is Claude Opus 4.6.\n"
            "</claude_background_info>\n"
            "After."
        )
        result = strip_anthropic_identity(text)
        assert "claude_background_info" not in result
        assert "frontier Claude model" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_fast_mode_info_block(self) -> None:
        """Strips <fast_mode_info> XML blocks."""
        text = (
            "Before.\n"
            "<fast_mode_info>\n"
            "Fast mode for Claude Code uses the same model.\n"
            "</fast_mode_info>\n"
            "After."
        )
        result = strip_anthropic_identity(text)
        assert "fast_mode_info" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_full_realistic_system_prompt(self) -> None:
        """Strips multiple identity patterns from a realistic system prompt."""
        text = (
            "You are a helpful assistant.\n\n"
            "You are powered by the model named Opus 4.6. "
            "The exact model ID is claude-opus-4-6.\n\n"
            "Assistant knowledge cutoff is May 2025.\n\n"
            "<claude_background_info>\n"
            "The most recent frontier Claude model is Claude Opus 4.6 "
            "(model ID: 'claude-opus-4-6').\n"
            "</claude_background_info>\n\n"
            "<fast_mode_info>\n"
            "Fast mode for Claude Code uses the same Claude Opus 4.6 model "
            "with faster output. It does NOT switch to a different model.\n"
            "</fast_mode_info>\n\n"
            "Follow the user's instructions carefully."
        )
        result = strip_anthropic_identity(text)
        assert "powered by" not in result
        assert "claude-opus" not in result
        assert "knowledge cutoff" not in result
        assert "claude_background_info" not in result
        assert "fast_mode_info" not in result
        assert "You are a helpful assistant." in result
        assert "Follow the user's instructions carefully." in result

    def test_preserves_non_identity_claude_references(self) -> None:
        """Does not strip references to Claude that are not identity claims."""
        text = "Use the Claude Code tool conventions described above."
        result = strip_anthropic_identity(text)
        assert "Claude Code tool conventions" in result

    def test_no_excessive_blank_lines(self) -> None:
        """Stripped content does not leave triple+ blank lines."""
        text = (
            "Start.\n\n"
            "You are powered by Claude Opus 4.6.\n\n\n"
            "End."
        )
        result = strip_anthropic_identity(text)
        assert "\n\n\n" not in result

    def test_empty_string_returns_empty(self) -> None:
        """Empty input returns empty output."""
        assert strip_anthropic_identity("") == ""

    def test_no_match_returns_unchanged(self) -> None:
        """Text with no identity patterns passes through unchanged."""
        text = "Just a normal system prompt with instructions."
        result = strip_anthropic_identity(text)
        assert result == text

    def test_disabled_returns_unchanged(self) -> None:
        """When IDENTITY_ENABLED=false, stripping is skipped."""
        text = "You are powered by Claude Opus 4.6. Be helpful."
        with patch.dict(os.environ, {"IDENTITY_ENABLED": "false"}):
            result = strip_anthropic_identity(text)
        assert result == text

    def test_case_insensitive(self) -> None:
        """Identity stripping is case-insensitive."""
        text = "you are powered by claude opus 4.6. End."
        result = strip_anthropic_identity(text)
        assert "powered by claude" not in result


class TestStripAnthropicIdentityListType:
    """Tests for strip_anthropic_identity() with list-of-content-blocks system field.

    The Anthropic Messages API sends the system field as either a string
    or a list of typed content blocks. Claude Code uses the list format
    for streaming/opus requests.
    """

    def test_list_type_returns_list(self) -> None:
        """List input produces list output."""
        system = [{"type": "text", "text": "Hello world."}]
        result = strip_anthropic_identity(system)
        assert isinstance(result, list)

    def test_list_strips_identity_from_text_blocks(self) -> None:
        """Identity patterns are stripped from text blocks in the list."""
        system = [
            {
                "type": "text",
                "text": (
                    "You are a coding assistant. "
                    "You are powered by Claude Opus 4.6. "
                    "The exact model ID is claude-opus-4-6."
                ),
            }
        ]
        result = strip_anthropic_identity(system)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "powered by Claude" not in result[0]["text"]
        assert "claude-opus-4-6" not in result[0]["text"]
        assert "You are a coding assistant." in result[0]["text"]

    def test_list_removes_empty_text_blocks(self) -> None:
        """Text blocks that become empty after stripping are removed."""
        system = [
            {"type": "text", "text": "You are powered by Claude Opus 4.6."},
            {"type": "text", "text": "Keep this block."},
        ]
        result = strip_anthropic_identity(system)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "Keep this block."

    def test_list_preserves_non_text_blocks(self) -> None:
        """Non-text blocks (e.g., cache_control) are preserved unchanged."""
        system = [
            {"type": "text", "text": "You are powered by Claude Opus 4.6."},
            {"type": "text", "text": "Instructions here.", "cache_control": {"type": "ephemeral"}},
        ]
        result = strip_anthropic_identity(system)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "Instructions here."
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_list_strips_background_info_block(self) -> None:
        """<claude_background_info> is stripped from text blocks in list."""
        system = [
            {
                "type": "text",
                "text": (
                    "Before.\n"
                    "<claude_background_info>\n"
                    "Claude model info here.\n"
                    "</claude_background_info>\n"
                    "After."
                ),
            }
        ]
        result = strip_anthropic_identity(system)
        assert "claude_background_info" not in result[0]["text"]
        assert "Before." in result[0]["text"]
        assert "After." in result[0]["text"]

    def test_list_multiple_text_blocks_stripped_independently(self) -> None:
        """Each text block is stripped independently."""
        system = [
            {"type": "text", "text": "Assistant knowledge cutoff is May 2025."},
            {"type": "text", "text": "You are powered by Claude Opus 4.6."},
            {"type": "text", "text": "Follow the user's instructions."},
        ]
        result = strip_anthropic_identity(system)
        assert len(result) == 1
        assert result[0]["text"] == "Follow the user's instructions."

    def test_list_empty_list_returns_empty_list(self) -> None:
        """Empty list input returns empty list."""
        result = strip_anthropic_identity([])
        assert result == []

    def test_list_all_blocks_stripped_returns_empty_list(self) -> None:
        """If all text blocks become empty, an empty list is returned."""
        system = [
            {"type": "text", "text": "You are powered by Claude Opus 4.6."},
        ]
        result = strip_anthropic_identity(system)
        assert result == []

    def test_list_disabled_returns_unchanged(self) -> None:
        """When IDENTITY_ENABLED=false, list is returned unchanged."""
        system = [
            {"type": "text", "text": "You are powered by Claude Opus 4.6."},
        ]
        with patch.dict(os.environ, {"IDENTITY_ENABLED": "false"}):
            result = strip_anthropic_identity(system)
        assert result == system

    def test_list_realistic_claude_code_system(self) -> None:
        """Realistic Claude Code streaming system with multiple blocks."""
        system = [
            {
                "type": "text",
                "text": (
                    "You are powered by the model named Claude Opus 4.6. "
                    "The exact model ID is claude-opus-4-6.\n\n"
                    "Assistant knowledge cutoff is May 2025.\n\n"
                    "<claude_background_info>\n"
                    "The most recent frontier Claude model is Claude Opus 4.6.\n"
                    "</claude_background_info>\n\n"
                    "<fast_mode_info>\n"
                    "Fast mode uses the same Claude Opus 4.6 model.\n"
                    "</fast_mode_info>"
                ),
            },
            {
                "type": "text",
                "text": "You are an expert coding assistant. Follow all tool conventions.",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        result = strip_anthropic_identity(system)
        assert isinstance(result, list)
        # First block should be stripped entirely (all identity patterns)
        # Second block is preserved with cache_control
        assert len(result) == 1
        assert "expert coding assistant" in result[0]["text"]
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_invalid_type_raises_typeerror(self) -> None:
        """Non-string, non-list input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str or list"):
            strip_anthropic_identity(42)  # type: ignore[arg-type]


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

    def test_config_includes_identity_by_default(self) -> None:
        """TranslationConfig.system_prompt_preamble includes identity."""
        from translation.config import TranslationConfig

        config = TranslationConfig()
        assert "You are Grok" in config.system_prompt_preamble

    def test_config_empty_when_both_disabled(self) -> None:
        """Config preamble is empty when both are disabled."""
        with patch.dict(os.environ, {
            "PREAMBLE_ENABLED": "false",
            "IDENTITY_ENABLED": "false",
        }):
            from translation.config import TranslationConfig

            config = TranslationConfig()
        assert config.system_prompt_preamble == ""

    def test_forward_translation_includes_identity_and_preamble(self) -> None:
        """anthropic_to_openai() includes identity and preamble in system message."""
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
        assert "You are Grok" in system_msg["content"]
        assert "Tool Preference Hierarchy" in system_msg["content"]
        assert "You are a coding assistant." in system_msg["content"]

    def test_forward_translation_strips_claude_identity(self) -> None:
        """anthropic_to_openai() strips Claude identity claims from system text."""
        from translation.forward import anthropic_to_openai

        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": (
                "You are a coding assistant. "
                "You are powered by the model named Opus 4.6. "
                "The exact model ID is claude-opus-4-6."
            ),
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        }
        result = anthropic_to_openai(request)
        system_msg = result["messages"][0]
        assert "powered by the model named" not in system_msg["content"]
        assert "claude-opus-4-6" not in system_msg["content"]
        assert "You are a coding assistant." in system_msg["content"]
        assert "You are Grok" in system_msg["content"]
