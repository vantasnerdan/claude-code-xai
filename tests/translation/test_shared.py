"""Tests for shared translation utilities."""

import pytest

from translation.shared import flatten_system


class TestFlattenSystem:
    """Tests for the shared flatten_system() function."""

    def test_string_passthrough(self):
        assert flatten_system("Hello world.") == "Hello world."

    def test_empty_string(self):
        assert flatten_system("") == ""

    def test_single_text_block(self):
        system = [{"type": "text", "text": "You are a coding assistant."}]
        assert flatten_system(system) == "You are a coding assistant."

    def test_multiple_text_blocks(self):
        system = [
            {"type": "text", "text": "First block."},
            {"type": "text", "text": "Second block."},
        ]
        assert flatten_system(system) == "First block.\n\nSecond block."

    def test_non_text_blocks_skipped(self):
        system = [
            {"type": "text", "text": "Content here."},
            {"type": "cache_control", "data": "ephemeral"},
        ]
        assert flatten_system(system) == "Content here."

    def test_empty_list(self):
        assert flatten_system([]) == ""

    def test_empty_text_skipped(self):
        system = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Instructions."},
        ]
        assert flatten_system(system) == "Instructions."

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected str or list"):
            flatten_system(123)  # type: ignore[arg-type]
