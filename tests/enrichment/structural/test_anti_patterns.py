"""Tests for Pattern 14: Anti-Patterns applicator."""
import copy
from typing import Any

import pytest

from enrichment.structural.anti_patterns import AntiPatternsApplicator


class TestAntiPatternsApplicator:
    """Tests for anti-pattern enrichment."""

    @pytest.fixture
    def applicator(self) -> AntiPatternsApplicator:
        """Fresh anti-patterns applicator."""
        return AntiPatternsApplicator()

    def test_pattern_metadata(self, applicator: AntiPatternsApplicator) -> None:
        """Pattern number and name are correct."""
        assert applicator.pattern_number == 14
        assert applicator.name == "Anti-Patterns"

    def test_adds_anti_patterns_to_known_tools(self, applicator: AntiPatternsApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Known tools get _anti_patterns with documented failure modes."""
        result = applicator.apply(sample_tools)
        edit_tool = next(t for t in result if t["name"] == "Edit")
        assert "_anti_patterns" in edit_tool
        assert len(edit_tool["_anti_patterns"]) > 0

    def test_anti_pattern_structure(self, applicator: AntiPatternsApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Each anti-pattern has anti_pattern, why_bad, and do_instead fields."""
        result = applicator.apply(sample_tools)
        bash_tool = next(t for t in result if t["name"] == "Bash")
        for ap in bash_tool["_anti_patterns"]:
            assert "anti_pattern" in ap
            assert "why_bad" in ap
            assert "do_instead" in ap

    def test_does_not_mutate_input(self, applicator: AntiPatternsApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Apply must not modify the input tools list."""
        original = copy.deepcopy(sample_tools)
        applicator.apply(sample_tools)
        assert sample_tools == original

    def test_unknown_tool_gets_no_anti_patterns(self, applicator: AntiPatternsApplicator, unknown_tool: dict[str, Any]) -> None:
        """Tools not in the anti-pattern database don't get _anti_patterns added."""
        result = applicator.apply([unknown_tool])
        assert "_anti_patterns" not in result[0]

    def test_validate_reports_missing_anti_patterns(self, applicator: AntiPatternsApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Validate reports tools that should have anti-patterns but don't."""
        issues = applicator.validate(sample_tools)
        tool_names = [i["tool"] for i in issues]
        assert "Read" in tool_names
        assert "Edit" in tool_names
        assert "Bash" in tool_names
