"""Tests for the ToolEnricher pipeline orchestrator."""
import copy
from typing import Any

import pytest

from enrichment.config import EnrichmentConfig
from enrichment.engine import ToolEnricher
from enrichment.structural.base import PatternApplicator
from enrichment.behavioral.base import BehavioralEnricher


class StubPattern(PatternApplicator):
    """A test pattern that adds a marker to prove it ran."""

    def __init__(self, number: int, marker: str) -> None:
        self._number = number
        self._marker = marker

    @property
    def pattern_number(self) -> int:
        return self._number

    @property
    def name(self) -> str:
        return f"StubPattern-{self._number}"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool[self._marker] = True
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        issues = []
        for tool in tools:
            if self._marker not in tool:
                issues.append({
                    "tool": tool.get("name", "<unnamed>"),
                    "issue": f"Missing {self._marker}",
                    "severity": "warning",
                })
        return issues


class StubBehavioral(BehavioralEnricher):
    """A test behavioral enricher that adds a marker."""

    def __init__(self, dim: str) -> None:
        self._dim = dim

    @property
    def dimension(self) -> str:
        return self._dim

    def enrich(self, tool: dict[str, Any]) -> dict[str, Any]:
        enriched = copy.deepcopy(tool)
        enriched[f"behavioral_{self._dim}"] = True
        return enriched


class TestToolEnricher:
    """Tests for the ToolEnricher pipeline."""

    def test_passthrough_mode_returns_tools_unchanged(self, sample_tools: list[dict[str, Any]]) -> None:
        """Passthrough mode returns the exact same list object (no processing)."""
        enricher = ToolEnricher(
            structural_patterns=[StubPattern(1, "p1_applied")],
            behavioral_enrichers=[StubBehavioral("what")],
            config=EnrichmentConfig(mode="passthrough"),
        )
        result = enricher.enrich(sample_tools)
        # Should be the exact same object — no deep copy in passthrough
        assert result is sample_tools
        # No enrichment applied
        assert "p1_applied" not in result[0]
        assert "behavioral_what" not in result[0]

    def test_structural_only_skips_behavioral(self, sample_tools: list[dict[str, Any]]) -> None:
        """Structural mode applies patterns but skips behavioral enrichment."""
        enricher = ToolEnricher(
            structural_patterns=[StubPattern(1, "p1_applied")],
            behavioral_enrichers=[StubBehavioral("what")],
            config=EnrichmentConfig(mode="structural"),
        )
        result = enricher.enrich(sample_tools)
        assert result[0]["p1_applied"] is True
        assert "behavioral_what" not in result[0]

    def test_full_mode_applies_both_layers(self, sample_tools: list[dict[str, Any]]) -> None:
        """Full mode applies both structural patterns and behavioral enrichment."""
        enricher = ToolEnricher(
            structural_patterns=[StubPattern(1, "p1_applied")],
            behavioral_enrichers=[StubBehavioral("what"), StubBehavioral("why")],
            config=EnrichmentConfig(mode="full"),
        )
        result = enricher.enrich(sample_tools)
        assert result[0]["p1_applied"] is True
        assert result[0]["behavioral_what"] is True
        assert result[0]["behavioral_why"] is True

    def test_enrichment_does_not_mutate_input(self, sample_tools: list[dict[str, Any]]) -> None:
        """The enrich method must not modify the original tools list."""
        original = copy.deepcopy(sample_tools)
        enricher = ToolEnricher(
            structural_patterns=[StubPattern(1, "p1_applied")],
            behavioral_enrichers=[StubBehavioral("what")],
            config=EnrichmentConfig(mode="full"),
        )
        enricher.enrich(sample_tools)
        assert sample_tools == original

    def test_disabled_pattern_is_skipped(self, sample_tools: list[dict[str, Any]]) -> None:
        """Patterns not in enabled_structural are skipped."""
        enricher = ToolEnricher(
            structural_patterns=[
                StubPattern(1, "p1_applied"),
                StubPattern(99, "p99_applied"),
            ],
            behavioral_enrichers=[],
            config=EnrichmentConfig(
                mode="structural",
                enabled_structural=frozenset({1}),  # Only pattern 1 enabled
            ),
        )
        result = enricher.enrich(sample_tools)
        assert result[0]["p1_applied"] is True
        assert "p99_applied" not in result[0]

    def test_validate_collects_all_issues(self, sample_tools: list[dict[str, Any]]) -> None:
        """Validate aggregates issues from all enabled patterns."""
        enricher = ToolEnricher(
            structural_patterns=[
                StubPattern(1, "p1_marker"),
                StubPattern(2, "p2_marker"),
            ],
            behavioral_enrichers=[],
            config=EnrichmentConfig(
                mode="structural",
                enabled_structural=frozenset({1, 2}),
            ),
        )
        issues = enricher.validate(sample_tools)
        # 3 tools x 2 patterns = 6 issues (none have the markers)
        assert len(issues) == 6

    def test_empty_tools_list(self) -> None:
        """Enriching an empty tools list returns an empty list."""
        enricher = ToolEnricher(
            structural_patterns=[StubPattern(1, "p1_applied")],
            behavioral_enrichers=[StubBehavioral("what")],
            config=EnrichmentConfig(mode="full"),
        )
        result = enricher.enrich([])
        assert result == []

    def test_disabled_behavioral_dimension(self, sample_tools: list[dict[str, Any]]) -> None:
        """Disabled behavioral dimensions are skipped even in full mode."""
        enricher = ToolEnricher(
            structural_patterns=[],
            behavioral_enrichers=[
                StubBehavioral("what"),
                StubBehavioral("why"),
                StubBehavioral("when"),
            ],
            config=EnrichmentConfig(
                mode="full",
                enable_what=True,
                enable_why=False,
                enable_when=True,
            ),
        )
        result = enricher.enrich(sample_tools)
        assert result[0]["behavioral_what"] is True
        assert "behavioral_why" not in result[0]
        assert result[0]["behavioral_when"] is True

    def test_multiple_patterns_applied_in_order(self, sample_tools: list[dict[str, Any]]) -> None:
        """Structural patterns are applied in the order they appear in the list."""
        enricher = ToolEnricher(
            structural_patterns=[
                StubPattern(1, "first"),
                StubPattern(2, "second"),
                StubPattern(3, "third"),
            ],
            behavioral_enrichers=[],
            config=EnrichmentConfig(
                mode="structural",
                enabled_structural=frozenset({1, 2, 3}),
            ),
        )
        result = enricher.enrich(sample_tools)
        assert result[0]["first"] is True
        assert result[0]["second"] is True
        assert result[0]["third"] is True
