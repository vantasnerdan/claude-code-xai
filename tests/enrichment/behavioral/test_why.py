"""Tests for WHY dimension behavioral enricher."""
import copy
from typing import Any

import pytest

from enrichment.behavioral.why_enricher import WhyEnricher


class TestWhyEnricher:
    """Tests for the WHY behavioral dimension."""

    @pytest.fixture
    def enricher(self) -> WhyEnricher:
        """Fresh WHY enricher."""
        return WhyEnricher()

    def test_dimension_is_why(self, enricher: WhyEnricher) -> None:
        """Dimension property returns 'why'."""
        assert enricher.dimension == "why"

    def test_adds_problem_context(self, enricher: WhyEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Known tools get behavioral_why with problem_context."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        result = enricher.enrich(read_tool)
        assert "behavioral_why" in result
        assert "problem_context" in result["behavioral_why"]
        assert isinstance(result["behavioral_why"]["problem_context"], str)

    def test_adds_failure_modes(self, enricher: WhyEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Known tools get behavioral_why with failure_modes list."""
        edit_tool = next(t for t in sample_tools if t["name"] == "Edit")
        result = enricher.enrich(edit_tool)
        assert "failure_modes" in result["behavioral_why"]
        assert isinstance(result["behavioral_why"]["failure_modes"], list)
        assert len(result["behavioral_why"]["failure_modes"]) > 0

    def test_does_not_mutate_input(self, enricher: WhyEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Enrich must not modify the original tool dict."""
        edit_tool = next(t for t in sample_tools if t["name"] == "Edit")
        original = copy.deepcopy(edit_tool)
        enricher.enrich(edit_tool)
        assert edit_tool == original

    def test_unknown_tool_gets_no_why(self, enricher: WhyEnricher, unknown_tool: dict[str, Any]) -> None:
        """Tools not in the knowledge base don't get behavioral_why."""
        result = enricher.enrich(unknown_tool)
        assert "behavioral_why" not in result
