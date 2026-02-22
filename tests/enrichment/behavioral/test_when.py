"""Tests for WHEN dimension behavioral enricher.

The WHEN dimension is critical — it teaches prerequisite ordering,
alternative tools, and sequencing that RL-trained models learn implicitly
but untrained models need explicitly.
"""
import copy
from typing import Any

import pytest

from enrichment.behavioral.when_enricher import WhenEnricher


class TestWhenEnricher:
    """Tests for the WHEN behavioral dimension."""

    @pytest.fixture
    def enricher(self) -> WhenEnricher:
        """Fresh WHEN enricher."""
        return WhenEnricher()

    def test_dimension_is_when(self, enricher: WhenEnricher) -> None:
        """Dimension property returns 'when'."""
        assert enricher.dimension == "when"

    def test_adds_prerequisites_to_tool(self, enricher: WhenEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Edit tool should get prerequisites (must Read first)."""
        edit_tool = next(t for t in sample_tools if t["name"] == "Edit")
        result = enricher.enrich(edit_tool)
        assert "behavioral_when" in result
        assert "prerequisites" in result["behavioral_when"]
        assert "Read" in result["behavioral_when"]["prerequisites"]

    def test_adds_alternatives_to_tool(self, enricher: WhenEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Read tool should list what it replaces (Bash cat, etc.)."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        result = enricher.enrich(read_tool)
        assert "use_instead_of" in result["behavioral_when"]
        assert "Bash cat" in result["behavioral_when"]["use_instead_of"]

    def test_adds_sequencing_to_tool(self, enricher: WhenEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Tools should get sequencing guidance."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        result = enricher.enrich(read_tool)
        assert "sequencing" in result["behavioral_when"]
        assert isinstance(result["behavioral_when"]["sequencing"], str)
        assert len(result["behavioral_when"]["sequencing"]) > 0

    def test_unknown_tool_gets_no_behavioral_enrichment(self, enricher: WhenEnricher, unknown_tool: dict[str, Any]) -> None:
        """Tools not in the knowledge base don't get behavioral_when."""
        result = enricher.enrich(unknown_tool)
        assert "behavioral_when" not in result

    def test_does_not_mutate_input(self, enricher: WhenEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Enrich must not modify the original tool dict."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        original = copy.deepcopy(read_tool)
        enricher.enrich(read_tool)
        assert read_tool == original

    def test_bash_has_do_not_use_for(self, enricher: WhenEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Bash tool should have do_not_use_for listing file operations."""
        bash_tool = next(t for t in sample_tools if t["name"] == "Bash")
        result = enricher.enrich(bash_tool)
        assert "do_not_use_for" in result["behavioral_when"]
        assert len(result["behavioral_when"]["do_not_use_for"]) > 0

    def test_read_use_before_edit(self, enricher: WhenEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Read tool should list Edit and Write in use_before."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        result = enricher.enrich(read_tool)
        assert "Edit" in result["behavioral_when"]["use_before"]
        assert "Write" in result["behavioral_when"]["use_before"]
