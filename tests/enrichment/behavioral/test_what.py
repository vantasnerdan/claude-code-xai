"""Tests for WHAT dimension behavioral enricher."""
import copy
from typing import Any

import pytest

from enrichment.behavioral.what_enricher import WhatEnricher


class TestWhatEnricher:
    """Tests for the WHAT behavioral dimension."""

    @pytest.fixture
    def enricher(self, structure_data) -> WhatEnricher:
        """Fresh WHAT enricher using YAML data."""
        b = structure_data.get("behavioral", {})
        what_tools = b.get("what", {}).get("tools", {})
        return WhatEnricher(tool_data=what_tools)

    def test_dimension_is_what(self, enricher: WhatEnricher) -> None:
        """Dimension property returns 'what'."""
        assert enricher.dimension == "what"

    def test_adds_enhanced_description(self, enricher: WhatEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Known tools get a behavioral_what with enhanced description."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        result = enricher.enrich(read_tool)
        assert "behavioral_what" in result
        assert isinstance(result["behavioral_what"], str)
        assert len(result["behavioral_what"]) > len(read_tool["description"])

    def test_preserves_original_description(self, enricher: WhatEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Original description field is preserved unchanged."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        original_desc = read_tool["description"]
        result = enricher.enrich(read_tool)
        assert result["description"] == original_desc

    def test_does_not_mutate_input(self, enricher: WhatEnricher, sample_tools: list[dict[str, Any]]) -> None:
        """Enrich must not modify the original tool dict."""
        read_tool = next(t for t in sample_tools if t["name"] == "Read")
        original = copy.deepcopy(read_tool)
        enricher.enrich(read_tool)
        assert read_tool == original

    def test_unknown_tool_gets_no_what(self, enricher: WhatEnricher, unknown_tool: dict[str, Any]) -> None:
        """Tools not in the knowledge base don't get behavioral_what."""
        result = enricher.enrich(unknown_tool)
        assert "behavioral_what" not in result
