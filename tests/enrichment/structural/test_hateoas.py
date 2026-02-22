"""Tests for Pattern 2: HATEOAS applicator."""
import copy
from typing import Any

import pytest

from enrichment.structural.hateoas import HateoasApplicator


class TestHateoasApplicator:
    """Tests for HATEOAS link enrichment."""

    @pytest.fixture
    def applicator(self) -> HateoasApplicator:
        """Fresh HATEOAS applicator."""
        return HateoasApplicator()

    def test_pattern_metadata(self, applicator: HateoasApplicator) -> None:
        """Pattern number and name are correct."""
        assert applicator.pattern_number == 2
        assert applicator.name == "HATEOAS"

    def test_adds_links_to_known_tools(self, applicator: HateoasApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Known tools get _links with related tools and error recovery."""
        result = applicator.apply(sample_tools)
        read_tool = next(t for t in result if t["name"] == "Read")
        assert "_links" in read_tool
        assert "related" in read_tool["_links"]
        assert "Edit" in read_tool["_links"]["related"]

    def test_error_recovery_links(self, applicator: HateoasApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Edit tool includes error recovery links for common failures."""
        result = applicator.apply(sample_tools)
        edit_tool = next(t for t in result if t["name"] == "Edit")
        assert "on_error" in edit_tool["_links"]
        assert "old_string_not_found" in edit_tool["_links"]["on_error"]

    def test_does_not_mutate_input(self, applicator: HateoasApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Apply must not modify the input tools list."""
        original = copy.deepcopy(sample_tools)
        applicator.apply(sample_tools)
        assert sample_tools == original

    def test_unknown_tool_gets_no_links(self, applicator: HateoasApplicator, unknown_tool: dict[str, Any]) -> None:
        """Tools not in the link database don't get _links added."""
        result = applicator.apply([unknown_tool])
        assert "_links" not in result[0]

    def test_validate_reports_missing_links(self, applicator: HateoasApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Validate reports tools that lack _links."""
        issues = applicator.validate(sample_tools)
        assert len(issues) == 3  # None of the sample tools have _links
        assert all(i["severity"] == "warning" for i in issues)
