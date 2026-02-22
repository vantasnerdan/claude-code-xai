"""Tests for Pattern 3: Error Format applicator."""
import copy
from typing import Any

import pytest

from enrichment.structural.errors import ErrorFormatApplicator


class TestErrorFormatApplicator:
    """Tests for error format enrichment."""

    @pytest.fixture
    def applicator(self) -> ErrorFormatApplicator:
        """Fresh error format applicator."""
        return ErrorFormatApplicator()

    def test_pattern_metadata(self, applicator: ErrorFormatApplicator) -> None:
        """Pattern number and name are correct."""
        assert applicator.pattern_number == 3
        assert applicator.name == "Error Format"

    def test_adds_error_format_to_known_tools(self, applicator: ErrorFormatApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Known tools get _error_format with common errors and suggestions."""
        result = applicator.apply(sample_tools)
        edit_tool = next(t for t in result if t["name"] == "Edit")
        assert "_error_format" in edit_tool
        assert "errors" in edit_tool["_error_format"]
        assert len(edit_tool["_error_format"]["errors"]) > 0

    def test_error_includes_suggestion(self, applicator: ErrorFormatApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Each error entry includes a suggestion for recovery."""
        result = applicator.apply(sample_tools)
        read_tool = next(t for t in result if t["name"] == "Read")
        for error in read_tool["_error_format"]["errors"]:
            assert "error" in error
            assert "suggestion" in error

    def test_does_not_mutate_input(self, applicator: ErrorFormatApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Apply must not modify the input tools list."""
        original = copy.deepcopy(sample_tools)
        applicator.apply(sample_tools)
        assert sample_tools == original

    def test_validate_reports_missing_error_format(self, applicator: ErrorFormatApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Validate reports tools that should have error format but don't."""
        issues = applicator.validate(sample_tools)
        # Read, Edit, Bash have known errors; all 3 should be reported
        tool_names = [i["tool"] for i in issues]
        assert "Read" in tool_names
        assert "Edit" in tool_names
        assert "Bash" in tool_names
