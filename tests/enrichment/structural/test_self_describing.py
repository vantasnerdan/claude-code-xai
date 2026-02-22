"""Tests for Pattern 6: Self-Describing schemas applicator.

Pattern 6 is the most important structural pattern — it ensures every tool
has explicit input and output schemas so the model knows exactly what to
send and what to expect back.
"""
import copy
from typing import Any

import pytest

from enrichment.structural.self_describing import SelfDescribingApplicator


class TestSelfDescribingApplicator:
    """Tests for self-describing schema enrichment."""

    @pytest.fixture
    def applicator(self) -> SelfDescribingApplicator:
        """Fresh self-describing applicator."""
        return SelfDescribingApplicator()

    def test_pattern_metadata(self, applicator: SelfDescribingApplicator) -> None:
        """Pattern number and name are correct."""
        assert applicator.pattern_number == 6
        assert applicator.name == "Self-Describing"

    def test_adds_input_schema_if_missing(self, applicator: SelfDescribingApplicator) -> None:
        """Tools without any input schema get a stub inputSchema added."""
        tools = [{"name": "NoSchema", "description": "A tool without schema"}]
        result = applicator.apply(tools)
        assert "inputSchema" in result[0]
        assert result[0]["inputSchema"]["type"] == "object"

    def test_preserves_existing_input_schema(self, applicator: SelfDescribingApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Tools with existing input_schema keep it unchanged."""
        result = applicator.apply(sample_tools)
        read_tool = next(t for t in result if t["name"] == "Read")
        # Original input_schema should still be present
        assert "input_schema" in read_tool
        assert "file_path" in read_tool["input_schema"]["properties"]

    def test_adds_output_schema_for_known_tools(self, applicator: SelfDescribingApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Known tools get an outputSchema added."""
        result = applicator.apply(sample_tools)
        read_tool = next(t for t in result if t["name"] == "Read")
        assert "outputSchema" in read_tool
        assert "content" in read_tool["outputSchema"]["properties"]

    def test_validates_missing_schema(self, applicator: SelfDescribingApplicator) -> None:
        """Validate reports tools missing inputSchema as errors."""
        tools = [{"name": "NoSchema", "description": "A tool without schema"}]
        issues = applicator.validate(tools)
        input_issues = [i for i in issues if "inputSchema" in i["issue"]]
        assert len(input_issues) == 1
        assert input_issues[0]["severity"] == "error"

    def test_validates_missing_output_schema(self, applicator: SelfDescribingApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Validate reports tools missing outputSchema as warnings."""
        issues = applicator.validate(sample_tools)
        output_issues = [i for i in issues if "outputSchema" in i["issue"]]
        assert len(output_issues) == 3  # All 3 sample tools lack outputSchema
        assert all(i["severity"] == "warning" for i in output_issues)

    def test_does_not_mutate_input(self, applicator: SelfDescribingApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Apply must not modify the input tools list."""
        original = copy.deepcopy(sample_tools)
        applicator.apply(sample_tools)
        assert sample_tools == original

    def test_output_schema_bash_includes_exit_code(self, applicator: SelfDescribingApplicator, sample_tools: list[dict[str, Any]]) -> None:
        """Bash output schema includes stdout, stderr, and exit_code."""
        result = applicator.apply(sample_tools)
        bash_tool = next(t for t in result if t["name"] == "Bash")
        assert "outputSchema" in bash_tool
        props = bash_tool["outputSchema"]["properties"]
        assert "stdout" in props
        assert "stderr" in props
        assert "exit_code" in props
