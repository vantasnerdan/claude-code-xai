"""Tests for factory.create_enricher() with YAML-based StructureLoader.

Verifies that the factory correctly loads YAML definitions and wires
them into structural applicators and behavioral enrichers.
"""
from pathlib import Path
from typing import Any

import pytest

from enrichment.factory import create_enricher
from enrichment.structure_loader import StructureLoadError


@pytest.fixture
def sample_tools() -> list[dict[str, Any]]:
    """Minimal tool definitions for enrichment testing."""
    return [
        {
            "name": "Read",
            "description": "Reads a file.",
            "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
        },
        {
            "name": "Edit",
            "description": "Edits a file.",
            "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
        },
    ]


@pytest.fixture
def real_structure_dir() -> Path:
    """Return path to the real structure/ directory."""
    return Path(__file__).resolve().parent.parent.parent / "structure"


class TestFactoryWithYaml:
    """Test factory creates enrichers from the real YAML structure."""

    def test_create_enricher_full_mode(self, real_structure_dir: Path) -> None:
        enricher = create_enricher(mode="full", structure_dir=real_structure_dir)
        assert enricher.config.mode == "full"
        assert len(enricher.structural_patterns) == 8
        assert len(enricher.behavioral_enrichers) == 3

    def test_create_enricher_structural_mode(self, real_structure_dir: Path) -> None:
        enricher = create_enricher(mode="structural", structure_dir=real_structure_dir)
        assert enricher.config.mode == "structural"
        assert len(enricher.structural_patterns) == 8

    def test_create_enricher_passthrough_mode(self, real_structure_dir: Path) -> None:
        enricher = create_enricher(mode="passthrough", structure_dir=real_structure_dir)
        assert enricher.config.is_passthrough

    def test_enrichment_adds_structural_fields(
        self, real_structure_dir: Path, sample_tools: list[dict[str, Any]],
    ) -> None:
        enricher = create_enricher(mode="structural", structure_dir=real_structure_dir)
        result = enricher.enrich(sample_tools)
        read_tool = next(t for t in result if t["name"] == "Read")
        # Check structural patterns were applied
        assert "_manifest" in read_tool
        assert "_links" in read_tool
        assert "_error_format" in read_tool
        assert "_near_miss" in read_tool
        assert "_quality" in read_tool
        assert "_anti_patterns" in read_tool
        assert "_registration" in read_tool

    def test_enrichment_adds_behavioral_fields(
        self, real_structure_dir: Path, sample_tools: list[dict[str, Any]],
    ) -> None:
        enricher = create_enricher(mode="full", structure_dir=real_structure_dir)
        result = enricher.enrich(sample_tools)
        read_tool = next(t for t in result if t["name"] == "Read")
        assert "behavioral_what" in read_tool
        assert "behavioral_why" in read_tool
        assert "behavioral_when" in read_tool

    def test_passthrough_does_not_enrich(
        self, real_structure_dir: Path, sample_tools: list[dict[str, Any]],
    ) -> None:
        enricher = create_enricher(mode="passthrough", structure_dir=real_structure_dir)
        result = enricher.enrich(sample_tools)
        read_tool = next(t for t in result if t["name"] == "Read")
        assert "_manifest" not in read_tool
        assert "behavioral_what" not in read_tool

    def test_original_tools_not_mutated(
        self, real_structure_dir: Path, sample_tools: list[dict[str, Any]],
    ) -> None:
        enricher = create_enricher(mode="full", structure_dir=real_structure_dir)
        import copy
        original = copy.deepcopy(sample_tools)
        enricher.enrich(sample_tools)
        assert sample_tools == original

    def test_structure_loader_attached(self, real_structure_dir: Path) -> None:
        enricher = create_enricher(mode="full", structure_dir=real_structure_dir)
        assert hasattr(enricher, "_structure_loader")
        assert enricher._structure_loader is not None


class TestFactoryFailFast:
    """Test that factory fails fast on missing/invalid structure directory."""

    def test_missing_structure_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(StructureLoadError, match="not found"):
            create_enricher(
                mode="full",
                structure_dir=tmp_path / "nonexistent",
            )

    def test_empty_structure_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "structural").mkdir()
        (tmp_path / "behavioral").mkdir()
        (tmp_path / "preamble").mkdir()
        with pytest.raises(StructureLoadError, match="No YAML files"):
            create_enricher(mode="full", structure_dir=tmp_path)
