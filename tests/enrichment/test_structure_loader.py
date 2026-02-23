"""Tests for StructureLoader — YAML-based enrichment definition loading.

Covers: loading, validation, mtime caching, reload behavior,
error cases (malformed YAML, missing files, missing required fields),
and integration with the enrichment pipeline via factory.
"""
import os
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from enrichment.structure_loader import StructureLoader, StructureLoadError, get_default_structure_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Helper to write a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


@pytest.fixture
def minimal_structure(tmp_path: Path) -> Path:
    """Create a minimal valid structure directory with one file per subdirectory."""
    _write_yaml(tmp_path / "manifest.yaml", {
        "schema_version": "1.0",
        "description": "test manifest",
    })
    _write_yaml(tmp_path / "structural" / "hateoas.yaml", {
        "schema_version": "1.0",
        "type": "structural",
        "pattern": "hateoas",
        "tools": {
            "Read": {"related": ["Edit"], "on_error": {"file_not_found": "Use Glob"}},
        },
    })
    _write_yaml(tmp_path / "behavioral" / "what.yaml", {
        "schema_version": "1.0",
        "type": "behavioral",
        "dimension": "what",
        "tools": {"Read": "Reads files from filesystem."},
    })
    _write_yaml(tmp_path / "preamble" / "identity.yaml", {
        "schema_version": "1.0",
        "type": "preamble",
        "name": "identity",
        "text": "You are Grok.",
    })
    return tmp_path


@pytest.fixture
def full_structure() -> Path:
    """Return path to the real structure/ directory in the repo."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "structure"


# ---------------------------------------------------------------------------
# Basic Loading
# ---------------------------------------------------------------------------

class TestBasicLoading:
    """Test that StructureLoader loads and returns structured data."""

    def test_load_minimal(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data = loader.load()
        assert "structural" in data
        assert "behavioral" in data
        assert "preamble" in data
        assert "manifest" in data

    def test_structural_data_keyed_by_stem(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data = loader.load()
        assert "hateoas" in data["structural"]

    def test_behavioral_data_keyed_by_stem(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data = loader.load()
        assert "what" in data["behavioral"]

    def test_preamble_data_keyed_by_stem(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data = loader.load()
        assert "identity" in data["preamble"]

    def test_manifest_loaded(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data = loader.load()
        assert data["manifest"]["schema_version"] == "1.0"

    def test_tool_data_accessible(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data = loader.load()
        tools = data["structural"]["hateoas"]["tools"]
        assert "Read" in tools
        assert tools["Read"]["related"] == ["Edit"]


# ---------------------------------------------------------------------------
# Real Structure Directory
# ---------------------------------------------------------------------------

class TestRealStructure:
    """Test loading the actual committed structure/ directory."""

    def test_loads_all_structural_patterns(self, full_structure: Path) -> None:
        loader = StructureLoader(full_structure)
        data = loader.load()
        expected = {
            "manifest", "hateoas", "errors", "near_miss",
            "self_describing", "quality_gates", "anti_patterns",
            "tool_registration",
        }
        assert set(data["structural"].keys()) == expected

    def test_loads_all_behavioral_dimensions(self, full_structure: Path) -> None:
        loader = StructureLoader(full_structure)
        data = loader.load()
        assert set(data["behavioral"].keys()) == {"what", "why", "when"}

    def test_loads_preamble_files(self, full_structure: Path) -> None:
        loader = StructureLoader(full_structure)
        data = loader.load()
        assert "identity" in data["preamble"]
        assert "behavioral" in data["preamble"]

    def test_structural_hateoas_has_tools(self, full_structure: Path) -> None:
        loader = StructureLoader(full_structure)
        data = loader.load()
        tools = data["structural"]["hateoas"]["tools"]
        assert "Read" in tools
        assert "Edit" in tools
        assert "Bash" in tools

    def test_behavioral_what_has_tools(self, full_structure: Path) -> None:
        loader = StructureLoader(full_structure)
        data = loader.load()
        tools = data["behavioral"]["what"]["tools"]
        assert "Read" in tools
        assert "WebFetch" in tools

    def test_preamble_has_text(self, full_structure: Path) -> None:
        loader = StructureLoader(full_structure)
        data = loader.load()
        assert "text" in data["preamble"]["identity"]
        assert "Grok" in data["preamble"]["identity"]["text"]


# ---------------------------------------------------------------------------
# Mtime Caching
# ---------------------------------------------------------------------------

class TestMtimeCaching:
    """Test lazy mtime-based reload behavior."""

    def test_returns_cached_on_same_mtime(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data1 = loader.load()
        data2 = loader.load()
        # Same object reference means cache was used
        assert data1 is data2

    def test_reloads_when_mtime_changes(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data1 = loader.load()

        # Touch directory to change mtime
        time.sleep(0.05)
        os.utime(minimal_structure, None)

        data2 = loader.load()
        # Different object reference means cache was invalidated
        assert data1 is not data2

    def test_cache_empty_initially(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        assert loader._cache is None
        assert loader._last_mtime == 0.0

    def test_cache_populated_after_load(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        loader.load()
        assert loader._cache is not None
        assert loader._last_mtime > 0.0

    def test_reloads_after_file_modification(self, minimal_structure: Path) -> None:
        loader = StructureLoader(minimal_structure)
        data1 = loader.load()

        # Modify a YAML file (this changes parent dir mtime)
        time.sleep(0.05)
        hateoas = minimal_structure / "structural" / "hateoas.yaml"
        with open(hateoas, "a") as f:
            f.write("\n# modified\n")
        # Touch the structure dir so mtime changes
        os.utime(minimal_structure, None)

        data2 = loader.load()
        assert data1 is not data2


# ---------------------------------------------------------------------------
# Validation and Error Cases
# ---------------------------------------------------------------------------

class TestValidation:
    """Test fail-fast validation on malformed or missing YAML."""

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        loader = StructureLoader(tmp_path / "nonexistent")
        with pytest.raises(StructureLoadError, match="not found"):
            loader.load()

    def test_missing_subdirectory_raises(self, tmp_path: Path) -> None:
        # Create only structural, missing behavioral and preamble
        _write_yaml(tmp_path / "structural" / "test.yaml", {
            "schema_version": "1.0", "type": "structural",
        })
        loader = StructureLoader(tmp_path)
        with pytest.raises(StructureLoadError, match="subdirectory missing"):
            loader.load()

    def test_empty_subdirectory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "structural").mkdir()
        (tmp_path / "behavioral").mkdir()
        (tmp_path / "preamble").mkdir()
        loader = StructureLoader(tmp_path)
        with pytest.raises(StructureLoadError, match="No YAML files"):
            loader.load()

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        (tmp_path / "structural").mkdir()
        (tmp_path / "behavioral").mkdir()
        (tmp_path / "preamble").mkdir()
        # Write invalid YAML
        with open(tmp_path / "structural" / "bad.yaml", "w") as f:
            f.write("{ invalid yaml: [")
        loader = StructureLoader(tmp_path)
        with pytest.raises(StructureLoadError, match="Malformed YAML"):
            loader.load()

    def test_missing_schema_version_raises(self, tmp_path: Path) -> None:
        (tmp_path / "structural").mkdir()
        (tmp_path / "behavioral").mkdir()
        (tmp_path / "preamble").mkdir()
        _write_yaml(tmp_path / "structural" / "test.yaml", {
            "type": "structural",
            # Missing schema_version
        })
        _write_yaml(tmp_path / "behavioral" / "test.yaml", {
            "schema_version": "1.0", "type": "behavioral",
        })
        _write_yaml(tmp_path / "preamble" / "test.yaml", {
            "schema_version": "1.0", "type": "preamble",
        })
        loader = StructureLoader(tmp_path)
        with pytest.raises(StructureLoadError, match="Missing required fields"):
            loader.load()

    def test_missing_type_raises(self, tmp_path: Path) -> None:
        (tmp_path / "structural").mkdir()
        (tmp_path / "behavioral").mkdir()
        (tmp_path / "preamble").mkdir()
        _write_yaml(tmp_path / "structural" / "test.yaml", {
            "schema_version": "1.0",
            # Missing type
        })
        _write_yaml(tmp_path / "behavioral" / "test.yaml", {
            "schema_version": "1.0", "type": "behavioral",
        })
        _write_yaml(tmp_path / "preamble" / "test.yaml", {
            "schema_version": "1.0", "type": "preamble",
        })
        loader = StructureLoader(tmp_path)
        with pytest.raises(StructureLoadError, match="Missing required fields"):
            loader.load()

    def test_invalid_type_raises(self, tmp_path: Path) -> None:
        (tmp_path / "structural").mkdir()
        (tmp_path / "behavioral").mkdir()
        (tmp_path / "preamble").mkdir()
        _write_yaml(tmp_path / "structural" / "test.yaml", {
            "schema_version": "1.0", "type": "invalid_type",
        })
        _write_yaml(tmp_path / "behavioral" / "test.yaml", {
            "schema_version": "1.0", "type": "behavioral",
        })
        _write_yaml(tmp_path / "preamble" / "test.yaml", {
            "schema_version": "1.0", "type": "preamble",
        })
        loader = StructureLoader(tmp_path)
        with pytest.raises(StructureLoadError, match="Invalid type"):
            loader.load()

    def test_non_dict_top_level_raises(self, tmp_path: Path) -> None:
        (tmp_path / "structural").mkdir()
        (tmp_path / "behavioral").mkdir()
        (tmp_path / "preamble").mkdir()
        with open(tmp_path / "structural" / "test.yaml", "w") as f:
            f.write("- just\n- a\n- list\n")
        loader = StructureLoader(tmp_path)
        with pytest.raises(StructureLoadError, match="Expected mapping"):
            loader.load()


# ---------------------------------------------------------------------------
# Default Path
# ---------------------------------------------------------------------------

class TestDefaultPath:
    """Test get_default_structure_dir() behavior."""

    def test_returns_path_object(self) -> None:
        result = get_default_structure_dir()
        assert isinstance(result, Path)

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRUCTURE_DIR", "/custom/path")
        result = get_default_structure_dir()
        assert result == Path("/custom/path")

    def test_fallback_to_repo_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STRUCTURE_DIR", raising=False)
        result = get_default_structure_dir()
        assert result.name == "structure"
        # Should be sibling of enrichment/ directory
        assert (result.parent / "enrichment").is_dir()


# ---------------------------------------------------------------------------
# Extensibility
# ---------------------------------------------------------------------------

class TestExtensibility:
    """Test that the loader supports extensible tool lists in YAML."""

    def test_new_tool_in_structural_yaml(self, minimal_structure: Path) -> None:
        """Adding a new tool to YAML should be picked up by the loader."""
        hateoas_path = minimal_structure / "structural" / "hateoas.yaml"
        data = yaml.safe_load(open(hateoas_path))
        data["tools"]["CustomNewTool"] = {
            "related": ["Read"],
            "on_error": {"generic": "Try again"},
        }
        with open(hateoas_path, "w") as f:
            yaml.safe_dump(data, f)
        # Touch dir to invalidate cache
        os.utime(minimal_structure, None)

        loader = StructureLoader(minimal_structure)
        result = loader.load()
        assert "CustomNewTool" in result["structural"]["hateoas"]["tools"]

    def test_new_tool_in_behavioral_yaml(self, minimal_structure: Path) -> None:
        """Adding a new tool to behavioral YAML should be picked up."""
        what_path = minimal_structure / "behavioral" / "what.yaml"
        data = yaml.safe_load(open(what_path))
        data["tools"]["NewTool"] = "Does something new."
        with open(what_path, "w") as f:
            yaml.safe_dump(data, f)
        os.utime(minimal_structure, None)

        loader = StructureLoader(minimal_structure)
        result = loader.load()
        assert "NewTool" in result["behavioral"]["what"]["tools"]
