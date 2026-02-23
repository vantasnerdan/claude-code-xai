"""Loads enrichment definitions from YAML files in the structure/ directory.

Implements lazy mtime-based reload: stats the structure directory once per
request. Only reparses YAML files when the directory mtime has changed.
Fails fast at startup if any YAML is malformed or missing required fields.

Consensus decisions (Issue #27):
  - YAML-only with fail-fast validation -- NO Python fallbacks
  - Structure directory committed in-repo (version-controlled)
  - Preamble included in lazy reload
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_PATTERN_FIELDS = {"schema_version", "type"}
_VALID_TYPES = {"structural", "behavioral", "preamble"}


class StructureLoadError(Exception):
    """Raised when structure files are missing, malformed, or invalid."""


class StructureLoader:
    """Loads and caches enrichment pattern definitions from YAML files.

    Args:
        structure_dir: Path to the structure/ directory containing YAML files.
    """

    def __init__(self, structure_dir: str | Path) -> None:
        self._dir = Path(structure_dir)
        self._cache: dict[str, Any] | None = None
        self._last_mtime: float = 0.0

    def load(self) -> dict[str, Any]:
        """Load all YAML definitions, using mtime cache when possible.

        Returns a dict with keys: structural, behavioral, preamble.
        Each maps pattern/dimension names to their parsed YAML data.

        Raises:
            StructureLoadError: If the directory is missing, any YAML is
                malformed, or required fields are absent.
        """
        if not self._dir.is_dir():
            raise StructureLoadError(
                f"Structure directory not found: {self._dir}"
            )

        current_mtime = self._dir.stat().st_mtime
        if self._cache is not None and current_mtime == self._last_mtime:
            return self._cache

        data = self._parse_all()
        self._cache = data
        self._last_mtime = current_mtime
        return data

    def _parse_all(self) -> dict[str, Any]:
        """Parse all YAML files in the structure directory tree."""
        structural = self._load_subdir("structural")
        behavioral = self._load_subdir("behavioral")
        preamble = self._load_subdir("preamble")

        manifest_path = self._dir / "manifest.yaml"
        if manifest_path.is_file():
            manifest = self._load_yaml(manifest_path)
        else:
            manifest = {}

        return {
            "manifest": manifest,
            "structural": structural,
            "behavioral": behavioral,
            "preamble": preamble,
        }

    def _load_subdir(self, subdir_name: str) -> dict[str, Any]:
        """Load all YAML files from a subdirectory.

        Returns a dict mapping the file stem (e.g., 'hateoas') to parsed data.
        """
        subdir = self._dir / subdir_name
        if not subdir.is_dir():
            raise StructureLoadError(
                f"Required subdirectory missing: {subdir}"
            )

        result: dict[str, Any] = {}
        yaml_files = sorted(subdir.glob("*.yaml"))
        if not yaml_files:
            raise StructureLoadError(
                f"No YAML files found in {subdir}"
            )

        for yaml_path in yaml_files:
            data = self._load_yaml(yaml_path)
            self._validate_fields(data, yaml_path)
            result[yaml_path.stem] = data

        return result

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Parse a single YAML file. Raises on malformed YAML."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise StructureLoadError(
                f"Malformed YAML in {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise StructureLoadError(
                f"Expected mapping at top level in {path}, got {type(data).__name__}"
            )
        return data

    def _validate_fields(self, data: dict[str, Any], path: Path) -> None:
        """Validate required fields are present in a pattern definition."""
        missing = _REQUIRED_PATTERN_FIELDS - set(data.keys())
        if missing:
            raise StructureLoadError(
                f"Missing required fields {missing} in {path}"
            )

        entry_type = data.get("type")
        if entry_type not in _VALID_TYPES:
            raise StructureLoadError(
                f"Invalid type '{entry_type}' in {path}. "
                f"Must be one of: {_VALID_TYPES}"
            )


def get_default_structure_dir() -> Path:
    """Return the default structure directory path.

    Checks STRUCTURE_DIR env var first, then falls back to
    structure/ relative to the repository root.
    """
    env_dir = os.getenv("STRUCTURE_DIR")
    if env_dir:
        return Path(env_dir)
    # Repo root is one level up from enrichment/
    return Path(__file__).resolve().parent.parent / "structure"
