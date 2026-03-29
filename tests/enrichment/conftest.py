"""Enrichment-specific test fixtures.

Provides sample tool definitions, configured enrichers, and pattern
applicators for use across all enrichment tests.
"""
import pytest
from typing import Any

from enrichment.config import EnrichmentConfig
from enrichment.engine import ToolEnricher
from enrichment.structural.manifest import ManifestApplicator
from enrichment.structural.hateoas import HateoasApplicator
from enrichment.structural.errors import ErrorFormatApplicator
from enrichment.structural.near_miss import NearMissApplicator
from enrichment.structural.self_describing import SelfDescribingApplicator
from enrichment.structural.quality_gates import QualityGatesApplicator
from enrichment.structural.anti_patterns import AntiPatternsApplicator
from enrichment.structural.tool_registration import ToolRegistrationApplicator
from enrichment.behavioral.what_enricher import WhatEnricher
from enrichment.behavioral.why_enricher import WhyEnricher
from enrichment.behavioral.when_enricher import WhenEnricher
from enrichment.structure_loader import StructureLoader, get_default_structure_dir


@pytest.fixture
def sample_tools() -> list[dict[str, Any]]:
    """Minimal tool definitions resembling Claude Code's tools."""
    return [
        {
            "name": "Read",
            "description": "Reads a file from the local filesystem.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The absolute path to the file"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "Edit",
            "description": "Performs exact string replacements in files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
        {
            "name": "Bash",
            "description": "Executes a given bash command.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    ]


@pytest.fixture
def unknown_tool() -> dict[str, Any]:
    """A tool not in the knowledge base."""
    return {
        "name": "CustomTool",
        "description": "A tool we don't have knowledge about.",
        "input_schema": {
            "type": "object",
            "properties": {"data": {"type": "string"}},
        },
    }


@pytest.fixture(scope="session")
def structure_data() -> dict[str, Any]:
    """Loaded YAML structure data (single source of truth for tests)."""
    loader = StructureLoader(get_default_structure_dir())
    return loader.load()


@pytest.fixture
def all_structural_patterns(structure_data) -> list:
    """All structural pattern applicators using real YAML data."""
    s = structure_data.get("structural", {})
    return [
        ManifestApplicator(manifest_data=s.get("manifest", {}).get("manifest")),
        HateoasApplicator(tool_data=s.get("hateoas", {}).get("tools")),
        ErrorFormatApplicator(tool_data=s.get("errors", {}).get("tools")),
        NearMissApplicator(tool_data=s.get("near_miss", {}).get("tools")),
        SelfDescribingApplicator(tool_data=s.get("self_describing", {}).get("tools")),
        QualityGatesApplicator(tool_data=s.get("quality_gates", {}).get("tools")),
        AntiPatternsApplicator(tool_data=s.get("anti_patterns", {}).get("tools")),
        ToolRegistrationApplicator(registration_data=s.get("tool_registration", {}).get("registration")),
    ]


@pytest.fixture
def all_behavioral_enrichers(structure_data) -> list:
    """All behavioral enrichers using real YAML data."""
    b = structure_data.get("behavioral", {})
    return [
        WhatEnricher(tool_data=b.get("what", {}).get("tools")),
        WhyEnricher(tool_data=b.get("why", {}).get("tools")),
        WhenEnricher(tool_data=b.get("when", {}).get("tools")),
    ]


@pytest.fixture
def all_behavioral_enrichers() -> list:
    """All behavioral enrichers."""
    return [
        WhatEnricher(),
        WhyEnricher(),
        WhenEnricher(),
    ]


@pytest.fixture
def full_enricher(all_structural_patterns, all_behavioral_enrichers) -> ToolEnricher:
    """Fully configured enricher with all patterns and behaviors enabled."""
    return ToolEnricher(
        structural_patterns=all_structural_patterns,
        behavioral_enrichers=all_behavioral_enrichers,
        config=EnrichmentConfig(mode="full"),
    )


@pytest.fixture
def structural_only_enricher(all_structural_patterns) -> ToolEnricher:
    """Enricher with structural patterns only (no behavioral)."""
    return ToolEnricher(
        structural_patterns=all_structural_patterns,
        behavioral_enrichers=[],
        config=EnrichmentConfig(mode="structural"),
    )


@pytest.fixture
def passthrough_enricher() -> ToolEnricher:
    """Passthrough enricher that does nothing (benchmark mode)."""
    return ToolEnricher(
        structural_patterns=[],
        behavioral_enrichers=[],
        config=EnrichmentConfig(mode="passthrough"),
    )
