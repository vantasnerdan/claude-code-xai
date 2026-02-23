"""Factory for creating configured ToolEnricher instances.

Reads ENRICHMENT_MODE from environment to determine which layers are active.
Loads pattern definitions from YAML via StructureLoader (fail-fast on startup).
"""
import os
from pathlib import Path
from typing import Any

from enrichment.config import EnrichmentConfig
from enrichment.engine import ToolEnricher
from enrichment.structure_loader import StructureLoader, StructureLoadError, get_default_structure_dir
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

from bridge.logging_config import get_logger

logger = get_logger("factory")


def create_enricher(
    mode: str | None = None,
    structure_dir: str | Path | None = None,
) -> ToolEnricher:
    """Create a ToolEnricher with all patterns and behavioral enrichers.

    Loads pattern definitions from YAML files in the structure directory.
    Fails fast at startup if the structure directory is missing or any
    YAML file is malformed (consensus decision: YAML-only, no fallbacks).

    Args:
        mode: "passthrough", "structural", or "full".
              Defaults to ENRICHMENT_MODE env var, then "full".
        structure_dir: Path to structure/ directory. Defaults to
              STRUCTURE_DIR env var, then structure/ at repo root.

    Raises:
        StructureLoadError: If structure directory or YAML files are invalid.
    """
    if mode is None:
        mode = os.getenv("ENRICHMENT_MODE", "full")

    config = EnrichmentConfig(mode=mode)

    if structure_dir is not None:
        s_dir = Path(structure_dir)
    else:
        s_dir = get_default_structure_dir()

    loader = StructureLoader(s_dir)
    data = loader.load()

    structural = _build_structural(data.get("structural", {}))
    behavioral = _build_behavioral(data.get("behavioral", {}))

    logger.info(
        "Enricher created mode=%s structure_dir=%s patterns=%d behavioral=%d",
        mode, s_dir, len(structural), len(behavioral),
    )

    enricher = ToolEnricher(
        structural_patterns=structural,
        behavioral_enrichers=behavioral,
        config=config,
    )
    enricher._structure_loader = loader
    return enricher


def _build_structural(
    structural_data: dict[str, Any],
) -> list:
    """Build structural applicators from loaded YAML data."""
    manifest_yaml = structural_data.get("manifest", {})
    hateoas_yaml = structural_data.get("hateoas", {})
    errors_yaml = structural_data.get("errors", {})
    near_miss_yaml = structural_data.get("near_miss", {})
    self_desc_yaml = structural_data.get("self_describing", {})
    quality_yaml = structural_data.get("quality_gates", {})
    anti_yaml = structural_data.get("anti_patterns", {})
    registration_yaml = structural_data.get("tool_registration", {})

    return [
        ManifestApplicator(
            manifest_data=manifest_yaml.get("manifest"),
        ),
        HateoasApplicator(
            tool_data=hateoas_yaml.get("tools"),
        ),
        ErrorFormatApplicator(
            tool_data=errors_yaml.get("tools"),
        ),
        NearMissApplicator(
            tool_data=near_miss_yaml.get("tools"),
        ),
        SelfDescribingApplicator(
            tool_data=self_desc_yaml.get("tools"),
        ),
        QualityGatesApplicator(
            tool_data=quality_yaml.get("tools"),
        ),
        AntiPatternsApplicator(
            tool_data=anti_yaml.get("tools"),
        ),
        ToolRegistrationApplicator(
            registration_data=registration_yaml.get("registration"),
        ),
    ]


def _build_behavioral(
    behavioral_data: dict[str, Any],
) -> list:
    """Build behavioral enrichers from loaded YAML data."""
    what_yaml = behavioral_data.get("what", {})
    why_yaml = behavioral_data.get("why", {})
    when_yaml = behavioral_data.get("when", {})

    return [
        WhatEnricher(tool_data=what_yaml.get("tools")),
        WhyEnricher(tool_data=why_yaml.get("tools")),
        WhenEnricher(tool_data=when_yaml.get("tools")),
    ]
