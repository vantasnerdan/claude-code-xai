"""Factory for creating configured ToolEnricher instances.

Reads ENRICHMENT_MODE from environment to determine which layers are active.
"""
import os
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


def create_enricher(mode: str | None = None) -> ToolEnricher:
    """Create a ToolEnricher with all patterns and behavioral enrichers.

    Args:
        mode: "passthrough", "structural", or "full".
              Defaults to ENRICHMENT_MODE env var, then "full".
    """
    if mode is None:
        mode = os.getenv("ENRICHMENT_MODE", "full")

    config = EnrichmentConfig(mode=mode)

    structural = [
        ManifestApplicator(),
        HateoasApplicator(),
        ErrorFormatApplicator(),
        NearMissApplicator(),
        SelfDescribingApplicator(),
        QualityGatesApplicator(),
        AntiPatternsApplicator(),
        ToolRegistrationApplicator(),
    ]

    behavioral = [
        WhatEnricher(),
        WhyEnricher(),
        WhenEnricher(),
    ]

    return ToolEnricher(
        structural_patterns=structural,
        behavioral_enrichers=behavioral,
        config=config,
    )
