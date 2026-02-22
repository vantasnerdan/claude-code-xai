"""Tool enrichment engine.

Applies the Agentic API Standard patterns and behavioral training transfer
to tool definitions before they're sent to the model. This is what makes
the bridge a universal upgrade layer rather than just a format translator.

Architecture:
  Layer 1 — Structural: Independent pattern applicators from the API standard
  Layer 2 — Behavioral: WHAT/WHY/WHEN knowledge from Claude Code's training

Both layers are composable and configurable via EnrichmentConfig.
"""
import copy
from typing import Any

from enrichment.config import EnrichmentConfig
from enrichment.structural.base import PatternApplicator
from enrichment.behavioral.base import BehavioralEnricher


class ToolEnricher:
    """Pipeline orchestrator for tool enrichment.

    Applies structural patterns (API standard) and behavioral knowledge
    (training transfer) to tool definitions in sequence.
    """

    def __init__(
        self,
        structural_patterns: list[PatternApplicator],
        behavioral_enrichers: list[BehavioralEnricher],
        config: EnrichmentConfig,
    ) -> None:
        self.structural_patterns = structural_patterns
        self.behavioral_enrichers = behavioral_enrichers
        self.config = config

    def enrich(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply all enabled enrichment layers to tool definitions.

        Returns a new list — never modifies the input.
        """
        if self.config.is_passthrough:
            return tools

        # Deep copy to avoid mutation
        enriched = copy.deepcopy(tools)

        # Layer 1: Structural patterns (API standard)
        for pattern in self.structural_patterns:
            if pattern.pattern_number in self.config.enabled_structural:
                enriched = pattern.apply(enriched)

        # Layer 2: Behavioral enrichment (training transfer)
        if self.config.include_behavioral:
            for i, tool in enumerate(enriched):
                for enricher in self.behavioral_enrichers:
                    if self._should_apply_behavioral(enricher):
                        enriched[i] = enricher.enrich(enriched[i])

        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check tools for compliance with all enabled patterns.

        Returns list of issue dicts.
        """
        issues: list[dict[str, Any]] = []
        for pattern in self.structural_patterns:
            if pattern.pattern_number in self.config.enabled_structural:
                issues.extend(pattern.validate(tools))
        return issues

    def _should_apply_behavioral(self, enricher: BehavioralEnricher) -> bool:
        """Check if a behavioral enricher is enabled in config."""
        dimension = enricher.dimension
        if dimension == "what":
            return self.config.enable_what
        elif dimension == "why":
            return self.config.enable_why
        elif dimension == "when":
            return self.config.enable_when
        return False
