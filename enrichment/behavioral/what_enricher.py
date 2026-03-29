"""WHAT dimension behavioral enricher.

Enhances tool descriptions with richer detail about what the tool does,
drawn from the knowledge base of Claude Code's trained behavior.
"""
from typing import Any

from enrichment.behavioral.base import DataDrivenBehavioralEnricher


class WhatEnricher(DataDrivenBehavioralEnricher):
    """Enriches tool definitions with enhanced WHAT descriptions.

    Uses DataDrivenBehavioralEnricher base for common logic. YAML is
    the single source of truth.
    """

    _enrichment_key = "behavioral_what"

    @property
    def dimension(self) -> str:
        return "what"
