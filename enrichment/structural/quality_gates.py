"""Pattern 8: Warnings and quality flags.

Adds quality metadata to tool definitions — known limitations,
edge cases, and quality signals that help the model make better decisions.
"""
from typing import Any

from enrichment.structural.base import DataDrivenPatternApplicator


class QualityGatesApplicator(DataDrivenPatternApplicator):
    """Applies Agentic API Standard Pattern 8: Warnings + Quality.

    Uses DataDrivenPatternApplicator base. YAML is single source of truth.
    """

    _field_name = "_quality"

    @property
    def pattern_number(self) -> int:
        return 8

    @property
    def name(self) -> str:
        return "Quality Gates"
