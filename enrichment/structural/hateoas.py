"""Pattern 2: HATEOAS — Hypermedia links for navigation and recovery.

Adds _links to tool definitions so the model knows how to recover
from errors, find related tools, and navigate the tool ecosystem.
"""
from typing import Any

from enrichment.structural.base import DataDrivenPatternApplicator


class HateoasApplicator(DataDrivenPatternApplicator):
    """Applies Agentic API Standard Pattern 2: HATEOAS.

    Uses DataDrivenPatternApplicator base to eliminate duplication.
    YAML (structure/structural/hateoas.yaml) is the single source of truth.
    """

    _field_name = "_links"

    @property
    def pattern_number(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "HATEOAS"
