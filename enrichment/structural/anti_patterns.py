"""Pattern 14: Anti-patterns — known failure modes per tool.

Documents what NOT to do with each tool. This is critical for models
without RL training on tool use, as they don't have negative examples
baked into their weights.
"""
from typing import Any

from enrichment.structural.base import DataDrivenPatternApplicator


class AntiPatternsApplicator(DataDrivenPatternApplicator):
    """Applies Agentic API Standard Pattern 14: Anti-Patterns.

    Uses DataDrivenPatternApplicator base. YAML is single source of truth.
    """

    _field_name = "_anti_patterns"

    @property
    def pattern_number(self) -> int:
        return 14

    @property
    def name(self) -> str:
        return "Anti-Patterns"
