"""Base class for structural enrichment pattern applicators.

Each pattern from the Agentic API Standard gets its own applicator.
Applicators are independent and composable — the engine pipelines them.
"""
from abc import ABC, abstractmethod
from typing import Any


class PatternApplicator(ABC):
    """Base class for structural enrichment patterns.

    Each pattern from the Agentic API Standard gets its own applicator.
    Applicators are independent and composable.
    """

    @property
    @abstractmethod
    def pattern_number(self) -> int:
        """The Agentic API Standard pattern number (1-20)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pattern name."""
        ...

    @abstractmethod
    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enrich tool definitions with this pattern.

        Must not modify the original tools list — return a new list.
        Must not change the semantic meaning of tool descriptions.
        """
        ...

    @abstractmethod
    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check tool definitions for compliance with this pattern.

        Returns a list of issue dicts:
        [{"tool": name, "issue": description, "severity": "error"|"warning"}]
        """
        ...
