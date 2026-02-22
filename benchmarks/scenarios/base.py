"""Base class for benchmark scenarios.

Each scenario defines a set of tool definitions, expected enrichment
outcomes per mode, and scoring criteria.
"""
from abc import ABC, abstractmethod
from typing import Any


class BenchmarkScenario(ABC):
    """Base class for benchmark scenarios.

    Subclasses define:
    - The tools to enrich
    - What fields each mode should produce
    - Scenario-specific scoring criteria
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scenario name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this scenario tests."""
        ...

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """Return the tool definitions to enrich (Anthropic format)."""
        ...

    @abstractmethod
    def get_expected_fields(self, mode: str) -> dict[str, list[str]]:
        """Return expected enrichment fields per tool for the given mode.

        Args:
            mode: "passthrough", "structural", or "full"

        Returns:
            Dict mapping tool name to list of expected field names.
        """
        ...
