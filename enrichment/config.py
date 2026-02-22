"""Enrichment configuration.

Controls which enrichment layers are active and which patterns to apply.
Frozen dataclass — immutable after creation for pipeline safety.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnrichmentConfig:
    """Configuration for which enrichment layers are active.

    Use PASSTHROUGH mode for before/after benchmarking (skips all enrichment).
    Use STRUCTURAL for API standard patterns only.
    Use FULL for structural + behavioral (Gold standard).
    """

    mode: str = "full"  # "passthrough" | "structural" | "full"

    # Structural patterns to apply (Pattern numbers from Agentic API Standard)
    enabled_structural: frozenset[int] = field(
        default_factory=lambda: frozenset({1, 2, 3, 5, 6, 8, 14, 15})
    )

    # Behavioral dimensions to apply
    enable_what: bool = True
    enable_why: bool = True
    enable_when: bool = True

    @property
    def is_passthrough(self) -> bool:
        """True if enrichment is completely disabled (benchmark mode)."""
        return self.mode == "passthrough"

    @property
    def include_behavioral(self) -> bool:
        """True if behavioral enrichment should be applied (full mode only)."""
        return self.mode == "full"
