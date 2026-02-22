"""Enrichment metrics — scoring and comparison across modes.

Measures enrichment quality by counting expected fields and comparing
across passthrough, structural, and full modes.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnrichmentMetrics:
    """Metrics for a single scenario run under one enrichment mode.

    Attributes:
        mode: The enrichment mode ("passthrough", "structural", "full").
        scenario: The scenario name that produced these metrics.
        scores: Dict mapping score categories to float values (0.0–1.0).
        enrichment_time_ms: Time taken for enrichment in milliseconds.
        tool_count: Number of tools in the scenario.
        field_counts: Dict mapping field names to counts of tools that have them.
    """

    mode: str
    scenario: str
    scores: dict[str, float] = field(default_factory=dict)
    enrichment_time_ms: float = 0.0
    tool_count: int = 0
    field_counts: dict[str, int] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Weighted average of all scores. Returns 0.0 if no scores."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON export."""
        return {
            "mode": self.mode,
            "scenario": self.scenario,
            "scores": self.scores,
            "overall_score": round(self.overall_score, 4),
            "enrichment_time_ms": round(self.enrichment_time_ms, 2),
            "tool_count": self.tool_count,
            "field_counts": self.field_counts,
        }


# -- Structural fields added by each pattern applicator --
STRUCTURAL_FIELDS = [
    "_manifest",       # P1: Manifest
    "_links",          # P2: HATEOAS
    "_error_format",   # P3: Error Format
    "_near_miss",      # P5: Near-Miss
    "outputSchema",    # P6: Self-Describing (outputSchema is the enrichment)
    "_quality",        # P8: Quality Gates
    "_anti_patterns",  # P14: Anti-Patterns
    "_registration",   # P15: Tool Registration
]

# -- Behavioral fields added by WHAT/WHY/WHEN enrichers --
BEHAVIORAL_FIELDS = [
    "behavioral_what",
    "behavioral_why",
    "behavioral_when",
]


def count_fields(
    enriched_tools: list[dict[str, Any]],
    fields: list[str],
) -> dict[str, int]:
    """Count how many tools have each expected field.

    Args:
        enriched_tools: List of enriched tool dicts.
        fields: List of field names to check.

    Returns:
        Dict mapping field name to count of tools that contain it.
    """
    counts: dict[str, int] = {}
    for field_name in fields:
        counts[field_name] = sum(
            1 for tool in enriched_tools if field_name in tool
        )
    return counts


def score_field_completeness(
    enriched_tools: list[dict[str, Any]],
    expected_fields: list[str],
) -> float:
    """Score enrichment completeness as a ratio of present/expected fields.

    For each tool, checks which expected fields are present. Returns
    the ratio of (fields found) / (fields possible) across all tools.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if no tools or no fields.
    """
    if not enriched_tools or not expected_fields:
        return 0.0

    total_possible = len(enriched_tools) * len(expected_fields)
    total_present = 0

    for tool in enriched_tools:
        for field_name in expected_fields:
            if field_name in tool:
                total_present += 1

    return total_present / total_possible


def score_structural(enriched_tools: list[dict[str, Any]]) -> float:
    """Score structural enrichment completeness (0.0–1.0)."""
    return score_field_completeness(enriched_tools, STRUCTURAL_FIELDS)


def score_behavioral(enriched_tools: list[dict[str, Any]]) -> float:
    """Score behavioral enrichment completeness (0.0–1.0)."""
    return score_field_completeness(enriched_tools, BEHAVIORAL_FIELDS)


def score_scenario(
    enriched_tools: list[dict[str, Any]],
    expected_fields: dict[str, list[str]],
) -> dict[str, float]:
    """Score enrichment against scenario-specific expectations.

    Args:
        enriched_tools: The enriched tool definitions.
        expected_fields: Dict mapping tool name to list of expected field names.

    Returns:
        Dict with score categories:
        - "structural": Structural pattern completeness
        - "behavioral": Behavioral dimension completeness
        - "scenario_specific": How many scenario-specific expected fields are present
    """
    scores: dict[str, float] = {
        "structural": score_structural(enriched_tools),
        "behavioral": score_behavioral(enriched_tools),
    }

    # Scenario-specific scoring
    if expected_fields:
        total_expected = 0
        total_found = 0
        for tool in enriched_tools:
            tool_name = tool.get("name", "")
            expected = expected_fields.get(tool_name, [])
            total_expected += len(expected)
            for field_name in expected:
                if field_name in tool:
                    total_found += 1

        scores["scenario_specific"] = (
            total_found / total_expected if total_expected > 0 else 0.0
        )

    return scores


def compare_modes(
    metrics_list: list[EnrichmentMetrics],
) -> dict[str, Any]:
    """Compare metrics across enrichment modes.

    Groups metrics by mode and computes summary statistics.

    Returns:
        Dict with per-mode averages and rankings.
    """
    by_mode: dict[str, list[EnrichmentMetrics]] = {}
    for m in metrics_list:
        by_mode.setdefault(m.mode, []).append(m)

    summary: dict[str, Any] = {}
    for mode, metrics in by_mode.items():
        avg_score = sum(m.overall_score for m in metrics) / len(metrics)
        avg_time = sum(m.enrichment_time_ms for m in metrics) / len(metrics)
        summary[mode] = {
            "avg_overall_score": round(avg_score, 4),
            "avg_enrichment_time_ms": round(avg_time, 2),
            "scenario_count": len(metrics),
            "scenarios": {m.scenario: round(m.overall_score, 4) for m in metrics},
        }

    # Rank modes by average score
    ranked = sorted(summary.items(), key=lambda x: x[1]["avg_overall_score"], reverse=True)
    for rank, (mode, data) in enumerate(ranked, 1):
        data["rank"] = rank

    return summary
