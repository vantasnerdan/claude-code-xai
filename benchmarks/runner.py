"""Benchmark runner — executes scenarios across enrichment modes.

Runs each scenario through passthrough, structural, and full enrichment
modes, captures metrics, and returns structured results.

No live API required. Measures enrichment quality, not API latency.
"""
import time
from typing import Any

from enrichment.factory import create_enricher
from benchmarks.metrics import (
    EnrichmentMetrics,
    STRUCTURAL_FIELDS,
    BEHAVIORAL_FIELDS,
    count_fields,
    score_scenario,
    compare_modes,
)
from benchmarks.scenarios.base import BenchmarkScenario
from benchmarks.scenarios import ALL_SCENARIOS

MODES = ["passthrough", "structural", "full"]


def run_scenario(
    scenario: BenchmarkScenario,
    mode: str,
) -> EnrichmentMetrics:
    """Run a single scenario under one enrichment mode.

    Args:
        scenario: The benchmark scenario to run.
        mode: The enrichment mode ("passthrough", "structural", "full").

    Returns:
        EnrichmentMetrics with scores and field counts.
    """
    tools = scenario.get_tools()
    expected_fields = scenario.get_expected_fields(mode)

    # Create enricher for this mode
    enricher = create_enricher(mode=mode)

    # Time the enrichment
    start = time.perf_counter()
    enriched = enricher.enrich(tools)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Score the enrichment
    scores = score_scenario(enriched, expected_fields)

    # Count all enrichment fields
    all_fields = STRUCTURAL_FIELDS + BEHAVIORAL_FIELDS
    field_counts = count_fields(enriched, all_fields)

    return EnrichmentMetrics(
        mode=mode,
        scenario=scenario.name,
        scores=scores,
        enrichment_time_ms=elapsed_ms,
        tool_count=len(tools),
        field_counts=field_counts,
    )


def run_all_scenarios(
    scenarios: list[type[BenchmarkScenario]] | None = None,
    modes: list[str] | None = None,
) -> list[EnrichmentMetrics]:
    """Run all scenarios across all modes.

    Args:
        scenarios: Scenario classes to run. Defaults to ALL_SCENARIOS.
        modes: Enrichment modes to test. Defaults to all three.

    Returns:
        List of EnrichmentMetrics, one per (scenario, mode) combination.
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS
    if modes is None:
        modes = MODES

    results: list[EnrichmentMetrics] = []
    for scenario_class in scenarios:
        scenario = scenario_class()
        for mode in modes:
            metrics = run_scenario(scenario, mode)
            results.append(metrics)

    return results


def run_benchmark() -> dict[str, Any]:
    """Run the full benchmark suite and return structured results.

    Returns:
        Dict with:
        - "results": List of per-scenario-per-mode metric dicts
        - "comparison": Cross-mode comparison summary
        - "metadata": Run metadata (scenario count, mode count)
    """
    metrics = run_all_scenarios()
    comparison = compare_modes(metrics)

    return {
        "results": [m.to_dict() for m in metrics],
        "comparison": comparison,
        "metadata": {
            "scenario_count": len(ALL_SCENARIOS),
            "mode_count": len(MODES),
            "modes": MODES,
            "scenarios": [s().name for s in ALL_SCENARIOS],
        },
    }
