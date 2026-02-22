"""Export benchmark results to JSON and CSV.

Supports two output formats:
- JSON: Full structured results for programmatic consumption
- CSV: Summary table (mode x scenario x score) for README charts
"""
import csv
import io
import json
from typing import Any

from benchmarks.metrics import EnrichmentMetrics


def export_json(
    results: dict[str, Any],
    path: str | None = None,
    indent: int = 2,
) -> str:
    """Export benchmark results to JSON.

    Args:
        results: The full benchmark results dict from run_benchmark().
        path: Optional file path to write. If None, returns JSON string.
        indent: JSON indentation level.

    Returns:
        JSON string of the results.
    """
    json_str = json.dumps(results, indent=indent, sort_keys=False)

    if path:
        with open(path, "w") as f:
            f.write(json_str)
            f.write("\n")

    return json_str


def export_csv(
    results: dict[str, Any],
    path: str | None = None,
) -> str:
    """Export benchmark results to CSV summary table.

    Columns: mode, scenario, overall_score, structural_score, behavioral_score,
             scenario_specific_score, enrichment_time_ms, tool_count

    Args:
        results: The full benchmark results dict from run_benchmark().
        path: Optional file path to write. If None, returns CSV string.

    Returns:
        CSV string of the summary table.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "mode",
        "scenario",
        "overall_score",
        "structural_score",
        "behavioral_score",
        "scenario_specific_score",
        "enrichment_time_ms",
        "tool_count",
    ])

    # Data rows
    for result in results.get("results", []):
        scores = result.get("scores", {})
        writer.writerow([
            result["mode"],
            result["scenario"],
            result["overall_score"],
            scores.get("structural", 0.0),
            scores.get("behavioral", 0.0),
            scores.get("scenario_specific", 0.0),
            result["enrichment_time_ms"],
            result["tool_count"],
        ])

    csv_str = output.getvalue()

    if path:
        with open(path, "w") as f:
            f.write(csv_str)

    return csv_str


def format_summary_table(results: dict[str, Any]) -> str:
    """Format a human-readable summary table for terminal output.

    Args:
        results: The full benchmark results dict from run_benchmark().

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Per-scenario results
    header = f"{'Mode':<15} {'Scenario':<25} {'Overall':>10} {'Struct':>10} {'Behav':>10} {'Time(ms)':>10}"
    lines.append(header)
    lines.append("-" * 80)

    for result in results.get("results", []):
        scores = result.get("scores", {})
        lines.append(
            f"{result['mode']:<15} "
            f"{result['scenario']:<25} "
            f"{result['overall_score']:>10.4f} "
            f"{scores.get('structural', 0.0):>10.4f} "
            f"{scores.get('behavioral', 0.0):>10.4f} "
            f"{result['enrichment_time_ms']:>10.2f}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("MODE COMPARISON")
    lines.append("=" * 80)

    comparison = results.get("comparison", {})
    for mode in ["passthrough", "structural", "full"]:
        data = comparison.get(mode, {})
        if data:
            rank = data.get("rank", "?")
            avg = data.get("avg_overall_score", 0.0)
            avg_time = data.get("avg_enrichment_time_ms", 0.0)
            lines.append(
                f"  #{rank} {mode:<15} avg_score={avg:.4f}  avg_time={avg_time:.2f}ms"
            )

    lines.append("")
    return "\n".join(lines)
