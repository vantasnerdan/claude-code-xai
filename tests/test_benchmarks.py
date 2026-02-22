"""Tests for the benchmark framework.

Verifies:
1. Runner executes all scenarios across all modes
2. Full mode scores higher than structural, which scores higher than passthrough
3. Export formats are valid (JSON and CSV)
4. Metrics calculations are correct
5. Scenarios produce deterministic results
"""
import csv
import io
import json

import pytest

from benchmarks.metrics import (
    EnrichmentMetrics,
    STRUCTURAL_FIELDS,
    BEHAVIORAL_FIELDS,
    count_fields,
    score_field_completeness,
    score_structural,
    score_behavioral,
    score_scenario,
    compare_modes,
)
from benchmarks.runner import (
    MODES,
    run_scenario,
    run_all_scenarios,
    run_benchmark,
)
from benchmarks.export import export_json, export_csv, format_summary_table
from benchmarks.scenarios import ALL_SCENARIOS
from benchmarks.scenarios.multi_tool_chain import MultiToolChainScenario
from benchmarks.scenarios.error_recovery import ErrorRecoveryScenario
from benchmarks.scenarios.complex_schema import ComplexSchemaScenario


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------

class TestEnrichmentMetrics:
    """Test the EnrichmentMetrics dataclass."""

    def test_overall_score_with_scores(self):
        m = EnrichmentMetrics(
            mode="full",
            scenario="test",
            scores={"a": 0.8, "b": 0.6},
        )
        assert m.overall_score == pytest.approx(0.7)

    def test_overall_score_empty(self):
        m = EnrichmentMetrics(mode="passthrough", scenario="test")
        assert m.overall_score == 0.0

    def test_to_dict_contains_all_fields(self):
        m = EnrichmentMetrics(
            mode="structural",
            scenario="test_scenario",
            scores={"structural": 0.5},
            enrichment_time_ms=12.5,
            tool_count=3,
            field_counts={"_manifest": 3},
        )
        d = m.to_dict()
        assert d["mode"] == "structural"
        assert d["scenario"] == "test_scenario"
        assert d["overall_score"] == 0.5
        assert d["enrichment_time_ms"] == 12.5
        assert d["tool_count"] == 3
        assert d["field_counts"]["_manifest"] == 3

    def test_to_dict_rounds_values(self):
        m = EnrichmentMetrics(
            mode="full",
            scenario="test",
            scores={"a": 1.0 / 3.0},
            enrichment_time_ms=1.23456789,
        )
        d = m.to_dict()
        assert d["overall_score"] == round(1.0 / 3.0, 4)
        assert d["enrichment_time_ms"] == 1.23


class TestCountFields:
    """Test field counting logic."""

    def test_counts_present_fields(self):
        tools = [
            {"name": "A", "_manifest": {}, "_links": {}},
            {"name": "B", "_manifest": {}},
        ]
        counts = count_fields(tools, ["_manifest", "_links", "_quality"])
        assert counts["_manifest"] == 2
        assert counts["_links"] == 1
        assert counts["_quality"] == 0

    def test_empty_tools(self):
        counts = count_fields([], ["_manifest"])
        assert counts["_manifest"] == 0

    def test_empty_fields(self):
        counts = count_fields([{"name": "A"}], [])
        assert counts == {}


class TestScoreFieldCompleteness:
    """Test field completeness scoring."""

    def test_perfect_score(self):
        tools = [
            {"name": "A", "x": 1, "y": 2},
            {"name": "B", "x": 1, "y": 2},
        ]
        assert score_field_completeness(tools, ["x", "y"]) == 1.0

    def test_zero_score(self):
        tools = [{"name": "A"}, {"name": "B"}]
        assert score_field_completeness(tools, ["x", "y"]) == 0.0

    def test_partial_score(self):
        tools = [
            {"name": "A", "x": 1},  # 1 of 2
            {"name": "B", "x": 1, "y": 2},  # 2 of 2
        ]
        # 3 out of 4 possible
        assert score_field_completeness(tools, ["x", "y"]) == pytest.approx(0.75)

    def test_empty_tools_returns_zero(self):
        assert score_field_completeness([], ["x"]) == 0.0

    def test_empty_fields_returns_zero(self):
        assert score_field_completeness([{"name": "A"}], []) == 0.0


class TestScoreScenario:
    """Test scenario scoring."""

    def test_scores_contain_expected_keys(self):
        tools = [{"name": "A", "_manifest": {}, "behavioral_what": "test"}]
        expected = {"A": ["_manifest", "behavioral_what"]}
        scores = score_scenario(tools, expected)
        assert "structural" in scores
        assert "behavioral" in scores
        assert "scenario_specific" in scores

    def test_scenario_specific_perfect(self):
        tools = [{"name": "A", "x": 1, "y": 2}]
        expected = {"A": ["x", "y"]}
        scores = score_scenario(tools, expected)
        assert scores["scenario_specific"] == 1.0

    def test_scenario_specific_zero(self):
        tools = [{"name": "A"}]
        expected = {"A": ["x", "y"]}
        scores = score_scenario(tools, expected)
        assert scores["scenario_specific"] == 0.0

    def test_empty_expected(self):
        tools = [{"name": "A"}]
        scores = score_scenario(tools, {})
        assert "structural" in scores
        assert "behavioral" in scores
        assert "scenario_specific" not in scores


class TestCompareModes:
    """Test cross-mode comparison."""

    def test_ranks_modes_by_score(self):
        metrics = [
            EnrichmentMetrics(mode="passthrough", scenario="s1", scores={"a": 0.0}),
            EnrichmentMetrics(mode="structural", scenario="s1", scores={"a": 0.5}),
            EnrichmentMetrics(mode="full", scenario="s1", scores={"a": 1.0}),
        ]
        comparison = compare_modes(metrics)
        assert comparison["full"]["rank"] == 1
        assert comparison["structural"]["rank"] == 2
        assert comparison["passthrough"]["rank"] == 3

    def test_averages_across_scenarios(self):
        metrics = [
            EnrichmentMetrics(mode="full", scenario="s1", scores={"a": 0.8}),
            EnrichmentMetrics(mode="full", scenario="s2", scores={"a": 0.6}),
        ]
        comparison = compare_modes(metrics)
        assert comparison["full"]["avg_overall_score"] == pytest.approx(0.7)
        assert comparison["full"]["scenario_count"] == 2


# ---------------------------------------------------------------------------
# Scenario unit tests
# ---------------------------------------------------------------------------

class TestScenarios:
    """Test that each scenario is well-formed."""

    @pytest.mark.parametrize("scenario_class", ALL_SCENARIOS)
    def test_scenario_has_name(self, scenario_class):
        s = scenario_class()
        assert isinstance(s.name, str)
        assert len(s.name) > 0

    @pytest.mark.parametrize("scenario_class", ALL_SCENARIOS)
    def test_scenario_has_description(self, scenario_class):
        s = scenario_class()
        assert isinstance(s.description, str)
        assert len(s.description) > 0

    @pytest.mark.parametrize("scenario_class", ALL_SCENARIOS)
    def test_scenario_returns_tools(self, scenario_class):
        s = scenario_class()
        tools = s.get_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    @pytest.mark.parametrize("scenario_class", ALL_SCENARIOS)
    def test_passthrough_expects_no_enrichment(self, scenario_class):
        s = scenario_class()
        expected = s.get_expected_fields("passthrough")
        for tool_name, fields in expected.items():
            assert fields == [], f"{tool_name} should have no expected fields in passthrough mode"

    @pytest.mark.parametrize("scenario_class", ALL_SCENARIOS)
    def test_structural_expects_structural_fields(self, scenario_class):
        s = scenario_class()
        expected = s.get_expected_fields("structural")
        for tool_name, fields in expected.items():
            for f in fields:
                assert f not in BEHAVIORAL_FIELDS, (
                    f"{tool_name} should not expect behavioral field {f} in structural mode"
                )

    @pytest.mark.parametrize("scenario_class", ALL_SCENARIOS)
    def test_full_is_superset_of_structural(self, scenario_class):
        s = scenario_class()
        structural = s.get_expected_fields("structural")
        full = s.get_expected_fields("full")
        for tool_name in structural:
            assert tool_name in full
            for f in structural[tool_name]:
                assert f in full[tool_name], (
                    f"Full mode for {tool_name} is missing structural field {f}"
                )


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------

class TestRunner:
    """Test the benchmark runner."""

    def test_run_scenario_returns_metrics(self):
        scenario = MultiToolChainScenario()
        metrics = run_scenario(scenario, "full")
        assert isinstance(metrics, EnrichmentMetrics)
        assert metrics.mode == "full"
        assert metrics.scenario == "multi_tool_chain"
        assert metrics.tool_count == 4
        assert metrics.enrichment_time_ms >= 0

    def test_run_all_scenarios_returns_correct_count(self):
        results = run_all_scenarios()
        expected_count = len(ALL_SCENARIOS) * len(MODES)
        assert len(results) == expected_count

    def test_run_all_scenarios_covers_all_modes(self):
        results = run_all_scenarios()
        modes_seen = {m.mode for m in results}
        assert modes_seen == set(MODES)

    def test_run_all_scenarios_covers_all_scenarios(self):
        results = run_all_scenarios()
        scenarios_seen = {m.scenario for m in results}
        expected_scenarios = {s().name for s in ALL_SCENARIOS}
        assert scenarios_seen == expected_scenarios

    def test_run_benchmark_returns_structured_result(self):
        result = run_benchmark()
        assert "results" in result
        assert "comparison" in result
        assert "metadata" in result
        assert result["metadata"]["scenario_count"] == len(ALL_SCENARIOS)
        assert result["metadata"]["mode_count"] == len(MODES)


class TestModeOrdering:
    """The core proof: full > structural > passthrough in enrichment quality."""

    def test_full_scores_higher_than_structural(self):
        """Full mode must score higher than structural across all scenarios."""
        results = run_all_scenarios()
        for scenario_class in ALL_SCENARIOS:
            scenario_name = scenario_class().name
            full = [m for m in results if m.scenario == scenario_name and m.mode == "full"]
            structural = [m for m in results if m.scenario == scenario_name and m.mode == "structural"]
            assert len(full) == 1
            assert len(structural) == 1
            assert full[0].overall_score > structural[0].overall_score, (
                f"Full mode ({full[0].overall_score:.4f}) should score higher than "
                f"structural ({structural[0].overall_score:.4f}) for {scenario_name}"
            )

    def test_structural_scores_higher_than_passthrough(self):
        """Structural mode must score higher than passthrough across all scenarios."""
        results = run_all_scenarios()
        for scenario_class in ALL_SCENARIOS:
            scenario_name = scenario_class().name
            structural = [m for m in results if m.scenario == scenario_name and m.mode == "structural"]
            passthrough = [m for m in results if m.scenario == scenario_name and m.mode == "passthrough"]
            assert len(structural) == 1
            assert len(passthrough) == 1
            assert structural[0].overall_score > passthrough[0].overall_score, (
                f"Structural mode ({structural[0].overall_score:.4f}) should score higher than "
                f"passthrough ({passthrough[0].overall_score:.4f}) for {scenario_name}"
            )

    def test_passthrough_scores_zero(self):
        """Passthrough mode should have zero enrichment scores."""
        results = run_all_scenarios()
        passthrough_results = [m for m in results if m.mode == "passthrough"]
        for m in passthrough_results:
            assert m.scores["structural"] == 0.0, (
                f"Passthrough structural score should be 0.0 for {m.scenario}"
            )
            assert m.scores["behavioral"] == 0.0, (
                f"Passthrough behavioral score should be 0.0 for {m.scenario}"
            )

    def test_full_mode_has_behavioral_fields(self):
        """Full mode must produce behavioral enrichment fields."""
        results = run_all_scenarios()
        full_results = [m for m in results if m.mode == "full"]
        for m in full_results:
            behavioral_total = sum(
                m.field_counts.get(f, 0) for f in BEHAVIORAL_FIELDS
            )
            assert behavioral_total > 0, (
                f"Full mode should have behavioral fields for {m.scenario}"
            )

    def test_structural_mode_has_no_behavioral_fields(self):
        """Structural mode must NOT produce behavioral enrichment fields."""
        results = run_all_scenarios()
        structural_results = [m for m in results if m.mode == "structural"]
        for m in structural_results:
            behavioral_total = sum(
                m.field_counts.get(f, 0) for f in BEHAVIORAL_FIELDS
            )
            assert behavioral_total == 0, (
                f"Structural mode should not have behavioral fields for {m.scenario}"
            )


class TestDeterminism:
    """Benchmark results must be deterministic."""

    def test_same_input_same_output(self):
        """Running the same benchmark twice produces identical scores."""
        result1 = run_benchmark()
        result2 = run_benchmark()

        for r1, r2 in zip(result1["results"], result2["results"]):
            assert r1["mode"] == r2["mode"]
            assert r1["scenario"] == r2["scenario"]
            assert r1["overall_score"] == r2["overall_score"]
            assert r1["scores"] == r2["scores"]
            assert r1["field_counts"] == r2["field_counts"]


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

class TestExportJson:
    """Test JSON export."""

    def test_export_produces_valid_json(self):
        results = run_benchmark()
        json_str = export_json(results)
        parsed = json.loads(json_str)
        assert "results" in parsed
        assert "comparison" in parsed

    def test_export_to_file(self, tmp_path):
        results = run_benchmark()
        path = str(tmp_path / "results.json")
        export_json(results, path=path)
        with open(path) as f:
            parsed = json.loads(f.read())
        assert len(parsed["results"]) == len(ALL_SCENARIOS) * len(MODES)


class TestExportCsv:
    """Test CSV export."""

    def test_export_produces_valid_csv(self):
        results = run_benchmark()
        csv_str = export_csv(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        # Header + data rows
        assert len(rows) == 1 + len(ALL_SCENARIOS) * len(MODES)

    def test_csv_header_matches_schema(self):
        results = run_benchmark()
        csv_str = export_csv(results)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header == [
            "mode",
            "scenario",
            "overall_score",
            "structural_score",
            "behavioral_score",
            "scenario_specific_score",
            "enrichment_time_ms",
            "tool_count",
        ]

    def test_export_to_file(self, tmp_path):
        results = run_benchmark()
        path = str(tmp_path / "results.csv")
        export_csv(results, path=path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) > 1


class TestFormatSummaryTable:
    """Test human-readable summary formatting."""

    def test_summary_contains_all_modes(self):
        results = run_benchmark()
        table = format_summary_table(results)
        assert "passthrough" in table
        assert "structural" in table
        assert "full" in table

    def test_summary_contains_all_scenarios(self):
        results = run_benchmark()
        table = format_summary_table(results)
        for scenario_class in ALL_SCENARIOS:
            assert scenario_class().name in table
