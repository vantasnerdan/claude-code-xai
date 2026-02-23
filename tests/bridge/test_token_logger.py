"""Tests for bridge.token_logger -- token count logging (Issue #26).

Verifies token estimation, enrichment overhead measurement, and
structured INFO-level token usage logging.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from bridge.token_logger import (
    estimate_tokens,
    log_token_usage,
    measure_enrichment_overhead,
)


class TestEstimateTokens:
    """Token estimation from character count."""

    def test_empty_string_returns_one(self) -> None:
        assert estimate_tokens("") == 1

    def test_short_text(self) -> None:
        # 12 chars -> 3 tokens
        assert estimate_tokens("hello world!") == 3

    def test_longer_text(self) -> None:
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_single_char_returns_one(self) -> None:
        assert estimate_tokens("x") == 1


class TestMeasureEnrichmentOverhead:
    """Enrichment overhead measurement via serialized JSON delta."""

    def test_no_tools_returns_zero(self) -> None:
        assert measure_enrichment_overhead([], []) == 0

    def test_identical_tools_returns_zero(self) -> None:
        tools: list[dict[str, Any]] = [
            {"name": "Read", "description": "Reads files."},
        ]
        assert measure_enrichment_overhead(tools, tools) == 0

    def test_enriched_tools_returns_positive(self) -> None:
        original: list[dict[str, Any]] = [
            {"name": "Read", "description": "Reads files."},
        ]
        enriched: list[dict[str, Any]] = [
            {
                "name": "Read",
                "description": "Reads files.",
                "anti_patterns": ["Do not use cat or head"],
                "quality_gates": {"required_params": ["file_path"]},
            },
        ]
        overhead = measure_enrichment_overhead(original, enriched)
        assert overhead > 0

    def test_smaller_enriched_returns_zero(self) -> None:
        original: list[dict[str, Any]] = [
            {"name": "Read", "description": "A very long description that is verbose."},
        ]
        enriched: list[dict[str, Any]] = [
            {"name": "Read", "description": "Short."},
        ]
        assert measure_enrichment_overhead(original, enriched) == 0

    def test_multiple_tools_overhead(self) -> None:
        original: list[dict[str, Any]] = [
            {"name": "Read", "description": "R"},
            {"name": "Write", "description": "W"},
        ]
        enriched: list[dict[str, Any]] = [
            {"name": "Read", "description": "R", "extra_field": "x" * 200},
            {"name": "Write", "description": "W", "extra_field": "y" * 200},
        ]
        overhead = measure_enrichment_overhead(original, enriched)
        assert overhead > 50  # ~400 extra chars -> ~100 tokens


class TestLogTokenUsage:
    """Token usage logging output."""

    def test_logs_at_info_level(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="bridge.tokens"):
            log_token_usage(
                input_tokens=100,
                output_tokens=50,
                enrichment_overhead_tokens=20,
                elapsed_seconds=1.5,
                is_streaming=False,
            )
        assert len(caplog.records) >= 1
        record = caplog.records[0]
        assert record.levelno == logging.INFO
        assert "bridge.tokens" in record.name

    def test_log_contains_all_fields(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="bridge.tokens"):
            log_token_usage(
                input_tokens=100,
                output_tokens=50,
                enrichment_overhead_tokens=20,
                elapsed_seconds=1.5,
                is_streaming=False,
            )
        msg = caplog.records[0].message
        assert "input=100" in msg
        assert "output=50" in msg
        assert "total=150" in msg
        assert "enrichment_overhead=20" in msg
        assert "mode=sync" in msg
        assert "elapsed=1.50s" in msg

    def test_streaming_mode_label(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="bridge.tokens"):
            log_token_usage(
                input_tokens=200,
                output_tokens=80,
                is_streaming=True,
            )
        msg = caplog.records[0].message
        assert "mode=stream" in msg

    def test_returns_summary_dict(self) -> None:
        result = log_token_usage(
            input_tokens=100,
            output_tokens=50,
            enrichment_overhead_tokens=20,
        )
        assert result == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "enrichment_overhead_tokens": 20,
        }

    def test_zero_overhead_when_no_tools(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="bridge.tokens"):
            log_token_usage(
                input_tokens=50,
                output_tokens=25,
                enrichment_overhead_tokens=0,
            )
        msg = caplog.records[0].message
        assert "enrichment_overhead=0" in msg

    def test_total_is_input_plus_output(self) -> None:
        result = log_token_usage(
            input_tokens=300,
            output_tokens=700,
        )
        assert result["total_tokens"] == 1000
