"""Benchmark scenarios for enrichment quality testing.

Each scenario defines:
- A set of tool definitions (Anthropic format)
- Expected enrichment outcomes per mode
- Scoring criteria
"""

from benchmarks.scenarios.multi_tool_chain import MultiToolChainScenario
from benchmarks.scenarios.error_recovery import ErrorRecoveryScenario
from benchmarks.scenarios.complex_schema import ComplexSchemaScenario

ALL_SCENARIOS = [
    MultiToolChainScenario,
    ErrorRecoveryScenario,
    ComplexSchemaScenario,
]

__all__ = [
    "ALL_SCENARIOS",
    "MultiToolChainScenario",
    "ErrorRecoveryScenario",
    "ComplexSchemaScenario",
]
