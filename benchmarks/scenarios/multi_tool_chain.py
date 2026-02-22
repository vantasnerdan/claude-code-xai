"""Scenario: Multi-tool chain — File search, read, edit workflow.

Tests whether enrichment adds sequencing guidance (WHEN dimension),
navigation links (HATEOAS), and prerequisite documentation.

The chain: Glob -> Grep -> Read -> Edit
This is the canonical file modification workflow in Claude Code.
"""
from typing import Any

from benchmarks.scenarios.base import BenchmarkScenario


class MultiToolChainScenario(BenchmarkScenario):
    """Multi-tool chain scenario: Glob -> Grep -> Read -> Edit.

    Tests:
    - WHEN dimension: Does enrichment document the sequencing?
    - P2 (HATEOAS): Does enrichment add navigation links between tools?
    - P1 (Manifest): Does enrichment add collection-level metadata?
    - P15 (Registration): Does enrichment add tool registration data?
    """

    @property
    def name(self) -> str:
        return "multi_tool_chain"

    @property
    def description(self) -> str:
        return (
            "File search -> read -> edit chain. "
            "Tests whether enrichment adds sequencing guidance (WHEN), "
            "navigation links (P2), and manifest metadata (P1)."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Four tools forming a natural file modification workflow."""
        return [
            {
                "name": "Glob",
                "description": "Fast file pattern matching tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The glob pattern to match files against.",
                        },
                        "path": {
                            "type": "string",
                            "description": "The directory to search in.",
                        },
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "Grep",
                "description": "Searches file contents using ripgrep.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The regex pattern to search for.",
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory to search in.",
                        },
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "Read",
                "description": "Reads a file from the local filesystem.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to read.",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "Edit",
                "description": "Performs exact string replacements in files.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to modify.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The text to replace.",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The text to replace it with.",
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        ]

    def get_expected_fields(self, mode: str) -> dict[str, list[str]]:
        """Expected fields per tool for each mode.

        Passthrough: no enrichment fields
        Structural: API standard patterns (manifest, links, registration, etc.)
        Full: structural + behavioral WHAT/WHY/WHEN
        """
        if mode == "passthrough":
            return {
                "Glob": [],
                "Grep": [],
                "Read": [],
                "Edit": [],
            }

        # Structural fields shared by structural and full modes
        structural = {
            "Glob": [
                "_manifest",        # P1
                "_links",           # P2
                "_near_miss",       # P5
                "outputSchema",     # P6
                "_anti_patterns",   # P14
                "_registration",    # P15
            ],
            "Grep": [
                "_manifest",
                "_links",
                "_near_miss",
                "outputSchema",
                "_quality",         # P8
                "_anti_patterns",
                "_registration",
            ],
            "Read": [
                "_manifest",
                "_links",
                "_error_format",    # P3
                "_near_miss",
                "outputSchema",
                "_quality",
                "_anti_patterns",
                "_registration",
            ],
            "Edit": [
                "_manifest",
                "_links",
                "_error_format",
                "_near_miss",
                "outputSchema",
                "_quality",
                "_anti_patterns",
                "_registration",
            ],
        }

        if mode == "structural":
            return structural

        # Full: structural + behavioral
        full = {}
        for tool_name, fields in structural.items():
            full[tool_name] = fields + [
                "behavioral_what",
                "behavioral_why",
                "behavioral_when",
            ]
        return full
