"""Scenario: Error recovery — Tools prone to misuse.

Tests whether enrichment adds error format (P3), near-miss suggestions (P5),
anti-patterns (P14), and behavioral failure modes (WHY dimension).

Focus: Edit and Write are the most error-prone Claude Code tools.
Models without RL training commonly misuse them.
"""
from typing import Any

from benchmarks.scenarios.base import BenchmarkScenario


class ErrorRecoveryScenario(BenchmarkScenario):
    """Error recovery scenario: Edit, Write, Bash.

    Tests:
    - P3 (Error Format): Are common errors and suggestions documented?
    - P5 (Near-Miss): Are aliases and confusion targets documented?
    - P14 (Anti-Patterns): Are known failure modes documented?
    - WHY dimension: Are failure modes from behavioral knowledge present?
    """

    @property
    def name(self) -> str:
        return "error_recovery"

    @property
    def description(self) -> str:
        return (
            "Error-prone tool definitions. "
            "Tests whether enrichment adds error format (P3), "
            "near-miss suggestions (P5), anti-patterns (P14), "
            "and behavioral failure modes (WHY)."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Three tools that are commonly misused without RL training."""
        return [
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
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace all occurrences.",
                            "default": False,
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
            {
                "name": "Write",
                "description": "Writes a file to the local filesystem.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "Bash",
                "description": "Executes a given bash command.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Optional timeout in milliseconds.",
                        },
                    },
                    "required": ["command"],
                },
            },
        ]

    def get_expected_fields(self, mode: str) -> dict[str, list[str]]:
        """Expected fields emphasizing error handling and failure prevention."""
        if mode == "passthrough":
            return {
                "Edit": [],
                "Write": [],
                "Bash": [],
            }

        structural = {
            "Edit": [
                "_manifest",        # P1
                "_links",           # P2
                "_error_format",    # P3 -- critical for this scenario
                "_near_miss",       # P5
                "outputSchema",     # P6
                "_quality",         # P8
                "_anti_patterns",   # P14 -- critical for this scenario
                "_registration",    # P15
            ],
            "Write": [
                "_manifest",
                "_links",
                "_error_format",    # P3
                "_near_miss",
                "outputSchema",
                "_anti_patterns",   # P14
                "_registration",
            ],
            "Bash": [
                "_manifest",
                "_links",
                "_error_format",    # P3
                "_near_miss",
                "outputSchema",
                "_quality",         # P8
                "_anti_patterns",   # P14
                "_registration",
            ],
        }

        if mode == "structural":
            return structural

        # Full: structural + behavioral (WHY is critical for error recovery)
        full = {}
        for tool_name, fields in structural.items():
            full[tool_name] = fields + [
                "behavioral_what",
                "behavioral_why",   # Critical: failure_modes from training
                "behavioral_when",
            ]
        return full
