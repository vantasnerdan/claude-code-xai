"""Scenario: Complex schemas — Nested objects, enum constraints, varied parameters.

Tests whether enrichment adds self-describing metadata (P6), quality gates (P8),
and manifest metadata for schema-heavy tools.

Focus: Read and Grep have the most complex input schemas with many optional
parameters. WebFetch and WebSearch test tools outside the core file workflow.
"""
from typing import Any

from benchmarks.scenarios.base import BenchmarkScenario


class ComplexSchemaScenario(BenchmarkScenario):
    """Complex schema scenario: Read, Grep, WebFetch, WebSearch, NotebookEdit.

    Tests:
    - P6 (Self-Describing): Are inputSchema and outputSchema present?
    - P8 (Quality Gates): Are warnings and quality flags present for tools that need them?
    - P1 (Manifest): Is collection metadata accurate for varied tool counts?
    - WHAT dimension: Do complex tools get enhanced descriptions?
    """

    @property
    def name(self) -> str:
        return "complex_schema"

    @property
    def description(self) -> str:
        return (
            "Nested objects, enum constraints, varied parameters. "
            "Tests whether enrichment adds self-describing metadata (P6), "
            "quality gates (P8), and enhanced descriptions (WHAT)."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Five tools with complex and varied schemas."""
        return [
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
                        "offset": {
                            "type": "number",
                            "description": "The line number to start reading from.",
                        },
                        "limit": {
                            "type": "number",
                            "description": "The number of lines to read.",
                        },
                        "pages": {
                            "type": "string",
                            "description": "Page range for PDF files (e.g., '1-5').",
                        },
                    },
                    "required": ["file_path"],
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
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern to filter files.",
                        },
                        "output_mode": {
                            "type": "string",
                            "enum": ["content", "files_with_matches", "count"],
                            "description": "Output mode.",
                        },
                        "multiline": {
                            "type": "boolean",
                            "description": "Enable multiline matching.",
                        },
                        "context": {
                            "type": "number",
                            "description": "Lines of context around matches.",
                        },
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "WebFetch",
                "description": "Fetches content from a URL and processes it with an AI model.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "format": "uri",
                            "description": "The URL to fetch content from.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to run on the fetched content.",
                        },
                    },
                    "required": ["url", "prompt"],
                },
            },
            {
                "name": "WebSearch",
                "description": "Searches the web and returns results with links.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use.",
                        },
                        "allowed_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only include results from these domains.",
                        },
                        "blocked_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Never include results from these domains.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "NotebookEdit",
                "description": "Replaces, inserts, or deletes cells in Jupyter notebooks.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "notebook_path": {
                            "type": "string",
                            "description": "The absolute path to the notebook.",
                        },
                        "new_source": {
                            "type": "string",
                            "description": "The new source for the cell.",
                        },
                        "cell_type": {
                            "type": "string",
                            "enum": ["code", "markdown"],
                            "description": "The type of the cell.",
                        },
                        "edit_mode": {
                            "type": "string",
                            "enum": ["replace", "insert", "delete"],
                            "description": "The type of edit to make.",
                        },
                    },
                    "required": ["notebook_path", "new_source"],
                },
            },
        ]

    def get_expected_fields(self, mode: str) -> dict[str, list[str]]:
        """Expected fields emphasizing schema completeness and quality."""
        if mode == "passthrough":
            return {
                "Read": [],
                "Grep": [],
                "WebFetch": [],
                "WebSearch": [],
                "NotebookEdit": [],
            }

        structural = {
            "Read": [
                "_manifest",        # P1
                "_links",           # P2
                "_error_format",    # P3
                "_near_miss",       # P5
                "outputSchema",     # P6 -- critical for this scenario
                "_quality",         # P8 -- critical for this scenario
                "_anti_patterns",   # P14
                "_registration",    # P15
            ],
            "Grep": [
                "_manifest",
                "_links",
                "_near_miss",
                "outputSchema",     # P6
                "_quality",         # P8
                "_anti_patterns",
                "_registration",
            ],
            "WebFetch": [
                "_manifest",
                # WebFetch has behavioral knowledge but limited structural
                # (not in HATEOAS, errors, near-miss, quality, anti-patterns maps)
                "_registration",
            ],
            "WebSearch": [
                "_manifest",
                "_registration",
            ],
            "NotebookEdit": [
                "_manifest",
                "_registration",
            ],
        }

        if mode == "structural":
            return structural

        # Full: structural + behavioral
        full = {}
        for tool_name, fields in structural.items():
            behavioral = []
            # Only tools in TOOL_KNOWLEDGE get behavioral enrichment
            if tool_name in ("Read", "Grep", "WebFetch", "WebSearch", "NotebookEdit"):
                behavioral = [
                    "behavioral_what",
                    "behavioral_why",
                    "behavioral_when",
                ]
            full[tool_name] = fields + behavioral
        return full
