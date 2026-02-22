"""Pattern 14: Anti-patterns — known failure modes per tool.

Documents what NOT to do with each tool. This is critical for models
without RL training on tool use, as they don't have negative examples
baked into their weights.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class AntiPatternsApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 14: Anti-Patterns.

    Adds documented failure modes and anti-patterns to each tool so
    the model avoids common mistakes that trained models learn to avoid.
    """

    TOOL_ANTI_PATTERNS: dict[str, list[dict[str, str]]] = {
        "Read": [
            {
                "anti_pattern": "Using Bash cat/head/tail instead of Read",
                "why_bad": "Loses line number context and permission integration",
                "do_instead": "Use Read tool with file_path parameter",
            },
        ],
        "Edit": [
            {
                "anti_pattern": "Editing without reading first",
                "why_bad": "old_string will not match actual file content",
                "do_instead": "Always Read the file before Edit",
            },
            {
                "anti_pattern": "Using Write to make small changes",
                "why_bad": "Overwrites entire file, losing content you didn't include",
                "do_instead": "Use Edit for surgical changes to existing files",
            },
        ],
        "Write": [
            {
                "anti_pattern": "Creating files proactively without being asked",
                "why_bad": "Creates unnecessary file bloat and documentation noise",
                "do_instead": "Only create files when explicitly required by the task",
            },
            {
                "anti_pattern": "Overwriting existing files without reading them",
                "why_bad": "Loses all content not in your write",
                "do_instead": "Read first, prefer Edit for existing files",
            },
        ],
        "Bash": [
            {
                "anti_pattern": "Using Bash for file operations (cat, sed, grep, find, echo >)",
                "why_bad": "Bypasses dedicated tools that have safety features and permission integration",
                "do_instead": "Use Read, Edit, Write, Grep, Glob respectively",
            },
            {
                "anti_pattern": "Running destructive git commands without explicit user request",
                "why_bad": "push --force, reset --hard, clean -f can permanently destroy work",
                "do_instead": "Only run destructive commands when the user explicitly asks",
            },
            {
                "anti_pattern": "Using interactive flags (-i) with commands",
                "why_bad": "Interactive input is not supported — command will hang",
                "do_instead": "Use non-interactive alternatives",
            },
        ],
        "Grep": [
            {
                "anti_pattern": "Using Bash grep/rg instead of Grep tool",
                "why_bad": "Loses permission integration and optimized output formatting",
                "do_instead": "Use the Grep tool with pattern and optional glob filter",
            },
        ],
        "Glob": [
            {
                "anti_pattern": "Using Bash find instead of Glob",
                "why_bad": "Slower and loses integration with the tool system",
                "do_instead": "Use Glob with pattern matching",
            },
        ],
    }

    @property
    def pattern_number(self) -> int:
        return 14

    @property
    def name(self) -> str:
        return "Anti-Patterns"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add anti-pattern documentation to tool definitions."""
        enriched = copy.deepcopy(tools)
        for tool in enriched:
            tool_name = tool.get("name", "")
            anti_patterns = self.TOOL_ANTI_PATTERNS.get(tool_name)
            if anti_patterns:
                tool["_anti_patterns"] = anti_patterns
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have anti-pattern documentation."""
        issues = []
        for tool in tools:
            tool_name = tool.get("name", "<unnamed>")
            if tool_name in self.TOOL_ANTI_PATTERNS and "_anti_patterns" not in tool:
                issues.append({
                    "tool": tool_name,
                    "issue": "Missing _anti_patterns documentation",
                    "severity": "warning",
                })
        return issues
