"""Pattern 1: Manifest metadata.

Adds manifest-level metadata to the tool collection — version, capabilities,
and tool count — so the consuming model knows what it's working with.
"""
import copy
from typing import Any

from enrichment.structural.base import PatternApplicator


class ManifestApplicator(PatternApplicator):
    """Applies Agentic API Standard Pattern 1: Manifest.

    Ensures tool definitions include manifest metadata that describes
    the overall tool collection (version, count, source).

    Args:
        manifest_data: Static manifest fields from YAML. When None, uses built-in defaults.
    """

    _DEFAULTS: dict[str, str] = {
        "source": "claude-code-xai-bridge",
        "enrichment_version": "0.1.0",
    }

    def __init__(self, manifest_data: dict[str, str] | None = None) -> None:
        self._manifest_fields = manifest_data if manifest_data is not None else self._DEFAULTS

    @property
    def pattern_number(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "Manifest"

    def apply(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add manifest metadata to tool definitions.

        Adds a '_manifest' key to each tool with collection-level metadata.
        tool_count is always computed from the input list.
        """
        enriched = copy.deepcopy(tools)
        manifest = {
            "tool_count": len(enriched),
            **self._manifest_fields,
        }
        for tool in enriched:
            tool["_manifest"] = manifest
        return enriched

    def validate(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that tools have manifest metadata."""
        issues = []
        for tool in tools:
            if "_manifest" not in tool:
                issues.append({
                    "tool": tool.get("name", "<unnamed>"),
                    "issue": "Missing _manifest metadata",
                    "severity": "warning",
                })
        return issues
