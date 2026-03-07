"""Fold enrichment metadata into tool descriptions for OpenAI format.

The OpenAI function-calling format only carries name, description, and
parameters to the model. All other fields on the tool dict are silently
discarded by the API. This module serializes enrichment metadata
(behavioral dimensions, structural patterns) into the description field
so the guest model actually receives the enrichment data.

This runs AFTER the enrichment engine and BEFORE format translation.
"""

from __future__ import annotations

import json
from typing import Any


# Enrichment field keys added by the engine, grouped by category.
_BEHAVIORAL_FIELDS = ("behavioral_what", "behavioral_why", "behavioral_when")
_STRUCTURAL_FIELDS = (
    "_links",
    "_near_miss",
    "_error_format",
    "_quality",
    "_anti_patterns",
    "outputSchema",
    "_manifest",
    "_registration",
)

# Fields that are part of the base Anthropic tool schema (not enrichment).
_BASE_FIELDS = frozenset({"name", "description", "input_schema", "type", "cache_control"})


def fold_enrichment_into_description(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append enrichment metadata to each tool's description.

    Modifies tools in-place (caller is expected to pass a deep copy
    from the enrichment engine). Returns the same list for chaining.

    Each enrichment field is serialized as a labeled section appended
    to the original description. The format is designed to be readable
    by any LLM without special parsing.
    """
    for tool in tools:
        sections: list[str] = []

        # --- Behavioral dimensions ---
        _fold_behavioral_what(tool, sections)
        _fold_behavioral_why(tool, sections)
        _fold_behavioral_when(tool, sections)

        # --- Structural patterns ---
        _fold_links(tool, sections)
        _fold_error_format(tool, sections)
        _fold_near_miss(tool, sections)
        _fold_quality(tool, sections)
        _fold_anti_patterns(tool, sections)
        _fold_output_schema(tool, sections)

        if sections:
            original = tool.get("description", "")
            enrichment_text = "\n".join(sections)
            tool["description"] = (
                f"{original}\n\n{enrichment_text}" if original else enrichment_text
            )

    return tools


def _fold_behavioral_what(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold enhanced WHAT description."""
    what = tool.pop("behavioral_what", None)
    if what:
        sections.append(f"[Enhanced Description]\n{what}")


def _fold_behavioral_why(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold WHY context (problem context + failure modes)."""
    why = tool.pop("behavioral_why", None)
    if not why:
        return
    parts: list[str] = []
    if isinstance(why, dict):
        if "problem_context" in why:
            parts.append(f"Problem context: {why['problem_context']}")
        if "failure_modes" in why:
            modes = why["failure_modes"]
            if isinstance(modes, list):
                for mode in modes:
                    parts.append(f"- {mode}")
            elif isinstance(modes, str):
                parts.append(f"Failure modes: {modes}")
    elif isinstance(why, str):
        parts.append(why)
    if parts:
        sections.append(f"[Why Use This Tool]\n" + "\n".join(parts))


def _fold_behavioral_when(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold WHEN context (prerequisites, sequencing, alternatives)."""
    when = tool.pop("behavioral_when", None)
    if not when:
        return
    parts: list[str] = []
    if isinstance(when, dict):
        if "prerequisites" in when:
            prereqs = when["prerequisites"]
            if isinstance(prereqs, list):
                parts.append(f"Prerequisites: {', '.join(prereqs)}")
            else:
                parts.append(f"Prerequisites: {prereqs}")
        if "use_before" in when:
            parts.append(f"Use before: {', '.join(when['use_before'])}")
        if "use_instead_of" in when:
            parts.append(f"Use instead of: {', '.join(when['use_instead_of'])}")
        if "do_not_use_for" in when:
            items = when["do_not_use_for"]
            if isinstance(items, list):
                for item in items:
                    parts.append(f"- Do NOT use for: {item}")
            else:
                parts.append(f"Do NOT use for: {items}")
        if "sequencing" in when:
            parts.append(f"Sequencing: {when['sequencing']}")
    elif isinstance(when, str):
        parts.append(when)
    if parts:
        sections.append(f"[When To Use]\n" + "\n".join(parts))


def _fold_links(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold HATEOAS links (related tools, error recovery)."""
    links = tool.pop("_links", None)
    if not links:
        return
    parts: list[str] = []
    if isinstance(links, dict):
        if "related" in links:
            related = links["related"]
            if isinstance(related, list):
                parts.append(f"Related tools: {', '.join(related)}")
        if "on_error" in links:
            on_error = links["on_error"]
            if isinstance(on_error, dict):
                for err_type, suggestion in on_error.items():
                    parts.append(f"On {err_type}: {suggestion}")
    if parts:
        sections.append(f"[Navigation]\n" + "\n".join(parts))


def _fold_error_format(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold error format documentation."""
    error_fmt = tool.pop("_error_format", None)
    if not error_fmt:
        return
    parts: list[str] = []
    if isinstance(error_fmt, dict):
        errors = error_fmt.get("errors", [])
        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, dict):
                    error_str = err.get("error", "")
                    suggestion = err.get("suggestion", "")
                    parts.append(f"- {error_str}: {suggestion}")
    if parts:
        sections.append(f"[Error Handling]\n" + "\n".join(parts))


def _fold_near_miss(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold near-miss alias data."""
    near_miss = tool.pop("_near_miss", None)
    if not near_miss:
        return
    parts: list[str] = []
    if isinstance(near_miss, dict):
        if "aliases" in near_miss:
            parts.append(f"Also known as: {', '.join(near_miss['aliases'])}")
        if "commonly_confused_with" in near_miss:
            confused = near_miss["commonly_confused_with"]
            if confused:
                parts.append(f"Commonly confused with: {', '.join(confused)}")
    if parts:
        sections.append(f"[Aliases]\n" + "\n".join(parts))


def _fold_quality(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold quality gate data."""
    quality = tool.pop("_quality", None)
    if not quality:
        return
    if isinstance(quality, dict):
        parts: list[str] = []
        for key, value in quality.items():
            if isinstance(value, str):
                parts.append(f"{key}: {value}")
            else:
                parts.append(f"{key}: {json.dumps(value, default=str)}")
        if parts:
            sections.append(f"[Quality]\n" + "\n".join(parts))


def _fold_anti_patterns(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold anti-pattern documentation."""
    anti = tool.pop("_anti_patterns", None)
    if not anti:
        return
    parts: list[str] = []
    if isinstance(anti, list):
        for item in anti:
            if isinstance(item, dict):
                pattern = item.get("anti_pattern", "")
                why_bad = item.get("why_bad", "")
                do_instead = item.get("do_instead", "")
                parts.append(f"- AVOID: {pattern}")
                if why_bad:
                    parts.append(f"  Why: {why_bad}")
                if do_instead:
                    parts.append(f"  Instead: {do_instead}")
    if parts:
        sections.append(f"[Anti-Patterns]\n" + "\n".join(parts))


def _fold_output_schema(tool: dict[str, Any], sections: list[str]) -> None:
    """Fold self-describing output schema."""
    schema = tool.pop("outputSchema", None)
    if not schema:
        return
    sections.append(
        f"[Output Schema]\n{json.dumps(schema, indent=2, default=str)}"
    )


def _remove_remaining_enrichment_fields(tool: dict[str, Any]) -> None:
    """Remove any enrichment fields not handled by specific folders.

    Safety net: ensures no enrichment-only fields leak into the OpenAI
    format where they would be silently discarded.
    """
    tool.pop("_manifest", None)
    tool.pop("_registration", None)
