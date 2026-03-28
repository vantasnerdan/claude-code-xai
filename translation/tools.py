"""Tool definition translation between Anthropic and xAI formats.

Two output formats:
- Chat Completions: {type: "function", function: {name, description, parameters}}
- Responses API: {type: "function", name: ..., description: ..., parameters: ...}

Both share the same enrichment pipeline (hook + folding + cleanup).
The core enrichment runs once; formatting is the final step.

Enrichment injection point: hook to modify tool definitions before translation.
Measures enrichment overhead for token logging (Issue #26).
"""

from __future__ import annotations

from typing import Any, Callable

from bridge.logging_config import get_logger
from bridge.token_logger import measure_enrichment_overhead
from translation.enrichment_folding import (
    fold_enrichment_into_description,
    _remove_remaining_enrichment_fields,
)

logger = get_logger("forward")


# --- Enrichment Hook Type ---
# A callable that receives the tool list and returns a modified tool list.
# Set via set_tool_enrichment_hook() before translation.
ToolEnrichmentHook = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

_tool_enrichment_hook: ToolEnrichmentHook | None = None

# Last measured enrichment overhead in estimated tokens.
# Updated on every enrich_tools() call where enrichment runs.
# Read by main.py to include in token usage logging.
_last_enrichment_overhead: int = 0


def set_tool_enrichment_hook(hook: ToolEnrichmentHook | None) -> None:
    """Register an enrichment hook for tool definitions.

    The hook receives Anthropic-format tool definitions and returns
    modified definitions. It runs BEFORE format translation.
    """
    global _tool_enrichment_hook
    _tool_enrichment_hook = hook
    logger.debug("Tool enrichment hook %s", "registered" if hook else "cleared")


def get_last_enrichment_overhead() -> int:
    """Return the enrichment overhead from the most recent enrich_tools() call.

    Returns estimated token count added by enrichment. Resets to 0 on
    each enrich_tools() call before measurement.
    """
    return _last_enrichment_overhead


def reset_enrichment_overhead() -> None:
    """Reset the enrichment overhead counter to zero.

    Call at the start of each request to prevent stale values from
    a previous request leaking into the current one when no tools
    are present (translate_tools is not called for tool-free requests).
    """
    global _last_enrichment_overhead
    _last_enrichment_overhead = 0


def enrich_tools(
    anthropic_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run enrichment pipeline on Anthropic tool definitions.

    This is the shared core: enrichment hook + description folding +
    cleanup of enrichment-only fields. The result is a list of dicts
    with {name, description, input_schema} -- format-neutral, ready
    for either Chat Completions or Responses API formatting.

    Args:
        anthropic_tools: List of Anthropic tool defs with input_schema.

    Returns:
        Enriched tool defs (name, description, input_schema only).
    """
    global _last_enrichment_overhead
    _last_enrichment_overhead = 0

    if not anthropic_tools:
        return []

    # --- Enrichment injection point ---
    tools = anthropic_tools
    if _tool_enrichment_hook is not None:
        logger.debug(
            "Running enrichment hook on %d tools",
            len(anthropic_tools),
        )
        tools = _tool_enrichment_hook(tools)
        _last_enrichment_overhead = measure_enrichment_overhead(
            anthropic_tools, tools,
        )

    # Fold enrichment metadata into descriptions so the guest model
    # actually receives the data. Both xAI formats only carry
    # name/description/parameters -- everything else is dropped.
    fold_enrichment_into_description(tools)

    for tool in tools:
        _remove_remaining_enrichment_fields(tool)

    return tools


def translate_tools(
    anthropic_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Anthropic tool definitions to OpenAI Chat Completions format.

    Chat Completions format: {type: "function", function: {name, description, parameters}}

    Args:
        anthropic_tools: List of Anthropic tool defs with input_schema.

    Returns:
        List of OpenAI function tool defs with parameters.
    """
    enriched = enrich_tools(anthropic_tools)

    result: list[dict[str, Any]] = []
    for tool in enriched:
        openai_tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        result.append(openai_tool)

    return result


def translate_tools_responses(
    anthropic_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Anthropic tool definitions to xAI Responses API format.

    Responses API format: {type: "function", name: ..., description: ..., parameters: ...}
    Flat structure -- no nested 'function' key.

    Args:
        anthropic_tools: List of Anthropic tool defs with input_schema.

    Returns:
        List of Responses API function tool defs.
    """
    enriched = enrich_tools(anthropic_tools)

    result: list[dict[str, Any]] = []
    for tool in enriched:
        result.append({
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        })

    return result
