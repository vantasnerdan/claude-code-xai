"""Tool definition translation between Anthropic and OpenAI formats.

Forward: Anthropic {name, description, input_schema} ->
         OpenAI {type: "function", function: {name, description, parameters}}

Enrichment injection point: hook to modify tool definitions before translation.
"""

from __future__ import annotations

from typing import Any, Callable

from bridge.logging_config import get_logger

logger = get_logger("forward")


# --- Enrichment Hook Type ---
# A callable that receives the tool list and returns a modified tool list.
# Set via set_tool_enrichment_hook() before translation.
ToolEnrichmentHook = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

_tool_enrichment_hook: ToolEnrichmentHook | None = None


def set_tool_enrichment_hook(hook: ToolEnrichmentHook | None) -> None:
    """Register an enrichment hook for tool definitions.

    The hook receives Anthropic-format tool definitions and returns
    modified definitions. It runs BEFORE format translation.
    """
    global _tool_enrichment_hook
    _tool_enrichment_hook = hook
    logger.debug("Tool enrichment hook %s", "registered" if hook else "cleared")


def translate_tools(
    anthropic_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Anthropic tool definitions to OpenAI function format.

    Args:
        anthropic_tools: List of Anthropic tool defs with input_schema.

    Returns:
        List of OpenAI function tool defs with parameters.
    """
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

    result: list[dict[str, Any]] = []
    for tool in tools:
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
