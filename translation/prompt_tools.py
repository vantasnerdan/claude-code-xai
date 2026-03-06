"""Prompt-based tool calling for multi-agent models.

Multi-agent models (grok-4.20-multi-agent) do NOT support client-side
tool definitions via the API. Instead, we serialize tool schemas into the
system prompt and instruct the model to output structured <tool_call> blocks
that the bridge parses back into Anthropic tool_use content blocks.

This module handles the serialization half: tools -> prompt text.
Parsing is in translation/tool_parser.py.
"""

from __future__ import annotations

import json
from typing import Any

from bridge.logging_config import get_logger

logger = get_logger("prompt_tools")

_TOOL_CALL_INSTRUCTIONS = """\

# Tool Calling

You have access to the tools listed below. To call a tool, output a \
<tool_call> block with the tool name and parameters as JSON:

```
<tool_call>
{"name": "ToolName", "parameters": {"param1": "value1"}}
</tool_call>
```

Rules:
- You may call multiple tools in a single response by using multiple \
<tool_call> blocks.
- Parameters MUST be valid JSON matching the tool's parameter schema.
- Always include required parameters. Omit optional parameters to use defaults.
- Tool calls can appear anywhere in your response — before, after, or \
between text.
- Do NOT wrap <tool_call> blocks in markdown code fences.
- Do NOT fabricate tools that are not listed below.

## Available Tools

"""

_TOOL_RESULT_HEADER = """\
The following tool results are from your previous tool calls:

"""


def serialize_tools_to_prompt(
    tools: list[dict[str, Any]],
) -> str:
    """Serialize tool definitions into structured prompt text.

    Accepts tools in Anthropic format (name, description, input_schema)
    and produces a text block suitable for appending to the system prompt.

    Args:
        tools: List of Anthropic-format tool definitions.

    Returns:
        A string containing tool descriptions and calling instructions.
        Returns empty string if no tools provided.
    """
    if not tools:
        return ""

    parts: list[str] = [_TOOL_CALL_INSTRUCTIONS]

    for tool in tools:
        name = tool.get("name", "unknown")
        description = tool.get("description", "No description.")
        schema = tool.get("input_schema", {})

        parts.append(f"### {name}\n")
        parts.append(f"{description}\n")
        parts.append(f"Parameters:\n```json\n{json.dumps(schema, indent=2)}\n```\n")

    logger.debug("Serialized %d tools to prompt text", len(tools))
    return "\n".join(parts)


def serialize_tool_results_to_text(
    tool_results: list[dict[str, Any]],
) -> str:
    """Serialize tool results into text for the conversation.

    Converts Anthropic tool_result content blocks into structured text
    that the model can read as part of the conversation.

    Args:
        tool_results: List of dicts with keys:
            - tool_use_id: The ID of the tool call this result is for
            - name: The tool name (if available)
            - content: The result content (string or nested blocks)

    Returns:
        Formatted text representation of tool results.
    """
    if not tool_results:
        return ""

    parts: list[str] = [_TOOL_RESULT_HEADER]

    for result in tool_results:
        tool_id = result.get("tool_use_id", "unknown")
        name = result.get("name", "")
        content = _extract_result_content(result.get("content", ""))

        header = f"<tool_result name=\"{name}\" id=\"{tool_id}\">" if name else f"<tool_result id=\"{tool_id}\">"
        parts.append(header)
        parts.append(content)
        parts.append("</tool_result>\n")

    return "\n".join(parts)


def _extract_result_content(raw: str | list[Any] | Any) -> str:
    """Extract text from tool result content."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        text_parts: list[str] = []
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    return str(raw)
