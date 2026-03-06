"""Parse tool calls from model text output.

When multi-agent models receive tools via the system prompt (not the API),
they output tool calls as structured <tool_call> blocks in their text
response. This module extracts those blocks and converts them to Anthropic
tool_use content blocks.

Paired with translation/prompt_tools.py which handles serialization.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from bridge.logging_config import get_logger

logger = get_logger("tool_parser")

# Regex to match <tool_call>...</tool_call> blocks.
# Uses DOTALL so the JSON payload can span multiple lines.
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls_from_text(
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse <tool_call> blocks from model text output.

    Extracts tool calls and returns them as Anthropic tool_use content
    blocks. Text surrounding tool calls is returned as text content blocks.

    Args:
        text: The full text output from the model, possibly containing
            one or more <tool_call> blocks.

    Returns:
        A tuple of (content_blocks, tool_calls_only) where:
        - content_blocks: Ordered list of text and tool_use blocks
        - tool_calls_only: Just the tool_use blocks (for stop_reason)
    """
    if "<tool_call>" not in text:
        return [{"type": "text", "text": text}], []

    content: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    last_end = 0

    for match in _TOOL_CALL_PATTERN.finditer(text):
        # Add preceding text as a text block.
        preceding = text[last_end:match.start()].strip()
        if preceding:
            content.append({"type": "text", "text": preceding})

        # Parse the JSON payload.
        raw_json = match.group(1)
        tool_block = _parse_single_tool_call(raw_json)
        if tool_block is not None:
            content.append(tool_block)
            tool_calls.append(tool_block)

        last_end = match.end()

    # Add trailing text after the last tool call.
    trailing = text[last_end:].strip()
    if trailing:
        content.append({"type": "text", "text": trailing})

    if not content:
        content.append({"type": "text", "text": ""})

    logger.debug(
        "Parsed %d tool calls from text (%d total content blocks)",
        len(tool_calls), len(content),
    )
    return content, tool_calls


def _parse_single_tool_call(raw_json: str) -> dict[str, Any] | None:
    """Parse a single tool call JSON payload into a tool_use block.

    Args:
        raw_json: The JSON string inside a <tool_call> block.

    Returns:
        An Anthropic tool_use content block, or None if parsing fails.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("Malformed tool call JSON: %s (error: %s)", raw_json[:200], e)
        return None

    if not isinstance(data, dict):
        logger.warning("Tool call payload is not a dict: %s", type(data).__name__)
        return None

    name = data.get("name")
    if not name:
        logger.warning("Tool call missing 'name' field: %s", raw_json[:200])
        return None

    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        logger.warning("Tool call parameters is not a dict for %s", name)
        parameters = {}

    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": parameters,
    }


def has_pending_tool_call(buffer: str) -> bool:
    """Check if a text buffer has an unclosed <tool_call> tag.

    Used by streaming to determine if we should keep buffering.

    Args:
        buffer: Accumulated text from streaming chunks.

    Returns:
        True if there is an opening <tool_call> without a matching
        </tool_call>.
    """
    opens = buffer.count("<tool_call>")
    closes = buffer.count("</tool_call>")
    return opens > closes
