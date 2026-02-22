"""Forward translation: Anthropic Messages API -> OpenAI Chat Completions API.

Converts Claude Code requests into xAI/Grok-compatible format.
Handles system prompt extraction, content block flattening, tool_use
to tool_calls conversion, and tool_result to tool messages.
"""

from __future__ import annotations

import json
from typing import Any

from translation.config import TranslationConfig, UNSUPPORTED_FEATURES
from translation.tools import translate_tools as _translate_tools

_config = TranslationConfig()


def anthropic_to_openai(request: dict[str, Any]) -> dict[str, Any]:
    """Translate a full Anthropic Messages API request to OpenAI format.

    Args:
        request: Anthropic request body with model, max_tokens, messages, etc.

    Returns:
        OpenAI Chat Completions request body.

    Raises:
        NotImplementedError: For unsupported features (extended thinking, etc).
    """
    # Fail loudly on unsupported features
    for feature in UNSUPPORTED_FEATURES:
        if feature in request and request[feature]:
            raise NotImplementedError(
                f"Feature '{feature}' is not supported by the xAI bridge. "
                f"Disable it in your Claude Code configuration."
            )

    messages: list[dict[str, Any]] = []

    # System prompt: top-level field -> system role message
    system = request.get("system", "")
    preamble = _config.system_prompt_preamble
    if preamble and system:
        system = f"{preamble}\n\n{system}"
    elif preamble:
        system = preamble

    if system:
        messages.append({"role": "system", "content": system})

    # Translate conversation messages
    messages.extend(translate_messages(request.get("messages", [])))

    result: dict[str, Any] = {
        "model": _config.resolve_model(request.get("model", "")),
        "messages": messages,
        "max_tokens": request.get("max_tokens", _config.default_max_tokens),
        "temperature": request.get("temperature", _config.default_temperature),
    }

    # Stream flag
    if request.get("stream"):
        result["stream"] = True
    else:
        result["stream"] = False

    # Tools
    anthropic_tools = request.get("tools")
    if anthropic_tools:
        result["tools"] = _translate_tools(anthropic_tools)
    else:
        result["tools"] = None

    return result


def translate_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate a list of Anthropic messages to OpenAI format.

    Handles text blocks, tool_use blocks, and tool_result blocks.

    Args:
        messages: List of Anthropic-format messages.

    Returns:
        List of OpenAI-format messages.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        result.extend(_translate_single_message(msg))
    return result


def _translate_single_message(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate a single Anthropic message to one or more OpenAI messages."""
    role = msg.get("role", "user")
    content = msg.get("content")

    # String content passthrough
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    # Missing content
    if content is None:
        return [{"role": role, "content": ""}]

    # Empty content array
    if isinstance(content, list) and len(content) == 0:
        return [{"role": role, "content": ""}]

    # Process content blocks
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append(_translate_tool_use(block))
        elif block_type == "tool_result":
            tool_results.append(_translate_tool_result(block))
        elif block_type == "image":
            raise NotImplementedError(
                "Image content blocks are not supported by the xAI bridge. "
                "Vision features require native Anthropic API."
            )
        else:
            raise NotImplementedError(
                f"Unsupported content block type: '{block_type}'. "
                f"The xAI bridge only supports text, tool_use, and tool_result."
            )

    # Build result messages
    if tool_results:
        # tool_result blocks become separate tool messages
        return [
            {
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["content"],
            }
            for tr in tool_results
        ]

    if tool_calls:
        # Assistant message with tool_calls
        openai_msg: dict[str, Any] = {
            "role": role,
            "tool_calls": tool_calls,
        }
        if text_parts:
            openai_msg["content"] = "\n".join(text_parts)
        else:
            openai_msg["content"] = None
        return [openai_msg]

    # Pure text message
    return [{"role": role, "content": "\n".join(text_parts)}]


def _translate_tool_use(block: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool_use block to an OpenAI tool_call."""
    return {
        "id": block["id"],
        "type": "function",
        "function": {
            "name": block["name"],
            "arguments": json.dumps(block.get("input", {})),
        },
    }


def _translate_tool_result(block: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool_result block to OpenAI tool message data."""
    content_raw = block.get("content", "")
    if isinstance(content_raw, str):
        content_text = content_raw
    elif isinstance(content_raw, list):
        # Extract text from content blocks
        parts = []
        for sub_block in content_raw:
            if isinstance(sub_block, dict) and sub_block.get("type") == "text":
                parts.append(sub_block.get("text", ""))
            elif isinstance(sub_block, str):
                parts.append(sub_block)
        content_text = "\n".join(parts)
    else:
        content_text = str(content_raw)

    return {
        "tool_call_id": block.get("tool_use_id", ""),
        "content": content_text,
    }
