"""Forward translation: Anthropic Messages API -> OpenAI Chat Completions API.

LEGACY PATH: As of issue #51, the Responses API (responses_forward.py) is
the default. This module is retained for the Chat Completions fallback,
activated via XAI_USE_CHAT_COMPLETIONS=true.

Handles system prompt extraction, content block flattening, tool_use
to tool_calls, and tool_result to tool role messages.
"""

from __future__ import annotations

import json
from typing import Any

from bridge.logging_config import get_logger
from translation.config import TranslationConfig, UNSUPPORTED_FEATURES, STRIPPED_FEATURES
from translation.shared import flatten_system as _flatten_system
from translation.tools import translate_tools as _translate_tools
from enrichment.system_preamble import strip_anthropic_identity

# Re-export so tests can import from translation.forward
translate_tools = _translate_tools

_config = TranslationConfig()
logger = get_logger("forward")

# Content block types that belong to Anthropic's thinking feature.
# These are stripped from messages when forwarding to xAI.
_THINKING_BLOCK_TYPES: frozenset[str] = frozenset({
    "thinking",
    "redacted_thinking",
})


def strip_thinking(request: dict[str, Any]) -> list[str]:
    """Strip thinking-related fields from an Anthropic request.

    Removes the top-level 'thinking' parameter and any thinking/redacted_thinking
    content blocks from messages. Grok models reason internally and do not need
    (or support) explicit thinking parameters.

    Returns a list of warning strings for stripped features.
    """
    warnings: list[str] = []

    for feature in STRIPPED_FEATURES:
        if feature in request and request[feature]:
            warnings.append(
                f"'{feature}' parameter stripped. "
                f"Grok models reason internally; response quality is not affected."
            )
            del request[feature]

    # Strip thinking content blocks from conversation history
    for msg in request.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            filtered = [
                block for block in content
                if block.get("type") not in _THINKING_BLOCK_TYPES
            ]
            if len(filtered) < len(content):
                msg["content"] = filtered
                if not any("thinking" in w for w in warnings):
                    warnings.append(
                        "Thinking content blocks stripped from conversation history."
                    )

    return warnings


def anthropic_to_openai(request: dict[str, Any]) -> dict[str, Any]:
    """Translate a full Anthropic Messages API request to OpenAI format."""
    for feature in UNSUPPORTED_FEATURES:
        if feature in request and request[feature]:
            raise NotImplementedError(
                f"Feature '{feature}' is not supported by the xAI bridge. "
                f"Disable it in your Claude Code configuration."
            )

    messages: list[dict[str, Any]] = []

    # System prompt: top-level field -> system role message
    # The Anthropic API sends system as either a string or a list of
    # content blocks (e.g. [{"type": "text", "text": "..."}]).
    # Strip Anthropic identity claims, then flatten to a string for OpenAI.
    raw_system = request.get("system", "")
    stripped = strip_anthropic_identity(raw_system)
    system = _flatten_system(stripped)
    preamble = _config.system_prompt_preamble
    if preamble and system:
        system = f"{preamble}\n\n{system}"
    elif preamble:
        system = preamble
    if system:
        messages.append({"role": "system", "content": system})

    messages.extend(translate_messages(request.get("messages", [])))

    result: dict[str, Any] = {
        "model": _config.resolve_model(request.get("model", "")),
        "messages": messages,
        "max_tokens": request.get("max_tokens", _config.default_max_tokens),
        "temperature": request.get("temperature", _config.default_temperature),
        "stream": bool(request.get("stream")),
    }

    tools = request.get("tools")
    result["tools"] = _translate_tools(tools) if tools else None
    return result


def translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic messages to OpenAI format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        result.extend(_translate_single_message(msg))
    return result


def _translate_single_message(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate a single Anthropic message to one or more OpenAI messages."""
    role = msg.get("role", "user")
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if content is None or (isinstance(content, list) and len(content) == 0):
        return [{"role": role, "content": ""}]

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in content:
        bt = block.get("type", "")
        if bt == "text":
            text_parts.append(block.get("text", ""))
        elif bt == "tool_use":
            tool_calls.append({
                "id": block["id"], "type": "function",
                "function": {"name": block["name"], "arguments": json.dumps(block.get("input", {}))},
            })
        elif bt == "tool_result":
            tool_results.append(_extract_tool_result(block))
        elif bt == "image":
            raise NotImplementedError(
                "Image content blocks are not supported by the xAI bridge. "
                "Vision features require native Anthropic API."
            )
        else:
            raise NotImplementedError(
                f"Unsupported content block type: '{bt}'. "
                f"The xAI bridge only supports text, tool_use, and tool_result."
            )

    if tool_results:
        return [{"role": "tool", "tool_call_id": tr["tool_call_id"], "content": tr["content"]} for tr in tool_results]

    if tool_calls:
        m: dict[str, Any] = {"role": role, "tool_calls": tool_calls}
        m["content"] = "\n".join(text_parts) if text_parts else None
        return [m]

    return [{"role": role, "content": "\n".join(text_parts)}]


def _extract_tool_result(block: dict[str, Any]) -> dict[str, Any]:
    """Extract tool result content into flat text."""
    raw = block.get("content", "")
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts = []
        for sub in raw:
            if isinstance(sub, dict) and sub.get("type") == "text":
                parts.append(sub.get("text", ""))
            elif isinstance(sub, str):
                parts.append(sub)
        text = "\n".join(parts)
    else:
        text = str(raw)
    return {"tool_call_id": block.get("tool_use_id", ""), "content": text}
