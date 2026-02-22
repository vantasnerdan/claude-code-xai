"""Forward translation: Anthropic Messages API -> OpenAI Chat Completions API.

Handles system prompt extraction, content block flattening, tool_use
to tool_calls, and tool_result to tool role messages.
"""

from __future__ import annotations

import json
from typing import Any

from translation.config import TranslationConfig, UNSUPPORTED_FEATURES
from translation.tools import translate_tools as _translate_tools

# Re-export so tests can import from translation.forward
translate_tools = _translate_tools

_config = TranslationConfig()


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
    system = request.get("system", "")
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
