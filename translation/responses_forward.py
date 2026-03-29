"""Forward translation: Anthropic Messages API -> xAI Responses API.

As of issue #51, this is the PRIMARY translation path for all models.
The Responses API uses a different field layout than Chat Completions:
- Messages go in 'input' (not 'messages')
- System prompt is a message with role 'system' in the input array
- Tools use 'name' at top level (not nested in 'function')
- Tool results use type 'function_call_output' with 'call_id'
- Supports 'previous_response_id' for stateful conversations
- Supports 'store' (boolean) for server-side persistence
"""

from __future__ import annotations

import json
from typing import Any

from bridge.logging_config import get_logger
from translation.config import TranslationConfig, UNSUPPORTED_FEATURES
from translation.shared import flatten_system
from translation.tools import translate_tools_responses as _translate_tools_responses
from enrichment.system_preamble import strip_anthropic_identity

_config = TranslationConfig()
logger = get_logger("responses_forward")


def anthropic_to_responses(request: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages API request to xAI Responses format.

    Args:
        request: The incoming Anthropic-format request body.

    Returns:
        An xAI Responses API request body.

    Raises:
        NotImplementedError: If the request uses unsupported features.
    """
    for feature in UNSUPPORTED_FEATURES:
        if feature in request and request[feature]:
            raise NotImplementedError(
                f"Feature '{feature}' is not supported by the xAI bridge. "
                f"Disable it in your Claude Code configuration."
            )

    input_messages: list[dict[str, Any]] = []

    # System prompt -> first message with role 'system' in input array.
    raw_system = request.get("system", "")
    stripped = strip_anthropic_identity(raw_system)
    system = flatten_system(stripped)
    preamble = _config.system_prompt_preamble
    if preamble and system:
        system = f"{preamble}\n\n{system}"
    elif preamble:
        system = preamble
    if system:
        input_messages.append({"role": "system", "content": system})

    input_messages.extend(_translate_messages(request.get("messages", [])))

    resolved_model = _config.resolve_model(request.get("model", ""))

    result: dict[str, Any] = {
        "model": resolved_model,
        "input": input_messages,
        "max_output_tokens": request.get("max_tokens", _config.default_max_tokens),
        "temperature": request.get("temperature", _config.default_temperature),
        "store": False,
        "stream": bool(request.get("stream")),
    }

    # Reasoning effort only for grok-4.20-multi-agent (only model that supports it).
    if "multi-agent" in resolved_model:
        result["reasoning"] = {"effort": "high"}

    # Translate tools to Responses API format (enrichment runs inside).
    tools = request.get("tools")
    if tools:
        result["tools"] = _translate_tools_responses(tools)

    return result


def _translate_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Anthropic messages to Responses API input format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        result.extend(_translate_single_message(msg))
    return result


def _translate_single_message(
    msg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate a single Anthropic message to Responses API input items."""
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
                "type": "function_call",
                "call_id": block["id"],
                "name": block["name"],
                "arguments": json.dumps(block.get("input", {})),
            })
        elif bt == "tool_result":
            tool_results.append(_extract_tool_result(block))
        elif bt == "image":
            raise NotImplementedError(
                "Image content blocks are not supported by the xAI bridge."
            )
        else:
            raise NotImplementedError(
                f"Unsupported content block type: '{bt}'."
            )

    # Function call outputs are sent as top-level input items.
    if tool_results:
        return tool_results

    # Assistant messages with tool calls: include text + function_call items.
    result: list[dict[str, Any]] = []
    if text_parts:
        combined = "\n".join(text_parts)
        if combined:
            result.append({"role": role, "content": combined})
    result.extend(tool_calls)
    if not result:
        result.append({"role": role, "content": ""})
    return result


def _extract_tool_result(block: dict[str, Any]) -> dict[str, Any]:
    """Extract tool result into Responses API function_call_output format."""
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
    return {
        "type": "function_call_output",
        "call_id": block.get("tool_use_id", ""),
        "output": text,
    }
