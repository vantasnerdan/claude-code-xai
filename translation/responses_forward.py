"""Forward translation: Anthropic Messages API -> xAI Responses API.

The Responses API uses a different field layout than Chat Completions:
- Messages go in 'input' (not 'messages')
- System prompt is a message with role 'system' in the input array
- Tools use 'name' at top level (not nested in 'function')
- Tool results use type 'function_call_output' with 'call_id'
- Supports 'previous_response_id' for stateful conversations
- Supports 'store' (boolean) for server-side persistence

Multi-agent models do NOT support client-side tool definitions. For these
models, tools are serialized into the system prompt and tool_use/tool_result
blocks in conversation history are converted to text format. The model
outputs <tool_call> blocks that the reverse translator parses.
"""

from __future__ import annotations

import json
from typing import Any

from bridge.logging_config import get_logger
from translation.config import TranslationConfig, UNSUPPORTED_FEATURES
from translation.model_routing import detect_endpoint, XAIEndpoint
from translation.tools import translate_tools as _translate_tools_chat
from translation.prompt_tools import serialize_tools_to_prompt, serialize_tool_results_to_text
from enrichment.system_preamble import strip_anthropic_identity

_config = TranslationConfig()
logger = get_logger("responses_forward")


def anthropic_to_responses(request: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages API request to xAI Responses format.

    For multi-agent models, tools are injected into the system prompt
    instead of the API tools array. Tool use/result blocks in conversation
    history are converted to text.

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

    resolved_model = _config.resolve_model(request.get("model", ""))
    use_prompt_tools = _is_multi_agent_model(resolved_model)
    anthropic_tools = request.get("tools", []) or []

    input_messages: list[dict[str, Any]] = []

    # System prompt -> first message with role 'system' in input array.
    raw_system = request.get("system", "")
    stripped = strip_anthropic_identity(raw_system)
    system = _flatten_system(stripped)
    preamble = _config.system_prompt_preamble
    if preamble and system:
        system = f"{preamble}\n\n{system}"
    elif preamble:
        system = preamble

    # For multi-agent models, append tool definitions to system prompt.
    if use_prompt_tools and anthropic_tools:
        tool_prompt = serialize_tools_to_prompt(anthropic_tools)
        system = f"{system}\n\n{tool_prompt}" if system else tool_prompt
        logger.info(
            "Prompt-based tools: injected %d tool definitions into system prompt",
            len(anthropic_tools),
        )

    if system:
        input_messages.append({"role": "system", "content": system})

    if use_prompt_tools:
        input_messages.extend(_translate_messages_prompt_tools(request.get("messages", [])))
    else:
        input_messages.extend(_translate_messages(request.get("messages", [])))

    result: dict[str, Any] = {
        "model": resolved_model,
        "input": input_messages,
        "store": False,
        "stream": bool(request.get("stream")),
    }

    # Only send tools array for non-multi-agent models.
    if not use_prompt_tools and anthropic_tools:
        result["tools"] = _translate_tools_responses(anthropic_tools)

    return result


def _is_multi_agent_model(resolved_model: str) -> bool:
    """Check if a model requires prompt-based tool calling.

    Multi-agent models do not accept client-side tool definitions via the
    API. We detect this from the model name to avoid the 400 error.
    """
    return detect_endpoint(resolved_model) == XAIEndpoint.RESPONSES


def _flatten_system(system: str | list[dict[str, Any]]) -> str:
    """Flatten a system prompt to a single string.

    Handles both Anthropic formats: plain string or list of content blocks.
    """
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    raise TypeError(
        f"Expected str or list for system field, got {type(system).__name__}"
    )


def _translate_tools_responses(
    anthropic_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate Anthropic tools to Responses API format.

    Responses API tools have a flatter structure:
    {type: "function", name: "...", description: "...", parameters: {...}}

    Unlike Chat Completions which nests under 'function' key.
    Enrichment hooks still run at the Anthropic level via _translate_tools_chat.
    """
    # Run enrichment hook at Anthropic level (via the chat tools module).
    # This calls the enrichment hook and measures overhead.
    chat_tools = _translate_tools_chat(anthropic_tools)

    # Convert from Chat Completions format to Responses API format.
    responses_tools: list[dict[str, Any]] = []
    for ct in chat_tools:
        func = ct.get("function", {})
        responses_tools.append({
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        })
    return responses_tools


def _translate_messages_prompt_tools(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate messages for prompt-based tool calling.

    Converts tool_use blocks to <tool_call> text format and tool_result
    blocks to <tool_result> text format. The model sees tools as text in
    the conversation, not as API-level function calls.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        result.extend(_translate_single_message_prompt_tools(msg))
    return result


def _translate_single_message_prompt_tools(
    msg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate a single message with tool blocks converted to text."""
    role = msg.get("role", "user")
    content = msg.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if content is None or (isinstance(content, list) and len(content) == 0):
        return [{"role": role, "content": ""}]

    text_parts: list[str] = []

    for block in content:
        bt = block.get("type", "")
        if bt == "text":
            text_parts.append(block.get("text", ""))
        elif bt == "tool_use":
            # Convert tool_use to <tool_call> text format.
            call_json = json.dumps({
                "name": block.get("name", ""),
                "parameters": block.get("input", {}),
            })
            text_parts.append(f"<tool_call>\n{call_json}\n</tool_call>")
        elif bt == "tool_result":
            # Convert tool_result to <tool_result> text format.
            raw = block.get("content", "")
            result_text = _extract_result_text(raw)
            name = block.get("name", "")
            tool_id = block.get("tool_use_id", "")
            header = f'<tool_result name="{name}" id="{tool_id}">' if name else f'<tool_result id="{tool_id}">'
            text_parts.append(f"{header}\n{result_text}\n</tool_result>")
        elif bt == "image":
            raise NotImplementedError(
                "Image content blocks are not supported by the xAI bridge."
            )
        # Skip thinking/redacted_thinking silently.
        elif bt not in ("thinking", "redacted_thinking"):
            logger.warning("Unknown content block type in prompt-tools mode: %s", bt)

    combined = "\n\n".join(text_parts) if text_parts else ""
    return [{"role": role, "content": combined}]


def _extract_result_text(raw: str | list[Any] | Any) -> str:
    """Extract text from a tool result content field."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(raw)


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
