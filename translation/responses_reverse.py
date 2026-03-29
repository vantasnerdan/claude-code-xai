"""Reverse translation: xAI Responses API -> Anthropic Messages API.

As of issue #51, this is the PRIMARY reverse translation path for all models.
Converts Responses API output format back to Anthropic content blocks.
The output array contains items with 'type' field:
- 'message' with content[{type: 'output_text', text: '...'}]
- 'function_call' with {call_id, name, arguments}
- 'reasoning' with optional encrypted_content (stripped)
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from bridge.logging_config import get_logger
from translation.reverse import unescape_text, _unescape_args, _ERROR_SUGGESTIONS

logger = get_logger("responses_reverse")


def responses_to_anthropic(response: dict[str, Any]) -> dict[str, Any]:
    """Translate an xAI Responses API response to Anthropic format.

    Args:
        response: The raw xAI Responses API response body.

    Returns:
        An Anthropic Messages API response dict.
    """
    output = response.get("output", [])
    content = _build_content(output)
    rid = response.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    if not rid.startswith("msg_"):
        rid = f"msg_{rid}"

    stop_reason = _infer_stop_reason(output)

    usage = response.get("usage", {})
    prompt_details = usage.get("input_tokens_details", {})
    cached_tokens = prompt_details.get("cached_tokens", 0)

    anthropic_usage: dict[str, Any] = {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
    }
    if cached_tokens:
        anthropic_usage["cache_read_input_tokens"] = cached_tokens
        anthropic_usage["cache_creation_input_tokens"] = 0

    return {
        "id": rid,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": response.get("model", "grok-4.20-reasoning-latest"),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }


def translate_responses_response(
    response: dict[str, Any],
    status_code: int = 200,
) -> dict[str, Any]:
    """Translate a Responses API response or error to Anthropic format.

    Args:
        response: The raw response body from xAI.
        status_code: The HTTP status code.

    Returns:
        An Anthropic-format response dict.
    """
    if 200 <= status_code < 300:
        result = responses_to_anthropic(response)
        logger.debug(
            "Responses reverse: %d content blocks, stop=%s",
            len(result.get("content", [])),
            result.get("stop_reason"),
        )
        return result
    logger.debug("Translating Responses error status=%d", status_code)
    return _translate_error(response, status_code)


def _build_content(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Anthropic content blocks from Responses API output array."""
    content: list[dict[str, Any]] = []

    for item in output:
        item_type = item.get("type", "")

        if item_type == "message":
            # Message items contain a content array with output_text blocks.
            for sub in item.get("content", []):
                sub_type = sub.get("type", "")
                if sub_type == "output_text":
                    text = sub.get("text", "")
                    content.append({
                        "type": "text",
                        "text": unescape_text(text),
                    })

        elif item_type == "function_call":
            arguments = item.get("arguments", "{}")
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (json.JSONDecodeError, TypeError):
                args = {}
            args = _unescape_args(args)
            content.append({
                "type": "tool_use",
                "id": item.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": item.get("name", ""),
                "input": args,
            })

        elif item_type == "reasoning":
            # Reasoning blocks (encrypted or plain) are not passed to Claude Code.
            logger.debug("Skipping reasoning block in Responses output")
            continue

        else:
            logger.warning("Unknown Responses output type: %s", item_type)

    if not content:
        content.append({"type": "text", "text": ""})

    return content


def _infer_stop_reason(output: list[dict[str, Any]]) -> str:
    """Infer the Anthropic stop_reason from Responses API output.

    If the output contains function_call items, stop_reason is 'tool_use'.
    Otherwise, it's 'end_turn'.
    """
    for item in output:
        if item.get("type") == "function_call":
            return "tool_use"
    return "end_turn"


def _translate_error(
    error_body: dict[str, Any],
    status_code: int,
) -> dict[str, Any]:
    """Translate a Responses API error to Anthropic error format."""
    error_data = error_body.get("error", {})
    if isinstance(error_data, str):
        error_message = error_data
        error_type = "api_error"
    else:
        error_message = error_data.get("message", "Unknown error from xAI Responses API.")
        error_type = error_data.get("type", "api_error")

    if status_code == 429:
        atype = "rate_limit_error"
    elif status_code == 400:
        atype = "invalid_request_error"
    else:
        atype = error_type if error_type in ("rate_limit_error", "invalid_request_error") else "api_error"

    return {
        "type": "error",
        "error": {
            "type": atype,
            "message": error_message,
            "suggestion": _ERROR_SUGGESTIONS.get(status_code, "Retry the request."),
        },
        "_links": {
            "retry": {"href": "/v1/messages", "method": "POST"},
            "manifest": {"href": "/manifest"},
        },
    }
