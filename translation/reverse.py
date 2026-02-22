"""Reverse translation: OpenAI Chat Completions API -> Anthropic Messages API.

Converts xAI/Grok responses back into the format Claude Code expects.
Handles content block synthesis, tool_calls to tool_use conversion,
finish_reason mapping, and usage field translation.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from translation.config import TranslationConfig, STOP_REASON_MAP

_config = TranslationConfig()


def openai_to_anthropic(response: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response to Anthropic format.

    Args:
        response: OpenAI response with choices, usage, etc.

    Returns:
        Anthropic Messages API response.

    Raises:
        ValueError: If response has empty choices array.
    """
    choices = response.get("choices", [])
    if not choices:
        raise ValueError(
            "OpenAI response has no choices. Cannot translate empty response."
        )

    choice = choices[0]
    message = choice.get("message", {})

    # Build content blocks
    content = _build_content_blocks(message)

    # Map finish_reason to stop_reason
    finish_reason = choice.get("finish_reason")
    stop_reason = _config.map_stop_reason(finish_reason)

    # Map usage
    usage = _translate_usage(response.get("usage"))

    # Build response ID
    response_id = response.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    if not response_id.startswith("msg_"):
        response_id = f"msg_{response_id}"

    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": response.get("model", "grok-4-1-fast-reasoning"),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def translate_response(
    response: dict[str, Any],
    status_code: int = 200,
) -> dict[str, Any]:
    """Translate an OpenAI response or error to Anthropic format.

    Handles both success (2xx) and error (4xx/5xx) responses.
    Error responses include Agentic API Standard Pattern 3 (suggestions)
    and Pattern 2 (_links).

    Args:
        response: OpenAI response or error body.
        status_code: HTTP status code from xAI.

    Returns:
        Anthropic-format response or error.
    """
    if 200 <= status_code < 300:
        return openai_to_anthropic(response)

    return _translate_error(response, status_code)


def _build_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Anthropic content blocks from an OpenAI message."""
    content: list[dict[str, Any]] = []
    text = message.get("content")
    tool_calls = message.get("tool_calls")

    # Add text block if content is present and non-null
    if text is not None:
        content.append({"type": "text", "text": text})
    elif not tool_calls:
        # No content and no tool calls: empty text block
        content.append({"type": "text", "text": ""})

    # Convert tool_calls to tool_use blocks
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args = {}

            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": args,
            })

    return content


def _translate_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Map OpenAI usage fields to Anthropic usage fields."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}

    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


# Error type mapping: OpenAI error type -> Anthropic error type
_ERROR_TYPE_MAP: dict[str, str] = {
    "rate_limit_error": "rate_limit_error",
    "server_error": "api_error",
    "invalid_request_error": "invalid_request_error",
}

# Suggestions per error type (Agentic API Standard Pattern 3)
_ERROR_SUGGESTIONS: dict[int, str] = {
    429: "Rate limited by xAI. Wait and retry, or reduce request frequency.",
    500: "xAI internal error. Retry the request. If persistent, simplify tool schemas.",
    400: "Invalid request format. Check message structure and tool definitions.",
    401: "Authentication failed. Verify XAI_API_KEY is set correctly.",
    403: "Access denied. Check API key permissions for the requested model.",
}


def _translate_error(
    error_body: dict[str, Any],
    status_code: int,
) -> dict[str, Any]:
    """Translate an OpenAI error response to Anthropic error format."""
    error_data = error_body.get("error", {})
    openai_type = error_data.get("type", "api_error")
    message = error_data.get("message", "Unknown error from xAI API.")

    # Map to Anthropic error type
    if status_code == 429:
        anthropic_type = "rate_limit_error"
    elif status_code == 400:
        anthropic_type = "invalid_request_error"
    else:
        anthropic_type = _ERROR_TYPE_MAP.get(openai_type, "api_error")

    suggestion = _ERROR_SUGGESTIONS.get(status_code, "Retry the request.")

    return {
        "type": "error",
        "error": {
            "type": anthropic_type,
            "message": message,
            "suggestion": suggestion,
        },
        "_links": {
            "retry": {"href": "/v1/messages", "method": "POST"},
            "manifest": {"href": "/manifest"},
        },
    }
