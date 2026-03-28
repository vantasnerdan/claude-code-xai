"""Reverse translation: OpenAI Chat Completions -> Anthropic Messages API.

LEGACY PATH: As of issue #51, the Responses API (responses_reverse.py) is
the default. This module is retained for the Chat Completions fallback,
activated via XAI_USE_CHAT_COMPLETIONS=true.

Converts xAI/Grok responses back to Claude Code format: content blocks,
tool_use blocks, stop_reason mapping, usage translation, error formatting.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from bridge.logging_config import get_logger
from translation.config import TranslationConfig, STOP_REASON_MAP

_config = TranslationConfig()
logger = get_logger("reverse")

# Pattern to match literal escape sequences that should be real control chars.
# Matches \n, \t, \r that are NOT preceded by an actual backslash (negative
# lookbehind ensures we don't corrupt already-escaped sequences like \\n).
_ESCAPE_MAP: dict[str, str] = {
    r"\n": "\n",
    r"\t": "\t",
    r"\r": "\r",
}
_LITERAL_ESCAPE_RE = re.compile(r"(?<!\\)\\([ntr])")


def unescape_text(text: str) -> str:
    """Unescape literal backslash-n/t/r sequences in model output text.

    xAI/Grok may return text with literal two-character escape sequences
    (backslash + n) instead of actual newline characters. This converts
    them to real control characters so Claude Code renders them correctly.

    Only applied to display text, never to tool call arguments.
    """
    if "\\" not in text:
        return text
    return _LITERAL_ESCAPE_RE.sub(
        lambda m: _ESCAPE_MAP[m.group(0)], text
    )

_ERROR_TYPE_MAP: dict[str, str] = {
    "rate_limit_error": "rate_limit_error",
    "server_error": "api_error",
    "invalid_request_error": "invalid_request_error",
}

_ERROR_SUGGESTIONS: dict[int, str] = {
    429: "Rate limited by xAI. Wait and retry, or reduce request frequency.",
    500: "xAI internal error. Retry the request. If persistent, simplify tool schemas.",
    400: "Invalid request format. Check message structure and tool definitions.",
    401: "Authentication failed. Verify XAI_API_KEY is set correctly.",
    403: "Access denied. Check API key permissions for the requested model.",
}


def openai_to_anthropic(response: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response to Anthropic format."""
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("OpenAI response has no choices. Cannot translate empty response.")

    choice = choices[0]
    message = choice.get("message", {})
    content = _build_content(message)
    rid = response.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    if not rid.startswith("msg_"):
        rid = f"msg_{rid}"

    usage = response.get("usage")
    return {
        "id": rid, "type": "message", "role": "assistant", "content": content,
        "model": response.get("model", "grok-4-1-fast-reasoning"),
        "stop_reason": _config.map_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": (usage or {}).get("prompt_tokens", 0),
                  "output_tokens": (usage or {}).get("completion_tokens", 0)},
    }


def translate_response(response: dict[str, Any], status_code: int = 200) -> dict[str, Any]:
    """Translate an OpenAI response or error to Anthropic format."""
    if 200 <= status_code < 300:
        result = openai_to_anthropic(response)
        logger.debug(
            "Reverse translation: %d content blocks, stop=%s",
            len(result.get("content", [])), result.get("stop_reason"),
        )
        return result
    logger.debug("Translating error response status=%d", status_code)
    return _translate_error(response, status_code)


def _build_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Anthropic content blocks from an OpenAI message."""
    content: list[dict[str, Any]] = []
    text = message.get("content")
    tool_calls = message.get("tool_calls")

    if text is not None:
        content.append({"type": "text", "text": unescape_text(text)})
    elif not tool_calls:
        content.append({"type": "text", "text": ""})

    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": args,
            })
    return content


def _translate_error(error_body: dict[str, Any], status_code: int) -> dict[str, Any]:
    """Translate an OpenAI error to Anthropic error format with Agentic Standard compliance."""
    error_data = error_body.get("error", {})
    openai_type = error_data.get("type", "api_error")

    if status_code == 429:
        atype = "rate_limit_error"
    elif status_code == 400:
        atype = "invalid_request_error"
    else:
        atype = _ERROR_TYPE_MAP.get(openai_type, "api_error")

    return {
        "type": "error",
        "error": {
            "type": atype,
            "message": error_data.get("message", "Unknown error from xAI API."),
            "suggestion": _ERROR_SUGGESTIONS.get(status_code, "Retry the request."),
        },
        "_links": {"retry": {"href": "/v1/messages", "method": "POST"}, "manifest": {"href": "/manifest"}},
    }
