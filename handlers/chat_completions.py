"""Handler for legacy Chat Completions API requests.

LEGACY FALLBACK: As of issue #52, this is a thin wrapper that converts
legacy Chat Completions routing into Responses API handling. The actual
translation and transport is delegated to handlers/responses.py.

Activated only when XAI_USE_CHAT_COMPLETIONS=true.

Previously this handler performed its own Anthropic-to-OpenAI translation,
posted to /v1/chat/completions, and translated responses back. Now it
delegates to the Responses API handler, preserving the same function
signature so callers (main.py) do not need to change.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from bridge.logging_config import get_logger
from handlers.responses import handle_responses

logger = get_logger("main")


async def handle_chat_completions(
    body: dict[str, Any],
    bridge_warnings: list[str],
    start: float,
    client: httpx.AsyncClient,
    api_key: str,
) -> JSONResponse | StreamingResponse:
    """Forward request via the Responses API (delegated).

    This is the legacy entry point. It preserves the original function
    signature for backward compatibility but delegates all work to
    :func:`handlers.responses.handle_responses`.

    A deprecation warning header is injected so callers know they are
    on the legacy path.
    """
    logger.info("Legacy Chat Completions path — delegating to Responses API handler")
    bridge_warnings = [*bridge_warnings, "Legacy Chat Completions path: delegating to Responses API"]
    return await handle_responses(body, bridge_warnings, start, client, api_key)


async def stream_chat(
    openai_body: dict[str, Any],
    headers: dict[str, str],
    client: httpx.AsyncClient,
    bridge_warnings: list[str] | None = None,
    start_time: float = 0,
    model: str = "",
) -> StreamingResponse:
    """Legacy streaming entry point.

    Retained for backward compatibility with any callers that imported
    ``stream_chat`` directly. Delegates to
    :func:`handlers.responses.stream_responses`.

    .. deprecated:: issue #52
        Use :func:`handlers.responses.stream_responses` directly.
    """
    from handlers.responses import stream_responses

    logger.warning("stream_chat() called directly — this is a deprecated code path")
    return await stream_responses(
        openai_body, headers, client, bridge_warnings, start_time, model,
    )
