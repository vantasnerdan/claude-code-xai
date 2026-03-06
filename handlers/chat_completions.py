"""Handler for Chat Completions API requests.

Routes standard (non-multi-agent) models through /v1/chat/completions.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from bridge.logging_config import get_logger, dump_json, sanitize_request
from bridge.token_logger import log_token_usage
from translation.forward import anthropic_to_openai
from translation.streaming import OpenAIToAnthropicStreamAdapter
from translation.tools import get_last_enrichment_overhead

logger = get_logger("main")


async def handle_chat_completions(
    body: dict[str, Any],
    bridge_warnings: list[str],
    start: float,
    client: httpx.AsyncClient,
    api_key: str,
) -> JSONResponse | StreamingResponse:
    """Forward request via /v1/chat/completions (standard models)."""
    openai_body = anthropic_to_openai(body)

    logger.debug("Translated request: %s", json.dumps(sanitize_request(openai_body), default=str))
    dump_json("request", sanitize_request(openai_body))

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if openai_body.get("stream"):
        return await stream_chat(
            openai_body, headers, client, bridge_warnings, start,
            model=openai_body.get("model", ""),
        )

    resp = await client.post("/chat/completions", json=openai_body, headers=headers)
    data = resp.json()
    elapsed = time.time() - start

    # Guard against non-dict responses (e.g. plain text error body).
    if not isinstance(data, dict):
        logger.warning("Non-dict response from xAI: %s", str(data)[:500])
        return JSONResponse(status_code=resp.status_code, content={
            "type": "error", "error": {"type": "api_error", "message": str(data),
                                        "suggestion": "Unexpected response format from xAI."}})

    usage = data.get("usage", {})
    choices = data.get("choices", [])
    stop_reason = choices[0].get("finish_reason", "?") if choices else "?"
    tool_calls_count = len(
        (choices[0].get("message", {}).get("tool_calls") or []) if choices else []
    )
    logger.info(
        "xAI response in %.2fs status=%d stop=%s tool_calls=%d tokens=%d/%d/%d",
        elapsed, resp.status_code, stop_reason, tool_calls_count,
        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
        usage.get("total_tokens", 0),
    )

    log_token_usage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        enrichment_overhead_tokens=get_last_enrichment_overhead(),
        elapsed_seconds=elapsed, is_streaming=False, model=openai_body.get("model", ""),
    )

    logger.debug("xAI response body: %s", json.dumps(data, default=str))
    dump_json("response", data)

    from translation.reverse import translate_response
    result = translate_response(data, status_code=resp.status_code)

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content=result)

    response_headers = {}
    if bridge_warnings:
        response_headers["X-Bridge-Warning"] = "; ".join(bridge_warnings)
    return JSONResponse(content=result, headers=response_headers)


async def stream_chat(
    openai_body: dict[str, Any],
    headers: dict[str, str],
    client: httpx.AsyncClient,
    bridge_warnings: list[str] | None = None,
    start_time: float = 0,
    model: str = "",
) -> StreamingResponse:
    """Stream a Chat Completions response."""
    event_count = 0
    enrichment_overhead = get_last_enrichment_overhead()

    async def gen():
        nonlocal event_count
        async with client.stream("POST", "/chat/completions", json=openai_body, headers=headers) as resp:
            logger.info("Streaming started status=%d", resp.status_code)

            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.warning(
                    "Streaming error status=%d body=%s",
                    resp.status_code, error_body.decode("utf-8", errors="replace")[:1000],
                )
                error_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": f"xAI returned {resp.status_code}"},
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                return

            async def lines():
                async for line in resp.aiter_lines():
                    if line:
                        yield line
            adapter = OpenAIToAnthropicStreamAdapter(lines())
            async for event in adapter:
                event_count += 1
                yield f"event: {event.get('type', 'unknown')}\ndata: {json.dumps(event)}\n\n"
            elapsed = time.time() - start_time if start_time else 0
            logger.info("Streaming complete events=%d elapsed=%.2fs", event_count, elapsed)

            usage = adapter.usage
            log_token_usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                enrichment_overhead_tokens=enrichment_overhead,
                elapsed_seconds=elapsed, is_streaming=True, model=model,
            )

    response_headers = {}
    if bridge_warnings:
        response_headers["X-Bridge-Warning"] = "; ".join(bridge_warnings)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=response_headers)
