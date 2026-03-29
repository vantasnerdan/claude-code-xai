"""Handler for Responses API requests.

As of issue #51, this is the DEFAULT handler for ALL models.
The Responses API is the primary translation path.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from bridge.logging_config import get_logger, dump_json, sanitize_request
from bridge.token_logger import log_token_usage
from translation.responses_forward import anthropic_to_responses
from translation.responses_reverse import translate_responses_response
from translation.responses_streaming import ResponsesStreamAdapter
from translation.tools import get_last_enrichment_overhead

logger = get_logger("main")

# Stable conversation ID for xAI prompt caching server affinity.
# Generated once per bridge process — all requests route to the same xAI
# server, enabling automatic prefix caching (90% input token discount).
# See: https://docs.x.ai/developers/advanced-api-usage/prompt-caching
_CONV_ID = str(uuid.uuid4())


async def handle_responses(
    body: dict[str, Any],
    bridge_warnings: list[str],
    start: float,
    client: httpx.AsyncClient,
    api_key: str,
) -> JSONResponse | StreamingResponse:
    """Forward request via /v1/responses (multi-agent models)."""
    responses_body = anthropic_to_responses(body)

    logger.debug("Translated Responses request: %s", json.dumps(sanitize_request(responses_body), default=str))
    dump_json("request_responses", sanitize_request(responses_body))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-grok-conv-id": _CONV_ID,
    }

    if responses_body.get("stream"):
        return await stream_responses(
            responses_body, headers, client, bridge_warnings, start,
            model=responses_body.get("model", ""),
        )

    resp = await client.post("/responses", json=responses_body, headers=headers)
    data = resp.json()
    elapsed = time.time() - start

    # Guard against non-dict responses.
    if not isinstance(data, dict):
        logger.warning("Non-dict Responses response from xAI: %s", str(data)[:500])
        return JSONResponse(status_code=resp.status_code, content={
            "type": "error", "error": {"type": "api_error", "message": str(data),
                                        "suggestion": "Unexpected response format from xAI."}})

    usage = data.get("usage", {})
    prompt_details = usage.get("input_tokens_details", {})
    output = data.get("output", [])
    output_types = [item.get("type", "?") for item in output] if isinstance(output, list) else []
    logger.info(
        "xAI Responses in %.2fs status=%d outputs=%s",
        elapsed, resp.status_code, output_types,
    )

    log_token_usage(
        input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
        cached_tokens=prompt_details.get("cached_tokens", 0),
        enrichment_overhead_tokens=get_last_enrichment_overhead(),
        elapsed_seconds=elapsed, is_streaming=False, model=responses_body.get("model", ""),
    )

    logger.debug("xAI Responses body: %s", json.dumps(data, default=str))
    dump_json("response_responses", data)

    result = translate_responses_response(data, status_code=resp.status_code)

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content=result)

    response_headers = {}
    if bridge_warnings:
        response_headers["X-Bridge-Warning"] = "; ".join(bridge_warnings)
    return JSONResponse(content=result, headers=response_headers)


async def stream_responses(
    responses_body: dict[str, Any],
    headers: dict[str, str],
    client: httpx.AsyncClient,
    bridge_warnings: list[str] | None = None,
    start_time: float = 0,
    model: str = "",
) -> StreamingResponse:
    """Stream a Responses API response."""
    event_count = 0
    enrichment_overhead = get_last_enrichment_overhead()

    async def gen():
        nonlocal event_count
        async with client.stream("POST", "/responses", json=responses_body, headers=headers) as resp:
            logger.info("Responses streaming started status=%d", resp.status_code)

            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.warning(
                    "Responses streaming error status=%d body=%s",
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
            adapter = ResponsesStreamAdapter(lines())
            async for event in adapter:
                event_count += 1
                yield f"event: {event.get('type', 'unknown')}\ndata: {json.dumps(event)}\n\n"
            elapsed = time.time() - start_time if start_time else 0
            logger.info("Responses streaming complete events=%d elapsed=%.2fs", event_count, elapsed)

            usage = adapter.usage
            logger.info("xAI raw streaming usage: %s", json.dumps(usage, default=str))
            stream_prompt_details = usage.get("input_tokens_details", {})
            log_token_usage(
                input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
                cached_tokens=stream_prompt_details.get("cached_tokens", 0),
                enrichment_overhead_tokens=enrichment_overhead,
                elapsed_seconds=elapsed, is_streaming=True, model=model,
            )

    response_headers = {}
    if bridge_warnings:
        response_headers["X-Bridge-Warning"] = "; ".join(bridge_warnings)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=response_headers)
