"""Claude Code xAI Bridge -- Anthropic Messages API to xAI Grok proxy.

Receives Claude Code traffic on /v1/messages, translates to OpenAI format,
forwards to xAI, translates response back. Enrichment hooks inject Agentic
API Standard context.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import os
import json
import time
from dotenv import load_dotenv

from bridge.logging_config import configure_logging, get_logger, dump_json, sanitize_request
from translation.forward import anthropic_to_openai, strip_thinking
from translation.reverse import translate_response
from translation.streaming import OpenAIToAnthropicStreamAdapter
from translation.tools import set_tool_enrichment_hook
from enrichment.factory import create_enricher

load_dotenv()
configure_logging()
logger = get_logger("main")

app = FastAPI(title="xai-agentic-claude-bridge")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
if not XAI_API_KEY:
    logger.warning("XAI_API_KEY not set. Requests to xAI will fail.")

client = httpx.AsyncClient(base_url="https://api.x.ai/v1", timeout=120.0)

enricher = create_enricher()
set_tool_enrichment_hook(enricher.enrich)
logger.info("Enrichment mode: %s", enricher.config.mode)


@app.get("/manifest")
async def get_manifest() -> dict:
    with open("manifest.json") as f:
        return json.load(f)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "model": os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning"),
        "enrichment_mode": enricher.config.mode,
    }


@app.post("/v1/messages", response_model=None)
async def messages(request: Request):
    body = await request.json()
    start = time.time()

    # -- Point 2: Outgoing request summary (INFO) --
    msg_count = len(body.get("messages", []))
    tool_count = len(body.get("tools", []) or [])
    has_thinking = "thinking" in body
    is_stream = bool(body.get("stream"))
    logger.info(
        "POST /v1/messages model=%s messages=%d tools=%d thinking=%s stream=%s",
        body.get("model", "?"), msg_count, tool_count, has_thinking, is_stream,
    )

    try:
        bridge_warnings = strip_thinking(body)
        if bridge_warnings:
            for w in bridge_warnings:
                logger.info("Degraded feature: %s", w)

        openai_body = anthropic_to_openai(body)

        # -- Point 2: Full translated payload at DEBUG --
        logger.debug(
            "Translated request: %s",
            json.dumps(sanitize_request(openai_body), default=str),
        )
        dump_json("request", sanitize_request(openai_body))

        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}

        if openai_body.get("stream"):
            return await _stream(openai_body, headers, bridge_warnings, start)

        resp = await client.post("/chat/completions", json=openai_body, headers=headers)
        data = resp.json()
        elapsed = time.time() - start

        # -- Point 3: Response summary (INFO) --
        usage = data.get("usage", {})
        choices = data.get("choices", [])
        stop_reason = choices[0].get("finish_reason", "?") if choices else "?"
        tool_calls_count = len(
            (choices[0].get("message", {}).get("tool_calls") or []) if choices else []
        )
        logger.info(
            "xAI response in %.2fs status=%d stop=%s tool_calls=%d "
            "tokens=%d/%d/%d",
            elapsed, resp.status_code, stop_reason, tool_calls_count,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )

        # -- Point 3: Full response at DEBUG --
        logger.debug("xAI response body: %s", json.dumps(data, default=str))
        dump_json("response", data)

        result = translate_response(data, status_code=resp.status_code)

        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content=result)

        response_headers = {}
        if bridge_warnings:
            response_headers["X-Bridge-Warning"] = "; ".join(bridge_warnings)
        return JSONResponse(content=result, headers=response_headers)

    except NotImplementedError as e:
        logger.warning("Unsupported feature: %s", e)
        return JSONResponse(status_code=400, content={
            "type": "error", "error": {"type": "invalid_request_error", "message": str(e),
                                        "suggestion": "Use native Anthropic API for this feature."}})
    except Exception as e:
        logger.exception("Bridge error: %s", e)
        return JSONResponse(status_code=500, content={
            "type": "error", "error": {"type": "api_error", "message": str(e),
                                        "suggestion": "Retry the request. Check XAI_API_KEY."},
            "_links": {"retry": {"href": "/v1/messages", "method": "POST"}, "manifest": {"href": "/manifest"}}})


async def _stream(
    openai_body: dict, headers: dict[str, str],
    bridge_warnings: list[str] | None = None, start_time: float = 0,
) -> StreamingResponse:
    event_count = 0

    async def gen():
        nonlocal event_count
        async with client.stream("POST", "/chat/completions", json=openai_body, headers=headers) as resp:
            logger.info("Streaming started status=%d", resp.status_code)

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

    response_headers = {}
    if bridge_warnings:
        response_headers["X-Bridge-Warning"] = "; ".join(bridge_warnings)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=response_headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "4000")))
