"""Claude Code xAI Bridge -- Anthropic Messages API to xAI Grok proxy.

Receives Claude Code traffic on /v1/messages, translates to either
OpenAI Chat Completions or xAI Responses API format depending on the
target model, forwards to xAI, translates response back. Enrichment
hooks inject Agentic API Standard context.

Multi-agent models (e.g. grok-4.20-multi-agent) route to /v1/responses.
All other models route to /v1/chat/completions.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import os
import json
import time
from dotenv import load_dotenv

from bridge.logging_config import configure_logging, get_logger
from translation.forward import strip_thinking
from translation.config import TranslationConfig
from translation.model_routing import detect_endpoint, XAIEndpoint
from translation.tools import set_tool_enrichment_hook, reset_enrichment_overhead
from enrichment.factory import create_enricher
from handlers.chat_completions import handle_chat_completions
from handlers.responses import handle_responses

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

_config = TranslationConfig()


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
    reset_enrichment_overhead()

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
                logger.debug("Degraded feature: %s", w)

        resolved_model = _config.resolve_model(body.get("model", ""))
        endpoint = detect_endpoint(resolved_model)

        if endpoint == XAIEndpoint.RESPONSES:
            return await handle_responses(body, bridge_warnings, start, client, XAI_API_KEY)

        return await handle_chat_completions(body, bridge_warnings, start, client, XAI_API_KEY)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "4000")))
