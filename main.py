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
import logging
from dotenv import load_dotenv

from translation.forward import anthropic_to_openai
from translation.reverse import translate_response
from translation.streaming import OpenAIToAnthropicStreamAdapter
from translation.tools import set_tool_enrichment_hook
from enrichment.factory import create_enricher

load_dotenv()
logger = logging.getLogger(__name__)

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
    try:
        openai_body = anthropic_to_openai(body)
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}

        if openai_body.get("stream"):
            return await _stream(openai_body, headers)

        resp = await client.post("/chat/completions", json=openai_body, headers=headers)
        data = resp.json()
        logger.info("xAI response in %.2fs (status=%d)", time.time() - start, resp.status_code)
        result = translate_response(data, status_code=resp.status_code)
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content=result)
        return result

    except NotImplementedError as e:
        return JSONResponse(status_code=400, content={
            "type": "error", "error": {"type": "invalid_request_error", "message": str(e),
                                        "suggestion": "Use native Anthropic API for this feature."}})
    except Exception as e:
        logger.exception("Bridge error: %s", e)
        return JSONResponse(status_code=500, content={
            "type": "error", "error": {"type": "api_error", "message": str(e),
                                        "suggestion": "Retry the request. Check XAI_API_KEY."},
            "_links": {"retry": {"href": "/v1/messages", "method": "POST"}, "manifest": {"href": "/manifest"}}})


async def _stream(openai_body: dict, headers: dict[str, str]) -> StreamingResponse:
    async def gen():
        async with client.stream("POST", "/chat/completions", json=openai_body, headers=headers) as resp:
            async def lines():
                async for line in resp.aiter_lines():
                    if line:
                        yield line
            adapter = OpenAIToAnthropicStreamAdapter(lines())
            async for event in adapter:
                yield f"event: {event.get('type', 'unknown')}\ndata: {json.dumps(event)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "4000")))
