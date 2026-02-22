from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import os
import json
import time
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from models import AgenticError
from agentic_enricher import enrich_tools_for_grok

load_dotenv()

app = FastAPI(title="xai-agentic-claude-bridge")
client = httpx.AsyncClient(base_url="https://api.x.ai/v1", timeout=120.0)

GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning")

# Metrics (prove your standard improves Grok)
tool_success = Counter('tool_calls_success_total', 'Successful tool calls', ['tool_name'])
tool_latency = Histogram('tool_call_latency_seconds', 'Latency', ['tool_name'])

@app.get("/manifest")
async def get_manifest():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return manifest  # full Gold tier

@app.get("/health")
@app.get("/metrics")
async def metrics():
    return JSONResponse(content={"status": "healthy", "model": GROK_MODEL}, media_type=CONTENT_TYPE_LATEST if "/metrics" in str else None)  # real prometheus in prod

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    start = time.time()

    # === YOUR AGENTIC STANDARD ENRICHMENT (the improvement layer) ===
    if "tools" in body and body["tools"]:
        body["tools"] = enrich_tools_for_grok(body["tools"])

    try:
        # Anthropic → OpenAI format for xAI
        openai_body = {
            "model": GROK_MODEL,
            "messages": body["messages"],
            "tools": [{"type": "function", "function": t} for t in body.get("tools", [])] or None,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 8192),
            "stream": body.get("stream", False)
        }

        resp = await client.post("/chat/completions", json=openai_body)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        if openai_body.get("stream"):
            async def stream():
                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\n"
            return StreamingResponse(stream(), media_type="text/event-stream")

        # Non-stream → add HATEOAS + standardized wrapper (Pattern 2 + 3)
        data = resp.json()
        return {
            "content": data["choices"][0]["message"]["content"] or "",
            "usage": data["usage"],
            "_links": {"self": {"href": "/v1/messages"}, "retry": {"href": "/v1/messages", "method": "POST"}},
            "warnings": []  # Pattern 8
        }

    except Exception as e:
        err = AgenticError(
            error="grok_bridge_error",
            code=type(e).__name__,
            message=str(e),
            suggestion="Simplify tool parameters or retry with clearer schema. Check XAI_API_KEY.",
            retry_after=5,
            _links={"retry": {"href": "/v1/messages", "method": "POST"}, "manifest": {"href": "/manifest"}}
        )
        return JSONResponse(status_code=500, content=err.model_dump())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 4000)))
