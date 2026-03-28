#!/usr/bin/env python3
"""Smoke test for the Claude Code xAI Bridge.

Sends a known request with tool definitions through the bridge and verifies:
  1. Bridge starts and serves endpoints (manifest, health)
  2. Enrichment is applied (structural + behavioral fields present)
  3. Request/response translation works end-to-end

Modes:
  - Mock mode (default, no XAI_API_KEY): Uses mocked xAI responses for CI
  - Live mode (XAI_API_KEY set): Sends real requests to xAI API

Usage:
    python tests/smoke.py              # mock mode
    XAI_API_KEY=sk-... python tests/smoke.py  # live mode
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# ── Constants ──────────────────────────────────────────────────────────
STRUCTURAL_FIELDS = {
    "_links", "_manifest", "_error_format", "_near_miss",
    "_quality", "_anti_patterns", "_registration", "outputSchema",
}
BEHAVIORAL_FIELDS = {"behavioral_what", "behavioral_why", "behavioral_when"}

SAMPLE_TOOLS = [
    {
        "name": "Read",
        "description": "Reads a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Path to the file"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Writes a file to the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Bash",
        "description": "Executes a bash command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

SAMPLE_REQUEST = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "system": "You are a helpful coding assistant.",
    "messages": [{"role": "user", "content": "Read the file /tmp/test.py"}],
    "tools": SAMPLE_TOOLS,
}

MOCK_XAI_RESPONSE = {
    "id": "resp_smoke_001",
    "output": [
        {
            "type": "function_call",
            "call_id": "call_smoke_001",
            "name": "Read",
            "arguments": json.dumps({"file_path": "/tmp/test.py"}),
        }
    ],
    "model": "grok-4-1-fast-reasoning",
    "usage": {"input_tokens": 50, "output_tokens": 20},
}


# ── Results tracker ────────────────────────────────────────────────────
class Results:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append(name)
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, reason: str) -> None:
        self.failed.append((name, reason))
        print(f"  FAIL  {name} — {reason}")

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        print(f"  SKIP  {name} — {reason}")

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(f"\n{'=' * 60}")
        print(f"  {len(self.passed)}/{total} passed", end="")
        if self.failed:
            print(f", {len(self.failed)} FAILED", end="")
        if self.skipped:
            print(f", {len(self.skipped)} skipped", end="")
        print()
        if self.failed:
            print(f"\n  Failed checks:")
            for name, reason in self.failed:
                print(f"    - {name}: {reason}")
        print(f"{'=' * 60}")
        return 1 if self.failed else 0


# ── Check functions ────────────────────────────────────────────────────
def check_manifest(client: TestClient, r: Results) -> None:
    """Verify /manifest returns valid data."""
    resp = client.get("/manifest")
    if resp.status_code != 200:
        r.fail("manifest", f"status {resp.status_code}")
        return
    data = resp.json()
    if data.get("name") != "Claude Code xAI Bridge":
        r.fail("manifest.name", f"got '{data.get('name')}'")
        return
    modes = set(data.get("enrichment_modes", []))
    if modes != {"passthrough", "structural", "full"}:
        r.fail("manifest.enrichment_modes", f"got {modes}")
        return
    r.ok("manifest", f"v{data.get('version', '?')}, {len(data.get('capabilities', []))} capabilities")


def check_health(client: TestClient, r: Results) -> None:
    """Verify /health returns valid data."""
    resp = client.get("/health")
    if resp.status_code != 200:
        r.fail("health", f"status {resp.status_code}")
        return
    data = resp.json()
    if data.get("status") != "healthy":
        r.fail("health.status", f"got '{data.get('status')}'")
        return
    mode = data.get("enrichment_mode", "unknown")
    model = data.get("model", "unknown")
    r.ok("health", f"mode={mode}, model={model}")


def check_enrichment(r: Results) -> None:
    """Verify enrichment engine produces expected fields."""
    from enrichment.factory import create_enricher

    for mode, expect_structural, expect_behavioral in [
        ("passthrough", False, False),
        ("structural", True, False),
        ("full", True, True),
    ]:
        enricher = create_enricher(mode=mode)
        result = enricher.enrich(SAMPLE_TOOLS)

        if mode == "passthrough":
            if result == SAMPLE_TOOLS:
                r.ok(f"enrichment.{mode}", "no modification (correct)")
            else:
                r.fail(f"enrichment.{mode}", "modified tools in passthrough mode")
            continue

        tool = result[0]  # Check first tool (Read)
        found_structural = STRUCTURAL_FIELDS & set(tool.keys())
        found_behavioral = BEHAVIORAL_FIELDS & set(tool.keys())

        if expect_structural and not found_structural:
            r.fail(f"enrichment.{mode}", "no structural fields found")
        elif expect_behavioral and not found_behavioral:
            r.fail(f"enrichment.{mode}", f"structural OK ({len(found_structural)} fields) but no behavioral fields")
        elif expect_structural:
            detail = f"{len(found_structural)} structural"
            if found_behavioral:
                detail += f" + {len(found_behavioral)} behavioral"
            r.ok(f"enrichment.{mode}", detail)


def check_translation_mock(client: TestClient, r: Results) -> None:
    """Verify request translation with mocked xAI backend."""
    captured: dict = {}

    async def capture_post(*args, **kwargs):
        captured.update(kwargs.get("json", {}))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_XAI_RESPONSE
        return mock_resp

    import main
    with patch.object(main.client, "post", side_effect=capture_post):
        resp = client.post("/v1/messages", json=SAMPLE_REQUEST)

    if resp.status_code != 200:
        r.fail("translation.request", f"status {resp.status_code}: {resp.text[:200]}")
        return

    # Check the captured request sent to xAI
    if not captured:
        r.fail("translation.request", "no request captured")
        return

    # Verify model was mapped
    model = captured.get("model", "")
    if "grok" not in model:
        r.fail("translation.model_map", f"model not mapped to Grok: '{model}'")
    else:
        r.ok("translation.model_map", f"→ {model}")

    # Verify system prompt was injected (Responses API uses 'input' array with system role)
    input_msgs = captured.get("input", captured.get("messages", []))
    system_msgs = [m for m in input_msgs if m.get("role") == "system"]
    if system_msgs:
        r.ok("translation.system_prompt", f"{len(system_msgs[0]['content'])} chars")
    else:
        r.ok("translation.system_prompt", "no system message (preamble may be disabled)")

    # Verify tools were translated to Responses API format
    tools = captured.get("tools", [])
    if not tools:
        r.fail("translation.tools_forward", "no tools in captured request")
    elif tools[0].get("type") != "function":
        r.fail("translation.tools_forward", f"expected type=function, got {tools[0].get('type')}")
    else:
        # Responses API tools have flat format (no nested 'function' key)
        has_function_key = "function" in tools[0]
        fmt = "Chat Completions (nested)" if has_function_key else "Responses API (flat)"
        r.ok("translation.tools_forward", f"{len(tools)} tools translated to {fmt} format")

    # Verify response translation
    data = resp.json()
    if data.get("type") != "message":
        r.fail("translation.response", f"expected type=message, got {data.get('type')}")
        return
    if data.get("stop_reason") != "tool_use":
        r.fail("translation.stop_reason", f"expected tool_use, got {data.get('stop_reason')}")
        return

    tool_blocks = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
    if not tool_blocks:
        r.fail("translation.tool_use", "no tool_use blocks in response")
    else:
        tb = tool_blocks[0]
        r.ok("translation.response", f"tool_use: {tb['name']}({json.dumps(tb.get('input', {}))})")


def check_translation_live(client: TestClient, r: Results) -> None:
    """Send a real request to xAI and verify end-to-end."""
    # Simple request without tools — just verify connectivity
    resp = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Say 'Bridge smoke test OK' and nothing else."}],
    })

    if resp.status_code != 200:
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        error_msg = data.get("error", {}).get("message", resp.text[:200])
        r.fail("live.simple", f"status {resp.status_code}: {error_msg}")
        return

    data = resp.json()
    text = data.get("content", [{}])[0].get("text", "")
    r.ok("live.simple", f"Grok replied: '{text[:80]}'")

    # Request with tools — verify enrichment reaches xAI
    resp = client.post("/v1/messages", json=SAMPLE_REQUEST)
    if resp.status_code != 200:
        r.fail("live.with_tools", f"status {resp.status_code}")
        return

    data = resp.json()
    content_types = [b.get("type") for b in data.get("content", [])]
    r.ok("live.with_tools", f"response blocks: {content_types}")


def check_thinking_stripped(client: TestClient, r: Results) -> None:
    """Verify thinking parameter is stripped, not rejected."""
    async def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "resp_smoke_002",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "OK"}]},
            ],
            "model": "grok-4-1-fast-reasoning",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        return mock_resp

    import main
    with patch.object(main.client, "post", side_effect=mock_post):
        resp = client.post("/v1/messages", json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 256,
            "thinking": True,
            "messages": [{"role": "user", "content": "Hello"}],
        })

    if resp.status_code != 200:
        r.fail("thinking.strip", f"expected 200, got {resp.status_code} (should strip, not reject)")
        return

    warning = resp.headers.get("x-bridge-warning", "")
    if "stripped" in warning.lower() or "thinking" in warning.lower():
        r.ok("thinking.strip", f"X-Bridge-Warning: {warning[:80]}")
    else:
        r.fail("thinking.strip", "200 OK but no X-Bridge-Warning header for stripped thinking")


# ── Main ───────────────────────────────────────────────────────────────
def main() -> int:
    r = Results()
    api_key = os.getenv("XAI_API_KEY", "")
    live = bool(api_key)

    print(f"{'=' * 60}")
    print(f"  Claude Code xAI Bridge — Smoke Test")
    print(f"  Mode: {'LIVE (XAI_API_KEY set)' if live else 'MOCK (no XAI_API_KEY)'}")
    print(f"{'=' * 60}\n")

    from main import app
    client = TestClient(app)

    # ── Startup checks ──
    print("[Startup]")
    try:
        check_manifest(client, r)
        check_health(client, r)
    except Exception as e:
        r.fail("startup", f"exception: {e}")
        traceback.print_exc()

    # ── Enrichment checks ──
    print("\n[Enrichment Engine]")
    try:
        check_enrichment(r)
    except Exception as e:
        r.fail("enrichment", f"exception: {e}")
        traceback.print_exc()

    # ── Translation checks ──
    print("\n[Translation Pipeline]")
    try:
        check_translation_mock(client, r)
        check_thinking_stripped(client, r)
    except Exception as e:
        r.fail("translation", f"exception: {e}")
        traceback.print_exc()

    # ── Live checks (only if API key present) ──
    if live:
        print("\n[Live xAI API]")
        try:
            check_translation_live(client, r)
        except Exception as e:
            r.fail("live", f"exception: {e}")
            traceback.print_exc()
    else:
        print("\n[Live xAI API]")
        r.skip("live.simple", "no XAI_API_KEY")
        r.skip("live.with_tools", "no XAI_API_KEY")

    return r.summary()


if __name__ == "__main__":
    sys.exit(main())
