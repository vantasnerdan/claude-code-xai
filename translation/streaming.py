"""Streaming SSE: OpenAI delta stream -> Anthropic event stream."""
from __future__ import annotations
import json, uuid
from typing import Any, AsyncIterator
from translation.config import STOP_REASON_MAP


def _msg_start(chunk: dict[str, Any] | None = None) -> dict[str, Any]:
    c = chunk or {}
    cid = c.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    return {"type": "message_start", "message": {
        "id": f"msg_{cid}" if not cid.startswith("msg_") else cid,
        "type": "message", "role": "assistant", "content": [],
        "model": c.get("model", "grok-4-1-fast-reasoning"),
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 1}}}


def _close(reason: str, usage: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [{"type": "message_delta", "delta": {"stop_reason": STOP_REASON_MAP.get(reason, "end_turn"),
             "stop_sequence": None}, "usage": {"output_tokens": (usage or {}).get("completion_tokens", 0)}},
            {"type": "message_stop"}]

def _tool_events(tc: dict[str, Any]) -> list[dict[str, Any]]:
    func, idx = tc.get("function", {}), tc.get("index", 0)
    evts: list[dict[str, Any]] = []
    if tc.get("id"):
        evts.append({"type": "content_block_start", "index": idx,
                      "content_block": {"type": "tool_use", "id": tc["id"], "name": func.get("name", ""), "input": {}}})
    if func.get("arguments", ""):
        evts.append({"type": "content_block_delta", "index": idx,
                      "delta": {"type": "input_json_delta", "partial_json": func["arguments"]}})
    return evts


def translate_sse_event(chunk: dict[str, Any], is_first: bool = False, is_last: bool = False) -> list[dict[str, Any]]:
    """Translate a single OpenAI chunk to Anthropic event(s)."""
    evts: list[dict[str, Any]] = []
    choices = chunk.get("choices", [])
    if not choices:
        return evts
    delta, finish = choices[0].get("delta", {}), choices[0].get("finish_reason")
    if is_first:
        evts.append(_msg_start(chunk))
    text = delta.get("content")
    if text is not None and text != "":
        evts.append({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
    for tc in delta.get("tool_calls", []):
        evts.extend(_tool_events(tc))
    if is_last or finish is not None:
        evts.append({"type": "content_block_stop", "index": 0})
        evts.extend(_close(finish or "stop", chunk.get("usage")))
    return evts


class OpenAIToAnthropicStreamAdapter:
    """Async adapter: OpenAI SSE lines -> Anthropic event dicts."""

    def __init__(self, source: AsyncIterator[str]) -> None:
        self._src = source
        self._on = self._done = self._topen = self._bopen = False
        self._q: list[dict[str, Any]] = []

    def __aiter__(self) -> OpenAIToAnthropicStreamAdapter:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._q:
            return self._q.pop(0)
        if self._done:
            raise StopAsyncIteration
        await self._fill()
        if self._q:
            return self._q.pop(0)
        raise StopAsyncIteration

    async def _fill(self) -> None:
        try:
            while not self._q:
                try:
                    line = await self._src.__anext__()
                except StopAsyncIteration:
                    self._q.extend(self._end())
                    return
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    self._q.extend(self._end())
                    return
                try:
                    self._q.extend(self._xlate(json.loads(payload)))
                except json.JSONDecodeError:
                    pass
        except ConnectionError as e:
            self._q.append({"type": "error", "error": {"type": "connection_error", "message": str(e)}})
            self._done = True

    def _xlate(self, c: dict[str, Any]) -> list[dict[str, Any]]:
        choices = c.get("choices", [])
        if not choices:
            return []
        ev: list[dict[str, Any]] = []
        delta, fin = choices[0].get("delta", {}), choices[0].get("finish_reason")
        if not self._on:
            self._on = True
            ev.append(_msg_start(c))
        text = delta.get("content")
        if text is not None:
            if not self._bopen and not self._topen:
                ev.append({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                self._bopen = True
            if text != "":
                ev.append({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
        for tc in delta.get("tool_calls", []):
            if tc.get("id") and self._bopen:
                ev.append({"type": "content_block_stop", "index": 0})
                self._bopen = False
            if tc.get("id"):
                self._topen = True
            ev.extend(_tool_events(tc))
        if fin is not None:
            ev.extend(self._cls(c, fin))
        return ev

    def _cls(self, c: dict[str, Any], reason: str) -> list[dict[str, Any]]:
        ev: list[dict[str, Any]] = []
        for f in ("_bopen", "_topen"):
            if getattr(self, f):
                ev.append({"type": "content_block_stop", "index": 0})
                setattr(self, f, False)
        ev.extend(_close(reason, c.get("usage")))
        self._done = True
        return ev

    def _end(self) -> list[dict[str, Any]]:
        if self._done:
            return []
        ev: list[dict[str, Any]] = []
        if not self._on:
            ev.append(_msg_start())
            self._on = True
        for f in ("_bopen", "_topen"):
            if getattr(self, f):
                ev.append({"type": "content_block_stop", "index": 0})
                setattr(self, f, False)
        ev.extend(_close("stop"))
        self._done = True
        return ev
