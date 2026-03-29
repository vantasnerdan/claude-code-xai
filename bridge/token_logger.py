"""Token count logging for the xAI bridge.

Tracks and logs token usage per request: input tokens, output tokens,
total tokens, and enrichment overhead. Uses the xAI response usage
field for actual token counts and serialized JSON size delta for
enrichment overhead estimation.

Enrichment overhead is measured as the character-length increase in
serialized tool definitions before and after enrichment. Dividing by 4
gives a rough token estimate (standard approximation for English text).
"""

from __future__ import annotations

import json
from typing import Any

from bridge.logging_config import get_logger

logger = get_logger("tokens")


def measure_enrichment_overhead(
    original_tools: list[dict[str, Any]],
    enriched_tools: list[dict[str, Any]],
) -> int:
    """Measure enrichment overhead as estimated token delta.

    Serializes tool definitions before and after enrichment, computes
    the character delta, and converts to estimated tokens.

    Returns:
        Estimated token overhead from enrichment (0 if no change or shrink).
    """
    if not original_tools and not enriched_tools:
        return 0

    original_chars = len(json.dumps(original_tools, default=str))
    enriched_chars = len(json.dumps(enriched_tools, default=str))
    delta_chars = enriched_chars - original_chars
    if delta_chars <= 0:
        return 0
    return max(1, delta_chars // 4)


def log_token_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    enrichment_overhead_tokens: int = 0,
    elapsed_seconds: float = 0.0,
    is_streaming: bool = False,
    model: str = "",
) -> dict[str, Any]:
    """Log token counts for a completed request.

    Emits a structured INFO log line with all token metrics. Returns
    the token summary dict for callers that need it.

    Args:
        input_tokens: Tokens consumed by the prompt (from xAI usage).
        output_tokens: Tokens generated in the response (from xAI usage).
        cached_tokens: Tokens served from xAI prompt cache (90% discount).
        enrichment_overhead_tokens: Estimated tokens added by enrichment.
        elapsed_seconds: Total request time in seconds.
        is_streaming: Whether this was a streaming request.
        model: The resolved model name used for this request.
    """
    total_tokens = input_tokens + output_tokens
    mode = "stream" if is_streaming else "sync"

    summary: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "enrichment_overhead_tokens": enrichment_overhead_tokens,
    }
    if cached_tokens:
        summary["cached_tokens"] = cached_tokens

    cache_part = f" cached={cached_tokens}" if cached_tokens else ""
    logger.info(
        "Token usage: model=%s input=%d output=%d total=%d%s "
        "enrichment_overhead=%d mode=%s elapsed=%.2fs",
        model or "unknown",
        input_tokens,
        output_tokens,
        total_tokens,
        cache_part,
        enrichment_overhead_tokens,
        mode,
        elapsed_seconds,
    )

    return summary
