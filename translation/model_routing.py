"""Model routing: detect which xAI endpoint a model requires.

Multi-agent models (grok-4.20-multi-agent) require /v1/responses.
All other models use /v1/chat/completions. Detection is based on the
resolved Grok model name, not the incoming Anthropic model name.
"""

from __future__ import annotations

from enum import Enum


class XAIEndpoint(Enum):
    """xAI API endpoint for a model."""

    CHAT_COMPLETIONS = "/chat/completions"
    RESPONSES = "/responses"


# Model name patterns that require the Responses API.
# Checked via substring match against the resolved Grok model name.
_RESPONSES_PATTERNS: frozenset[str] = frozenset({
    "multi-agent",
})

# Explicit model names that require the Responses API.
_RESPONSES_MODELS: frozenset[str] = frozenset({
    "grok-4.20-multi-agent",
})


def detect_endpoint(resolved_model: str) -> XAIEndpoint:
    """Determine which xAI endpoint a resolved model requires.

    Args:
        resolved_model: The Grok model name after MODEL_MAP resolution.

    Returns:
        The endpoint enum value for the model.
    """
    if resolved_model in _RESPONSES_MODELS:
        return XAIEndpoint.RESPONSES

    model_lower = resolved_model.lower()
    for pattern in _RESPONSES_PATTERNS:
        if pattern in model_lower:
            return XAIEndpoint.RESPONSES

    return XAIEndpoint.CHAT_COMPLETIONS
