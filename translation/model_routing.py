"""Model routing: detect which xAI endpoint a model requires.

As of issue #51, the Responses API (/v1/responses) is the DEFAULT endpoint
for ALL models. The Chat Completions API (/v1/chat/completions) is retained
as an opt-in legacy fallback via XAI_USE_CHAT_COMPLETIONS=true.

Detection is based on:
1. The XAI_USE_CHAT_COMPLETIONS environment variable (overrides everything)
2. The resolved Grok model name (explicit legacy models)
3. Default: Responses API

As of issue #52, the env var is cached at import time (not per-request),
and the public API is ``use_legacy_chat_completions()`` instead of the
previously-private ``_force_chat_completions()``.
"""

from __future__ import annotations

import os
from enum import Enum


class XAIEndpoint(Enum):
    """xAI API endpoint for a model."""

    CHAT_COMPLETIONS = "/chat/completions"
    RESPONSES = "/responses"


# Models that are known to NOT support the Responses API.
# These are forced to Chat Completions regardless of the default.
# Empty for now — all current xAI models support /v1/responses.
_CHAT_COMPLETIONS_ONLY_MODELS: frozenset[str] = frozenset()

# Cache the env var at import time so we don't call os.getenv() per-request.
# Issue #52: Kelvin's PR #55 review observation.
_USE_CHAT_COMPLETIONS: bool = os.getenv(
    "XAI_USE_CHAT_COMPLETIONS", ""
).lower() in ("true", "1", "yes")


def use_legacy_chat_completions() -> bool:
    """Check if the user has opted into legacy Chat Completions mode.

    Set XAI_USE_CHAT_COMPLETIONS=true to force all requests through
    the Chat Completions endpoint. This is a migration escape hatch.

    The value is cached at import time — restart the server to change it.
    """
    return _USE_CHAT_COMPLETIONS


# Backward-compatible alias — still works for internal callers but the
# underscore prefix is no longer the canonical name.
_force_chat_completions = use_legacy_chat_completions


def detect_endpoint(resolved_model: str) -> XAIEndpoint:
    """Determine which xAI endpoint a resolved model requires.

    Default: Responses API for all models (issue #51 migration).
    Override: Set XAI_USE_CHAT_COMPLETIONS=true to force Chat Completions.

    Args:
        resolved_model: The Grok model name after MODEL_MAP resolution.

    Returns:
        The endpoint enum value for the model.
    """
    # Global override: force legacy Chat Completions for all models.
    if use_legacy_chat_completions():
        return XAIEndpoint.CHAT_COMPLETIONS

    # Model-specific override: some models may only support Chat Completions.
    if resolved_model in _CHAT_COMPLETIONS_ONLY_MODELS:
        return XAIEndpoint.CHAT_COMPLETIONS

    # Default: Responses API for all models.
    return XAIEndpoint.RESPONSES
