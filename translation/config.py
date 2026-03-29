"""Translation layer configuration.

Model name mapping, feature support matrix, and system prompt
template path. All configuration is centralized here.

Environment variables:
- GROK_MODEL: Override the resolved model name for all requests.
- XAI_API_KEY: xAI API key for authentication.
- XAI_USE_CHAT_COMPLETIONS: Set to "true" to force legacy Chat Completions
  path instead of the default Responses API (issue #51 migration escape hatch).
- PREAMBLE_ENABLED: Set to "false" to disable behavioral preamble injection.
- IDENTITY_ENABLED: Set to "false" to disable identity stripping/injection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from enrichment.system_preamble import get_system_preamble


# Anthropic model name -> xAI/Grok model name
MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4-20250514": "grok-4.20-reasoning-latest",
    "claude-opus-4-20250514": "grok-4",
    "claude-haiku-3-20240307": "grok-4.20-reasoning-latest",
    "claude-3-5-sonnet-20241022": "grok-4.20-reasoning-latest",
    "claude-3-5-haiku-20241022": "grok-4.20-reasoning-latest",
}

# OpenAI finish_reason -> Anthropic stop_reason
STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}

# Features that degrade gracefully (log warning, continue)
DEGRADED_FEATURES: frozenset[str] = frozenset({
    "stop_sequences",
    "top_k",
    "metadata",
    "tool_choice",
})

# Features that fail loudly (raise NotImplementedError)
UNSUPPORTED_FEATURES: frozenset[str] = frozenset()

# Features stripped from requests (Grok handles reasoning internally)
STRIPPED_FEATURES: frozenset[str] = frozenset({
    "thinking",
})


@dataclass(frozen=True)
class TranslationConfig:
    """Immutable translation configuration."""

    default_model: str = "grok-4.20-reasoning-latest"
    default_temperature: float = 0.7
    default_max_tokens: int = 131072
    system_prompt_preamble: str = field(
        default_factory=get_system_preamble
    )
    xai_api_key: str = field(
        default_factory=lambda: os.getenv("XAI_API_KEY", "")
    )

    def resolve_model(self, anthropic_model: str) -> str:
        """Map an Anthropic model name to a Grok model name."""
        env_model = os.getenv("GROK_MODEL")
        if env_model:
            return env_model
        return MODEL_MAP.get(anthropic_model, self.default_model)

    def map_stop_reason(self, finish_reason: str | None) -> str | None:
        """Map OpenAI finish_reason to Anthropic stop_reason."""
        if finish_reason is None:
            return None
        return STOP_REASON_MAP.get(finish_reason, "end_turn")
