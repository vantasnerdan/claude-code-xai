"""Bidirectional Anthropic <-> OpenAI protocol translation layer.

Converts Claude Code (Anthropic Messages API) traffic into xAI/Grok
(OpenAI Chat Completions API) format and back. Custom translation with
enrichment injection hooks at every boundary.
"""

from translation.forward import anthropic_to_openai, translate_messages, translate_tools, strip_thinking
from translation.reverse import openai_to_anthropic, translate_response, unescape_text
from translation.streaming import translate_sse_event, OpenAIToAnthropicStreamAdapter
from translation.config import TranslationConfig

__all__ = [
    "anthropic_to_openai",
    "translate_messages",
    "translate_tools",
    "strip_thinking",
    "openai_to_anthropic",
    "translate_response",
    "unescape_text",
    "translate_sse_event",
    "OpenAIToAnthropicStreamAdapter",
    "TranslationConfig",
]
