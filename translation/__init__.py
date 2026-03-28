"""Bidirectional Anthropic <-> xAI protocol translation layer.

Converts Claude Code (Anthropic Messages API) traffic into xAI/Grok
format and back. As of issue #51, the Responses API is the default path
for all models. Chat Completions is retained as a legacy fallback
(XAI_USE_CHAT_COMPLETIONS=true). Custom translation with enrichment
injection hooks.
"""

from translation.forward import anthropic_to_openai, translate_messages, translate_tools, strip_thinking
from translation.reverse import openai_to_anthropic, translate_response, unescape_text
from translation.streaming import translate_sse_event, OpenAIToAnthropicStreamAdapter
from translation.config import TranslationConfig
from translation.model_routing import detect_endpoint, XAIEndpoint
from translation.shared import flatten_system
from translation.responses_forward import anthropic_to_responses
from translation.responses_reverse import responses_to_anthropic, translate_responses_response
from translation.responses_streaming import ResponsesStreamAdapter
from translation.tools import enrich_tools, translate_tools_responses

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
    "detect_endpoint",
    "XAIEndpoint",
    "flatten_system",
    "anthropic_to_responses",
    "responses_to_anthropic",
    "translate_responses_response",
    "ResponsesStreamAdapter",
    "enrich_tools",
    "translate_tools_responses",
]
