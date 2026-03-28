"""Tests for model routing logic.

As of issue #51, Responses API is the DEFAULT for all models.
Chat Completions is only used when XAI_USE_CHAT_COMPLETIONS=true
or for models in the _CHAT_COMPLETIONS_ONLY_MODELS set.

As of issue #52, the env var is cached at import time.  Tests patch
the cached ``_USE_CHAT_COMPLETIONS`` module variable directly so
they are decoupled from os.getenv timing.
"""

from unittest.mock import patch

import translation.model_routing as _mod
from translation.model_routing import detect_endpoint, XAIEndpoint, use_legacy_chat_completions


class TestDetectEndpointDefault:
    """Default behavior: all models route to Responses API."""

    def test_standard_model_uses_responses(self):
        assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.RESPONSES

    def test_grok_4_uses_responses(self):
        assert detect_endpoint("grok-4") == XAIEndpoint.RESPONSES

    def test_multi_agent_model_uses_responses(self):
        assert detect_endpoint("grok-4.20-multi-agent") == XAIEndpoint.RESPONSES

    def test_empty_model_uses_responses(self):
        assert detect_endpoint("") == XAIEndpoint.RESPONSES

    def test_grok_code_fast_uses_responses(self):
        assert detect_endpoint("grok-code-fast-1") == XAIEndpoint.RESPONSES

    def test_unknown_model_uses_responses(self):
        assert detect_endpoint("some-future-model") == XAIEndpoint.RESPONSES


class TestLegacyChatCompletionsOverride:
    """``_USE_CHAT_COMPLETIONS=True`` forces Chat Completions for all models."""

    def test_cached_true_forces_chat_completions(self):
        with patch.object(_mod, "_USE_CHAT_COMPLETIONS", True):
            assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.CHAT_COMPLETIONS

    def test_cached_false_keeps_responses(self):
        with patch.object(_mod, "_USE_CHAT_COMPLETIONS", False):
            assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.RESPONSES

    def test_multi_agent_also_forced_to_chat_completions(self):
        """Even multi-agent models are forced to Chat Completions when flag is set."""
        with patch.object(_mod, "_USE_CHAT_COMPLETIONS", True):
            assert detect_endpoint("grok-4.20-multi-agent") == XAIEndpoint.CHAT_COMPLETIONS

    def test_use_legacy_chat_completions_returns_cached_value(self):
        with patch.object(_mod, "_USE_CHAT_COMPLETIONS", True):
            assert use_legacy_chat_completions() is True
        with patch.object(_mod, "_USE_CHAT_COMPLETIONS", False):
            assert use_legacy_chat_completions() is False


class TestEndpointEnum:
    """Endpoint enum values remain correct."""

    def test_endpoint_enum_values(self):
        assert XAIEndpoint.CHAT_COMPLETIONS.value == "/chat/completions"
        assert XAIEndpoint.RESPONSES.value == "/responses"
