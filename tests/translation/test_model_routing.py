"""Tests for model routing logic.

As of issue #51, Responses API is the DEFAULT for all models.
Chat Completions is only used when XAI_USE_CHAT_COMPLETIONS=true
or for models in the _CHAT_COMPLETIONS_ONLY_MODELS set.
"""

import os
from unittest.mock import patch

from translation.model_routing import detect_endpoint, XAIEndpoint


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
    """XAI_USE_CHAT_COMPLETIONS=true forces Chat Completions for all models."""

    def test_env_true_forces_chat_completions(self):
        with patch.dict(os.environ, {"XAI_USE_CHAT_COMPLETIONS": "true"}):
            assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.CHAT_COMPLETIONS

    def test_env_1_forces_chat_completions(self):
        with patch.dict(os.environ, {"XAI_USE_CHAT_COMPLETIONS": "1"}):
            assert detect_endpoint("grok-4") == XAIEndpoint.CHAT_COMPLETIONS

    def test_env_yes_forces_chat_completions(self):
        with patch.dict(os.environ, {"XAI_USE_CHAT_COMPLETIONS": "yes"}):
            assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.CHAT_COMPLETIONS

    def test_env_TRUE_case_insensitive(self):
        with patch.dict(os.environ, {"XAI_USE_CHAT_COMPLETIONS": "TRUE"}):
            assert detect_endpoint("grok-4") == XAIEndpoint.CHAT_COMPLETIONS

    def test_env_false_keeps_responses(self):
        with patch.dict(os.environ, {"XAI_USE_CHAT_COMPLETIONS": "false"}):
            assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.RESPONSES

    def test_env_empty_keeps_responses(self):
        with patch.dict(os.environ, {"XAI_USE_CHAT_COMPLETIONS": ""}):
            assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.RESPONSES

    def test_env_unset_keeps_responses(self):
        env = os.environ.copy()
        env.pop("XAI_USE_CHAT_COMPLETIONS", None)
        with patch.dict(os.environ, env, clear=True):
            assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.RESPONSES

    def test_multi_agent_also_forced_to_chat_completions(self):
        """Even multi-agent models are forced to Chat Completions when env is set."""
        with patch.dict(os.environ, {"XAI_USE_CHAT_COMPLETIONS": "true"}):
            assert detect_endpoint("grok-4.20-multi-agent") == XAIEndpoint.CHAT_COMPLETIONS


class TestEndpointEnum:
    """Endpoint enum values remain correct."""

    def test_endpoint_enum_values(self):
        assert XAIEndpoint.CHAT_COMPLETIONS.value == "/chat/completions"
        assert XAIEndpoint.RESPONSES.value == "/responses"
