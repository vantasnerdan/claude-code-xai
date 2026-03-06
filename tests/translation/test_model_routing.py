"""Tests for model routing logic."""

from translation.model_routing import detect_endpoint, XAIEndpoint


class TestDetectEndpoint:
    """Tests for detect_endpoint()."""

    def test_standard_model_uses_chat_completions(self):
        assert detect_endpoint("grok-4-1-fast-reasoning") == XAIEndpoint.CHAT_COMPLETIONS

    def test_grok_4_uses_chat_completions(self):
        assert detect_endpoint("grok-4") == XAIEndpoint.CHAT_COMPLETIONS

    def test_multi_agent_model_uses_responses(self):
        assert detect_endpoint("grok-4.20-multi-agent") == XAIEndpoint.RESPONSES

    def test_multi_agent_substring_match(self):
        assert detect_endpoint("some-multi-agent-model") == XAIEndpoint.RESPONSES

    def test_multi_agent_case_insensitive(self):
        assert detect_endpoint("GROK-MULTI-AGENT-TEST") == XAIEndpoint.RESPONSES

    def test_empty_model_uses_chat_completions(self):
        assert detect_endpoint("") == XAIEndpoint.CHAT_COMPLETIONS

    def test_grok_code_fast_uses_chat_completions(self):
        assert detect_endpoint("grok-code-fast-1") == XAIEndpoint.CHAT_COMPLETIONS

    def test_endpoint_enum_values(self):
        assert XAIEndpoint.CHAT_COMPLETIONS.value == "/chat/completions"
        assert XAIEndpoint.RESPONSES.value == "/responses"
