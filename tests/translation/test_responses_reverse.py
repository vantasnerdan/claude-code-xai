"""Tests for Responses API reverse translation."""

from translation.responses_reverse import (
    responses_to_anthropic,
    translate_responses_response,
    _build_content,
    _infer_stop_reason,
)


class TestResponsesToAnthropic:
    """Tests for the main reverse translation function."""

    def test_text_message_output(self):
        response = {
            "id": "rs_abc123",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello there!"}],
                }
            ],
            "model": "grok-4.20-multi-agent",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = responses_to_anthropic(response)
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["id"].startswith("msg_")
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello there!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_function_call_output(self):
        response = {
            "id": "rs_def456",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "Read",
                    "arguments": '{"file_path": "/tmp/test.py"}',
                }
            ],
            "model": "grok-4.20-multi-agent",
            "usage": {},
        }
        result = responses_to_anthropic(response)
        assert len(result["content"]) == 1
        tool = result["content"][0]
        assert tool["type"] == "tool_use"
        assert tool["id"] == "call_abc"
        assert tool["name"] == "Read"
        assert tool["input"] == {"file_path": "/tmp/test.py"}
        assert result["stop_reason"] == "tool_use"

    def test_mixed_text_and_function_call(self):
        response = {
            "id": "rs_mixed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Let me check."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_xyz",
                    "name": "Bash",
                    "arguments": '{"command": "ls"}',
                },
            ],
            "model": "grok-4.20-multi-agent",
            "usage": {},
        }
        result = responses_to_anthropic(response)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"
        assert result["stop_reason"] == "tool_use"

    def test_reasoning_output_skipped(self):
        response = {
            "id": "rs_reasoning",
            "output": [
                {"type": "reasoning", "encrypted_content": "abc123..."},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "The answer is 42."}],
                },
            ],
            "model": "grok-4.20-multi-agent",
            "usage": {},
        }
        result = responses_to_anthropic(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "The answer is 42."

    def test_empty_output_produces_empty_text(self):
        response = {
            "id": "rs_empty",
            "output": [],
            "model": "grok-4.20-multi-agent",
            "usage": {},
        }
        result = responses_to_anthropic(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == ""

    def test_id_prefix(self):
        response = {"id": "rs_test", "output": [], "model": "grok-4.20-multi-agent", "usage": {}}
        result = responses_to_anthropic(response)
        assert result["id"].startswith("msg_")

    def test_usage_with_openai_field_names(self):
        response = {
            "id": "rs_usage",
            "output": [],
            "model": "grok-4.20-multi-agent",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        result = responses_to_anthropic(response)
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50


class TestTranslateResponsesResponse:
    """Tests for translate_responses_response() including errors."""

    def test_success_response(self):
        response = {
            "id": "rs_ok",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}
            ],
            "model": "grok-4.20-multi-agent",
            "usage": {},
        }
        result = translate_responses_response(response, status_code=200)
        assert result["type"] == "message"

    def test_400_error(self):
        error_response = {
            "error": {
                "type": "invalid_request_error",
                "message": "Multi Agent requests are not allowed on chat completions.",
            }
        }
        result = translate_responses_response(error_response, status_code=400)
        assert result["type"] == "error"
        assert result["error"]["type"] == "invalid_request_error"
        assert "Multi Agent" in result["error"]["message"]

    def test_429_error(self):
        error_response = {"error": {"type": "rate_limit", "message": "Too many requests"}}
        result = translate_responses_response(error_response, status_code=429)
        assert result["error"]["type"] == "rate_limit_error"

    def test_500_error(self):
        error_response = {"error": {"type": "server_error", "message": "Internal error"}}
        result = translate_responses_response(error_response, status_code=500)
        assert result["error"]["type"] == "api_error"

    def test_string_error_body(self):
        """Error body may be a string instead of a dict."""
        error_response = {"error": "Something went wrong"}
        result = translate_responses_response(error_response, status_code=500)
        assert result["type"] == "error"
        assert result["error"]["message"] == "Something went wrong"


class TestBuildContent:
    """Tests for _build_content()."""

    def test_unescape_text(self):
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "line1\\nline2"}],
            }
        ]
        content = _build_content(output)
        assert content[0]["text"] == "line1\nline2"

    def test_invalid_arguments_json(self):
        output = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "Test",
                "arguments": "not valid json",
            }
        ]
        content = _build_content(output)
        assert content[0]["type"] == "tool_use"
        assert content[0]["input"] == {}

    def test_dict_arguments(self):
        output = [
            {
                "type": "function_call",
                "call_id": "c2",
                "name": "Test",
                "arguments": {"key": "value"},
            }
        ]
        content = _build_content(output)
        assert content[0]["input"] == {"key": "value"}


class TestInferStopReason:
    """Tests for _infer_stop_reason()."""

    def test_text_only_is_end_turn(self):
        output = [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]
        assert _infer_stop_reason(output) == "end_turn"

    def test_function_call_is_tool_use(self):
        output = [{"type": "function_call", "call_id": "c1", "name": "Read"}]
        assert _infer_stop_reason(output) == "tool_use"

    def test_empty_output_is_end_turn(self):
        assert _infer_stop_reason([]) == "end_turn"
