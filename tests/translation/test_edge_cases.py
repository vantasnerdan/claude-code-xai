"""Tests for edge cases in the translation layer.

These tests cover unusual, malformed, or boundary-condition inputs that
the translator must handle gracefully. Real-world traffic from Claude Code
and xAI will include unexpected payloads -- the translator must never crash.

Tests are organized as:
- Forward translation edge cases (Anthropic -> xAI, shared by both paths)
- LEGACY reverse edge cases (OpenAI CC -> Anthropic, via translation.reverse)
- PRIMARY reverse edge cases (Responses API -> Anthropic, via translation.responses_reverse)
"""

import json
import string
from typing import Any

import pytest

from translation.forward import anthropic_to_openai, translate_messages, translate_tools
from translation.reverse import openai_to_anthropic, translate_response
from translation.responses_reverse import responses_to_anthropic, translate_responses_response

from tests.translation.fixtures.anthropic_messages import (
    simple_text_message,
    full_request_with_tools,
)
from tests.translation.fixtures.openai_completions import (
    simple_completion,
    tool_call_completion,
)
from tests.translation.fixtures.responses_api import (
    simple_response as responses_simple,
    function_call_response as responses_function_call,
)


class TestEmptyContent:
    """Handling of empty, null, or missing content fields."""

    def test_empty_content_array(self) -> None:
        """An empty content array should not crash the translator."""
        message = {"role": "user", "content": []}
        result = translate_messages([message])

        assert len(result) >= 1
        assert result[0]["role"] == "user"

    def test_empty_string_content(self) -> None:
        """An empty string content should translate without error."""
        message = {"role": "user", "content": ""}
        result = translate_messages([message])

        assert result[0]["content"] == ""

    def test_none_content_in_openai_response(self) -> None:
        """LEGACY: A CC response with content=None (no tool_calls) should be handled."""
        response = simple_completion()
        response["choices"][0]["message"]["content"] = None

        result = openai_to_anthropic(response)
        # Should not crash; content should be an array (possibly with empty text)
        assert isinstance(result["content"], list)

    def test_empty_output_in_responses_api(self) -> None:
        """PRIMARY: A Responses API response with empty output should be handled."""
        response = {"id": "resp_empty", "output": [], "model": "grok-4-1-fast-reasoning", "usage": {}}

        result = responses_to_anthropic(response)
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"

    def test_empty_text_block(self) -> None:
        """A text block with empty text should translate."""
        message = {
            "role": "user",
            "content": [{"type": "text", "text": ""}],
        }
        result = translate_messages([message])
        assert result[0]["content"] == ""

    def test_missing_content_key(self) -> None:
        """A message without a content key should be handled gracefully."""
        message: dict[str, Any] = {"role": "assistant"}
        result = translate_messages([message])

        # Should not crash
        assert len(result) >= 1


class TestDeeplyNestedSchema:
    """Handling of complex nested JSON Schemas in tool definitions."""

    def test_nested_object_schema(self) -> None:
        """A tool with deeply nested object schema should translate."""
        tool = {
            "name": "ComplexTool",
            "description": "A tool with nested params.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "object",
                                "properties": {
                                    "host": {"type": "string"},
                                    "port": {"type": "integer"},
                                    "options": {
                                        "type": "object",
                                        "properties": {
                                            "ssl": {"type": "boolean"},
                                            "timeout": {"type": "number"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "required": ["config"],
            },
        }
        result = translate_tools([tool])

        assert len(result) == 1
        params = result[0]["function"]["parameters"]
        # Verify deep nesting survived
        db = params["properties"]["config"]["properties"]["database"]
        assert db["properties"]["options"]["properties"]["ssl"]["type"] == "boolean"

    def test_array_of_objects_schema(self) -> None:
        """A tool with an array of objects in the schema should translate."""
        tool = {
            "name": "BatchTool",
            "description": "Processes batches.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "value": {"type": "number"},
                            },
                            "required": ["id"],
                        },
                    },
                },
                "required": ["items"],
            },
        }
        result = translate_tools([tool])

        params = result[0]["function"]["parameters"]
        items_schema = params["properties"]["items"]["items"]
        assert items_schema["type"] == "object"
        assert "id" in items_schema["properties"]

    def test_schema_with_refs(self) -> None:
        """A schema with $ref should pass through without resolution."""
        tool = {
            "name": "RefTool",
            "description": "Uses $ref.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {"$ref": "#/definitions/DataType"},
                },
                "definitions": {
                    "DataType": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }
        result = translate_tools([tool])

        params = result[0]["function"]["parameters"]
        assert params["properties"]["data"]["$ref"] == "#/definitions/DataType"


class TestVeryLongContent:
    """Handling of extremely long content strings."""

    def test_long_text_message(self) -> None:
        """A very long text message should translate without truncation."""
        long_text = "x" * 100_000
        message = {
            "role": "user",
            "content": [{"type": "text", "text": long_text}],
        }
        result = translate_messages([message])

        assert len(result[0]["content"]) == 100_000

    def test_long_tool_result(self) -> None:
        """A very long tool result should translate without truncation."""
        long_output = "line\n" * 50_000
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_long_result",
                    "content": [{"type": "text", "text": long_output}],
                }
            ],
        }
        result = translate_messages([message])

        assert "line" in result[0]["content"]
        # Content should not be silently truncated
        assert result[0]["content"].count("line") == 50_000

    def test_long_openai_response(self) -> None:
        """LEGACY: A very long CC response should translate without truncation."""
        long_text = "word " * 20_000
        response = simple_completion()
        response["choices"][0]["message"]["content"] = long_text

        result = openai_to_anthropic(response)
        assert result["content"][0]["text"] == long_text

    def test_long_responses_api_response(self) -> None:
        """PRIMARY: A very long Responses API response should translate without truncation."""
        long_text = "word " * 20_000
        response = {
            "id": "resp_long",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": long_text}]},
            ],
            "model": "grok-4-1-fast-reasoning",
            "usage": {},
        }
        result = responses_to_anthropic(response)
        assert result["content"][0]["text"] == long_text


class TestSpecialCharactersInToolNames:
    """Handling of unusual characters in tool names and arguments."""

    def test_tool_name_with_underscores(self) -> None:
        """Tool names with underscores should translate."""
        tool = {
            "name": "my_custom_tool",
            "description": "Has underscores.",
            "input_schema": {"type": "object", "properties": {}},
        }
        result = translate_tools([tool])
        assert result[0]["function"]["name"] == "my_custom_tool"

    def test_tool_name_with_hyphens(self) -> None:
        """Tool names with hyphens should translate."""
        tool = {
            "name": "my-custom-tool",
            "description": "Has hyphens.",
            "input_schema": {"type": "object", "properties": {}},
        }
        result = translate_tools([tool])
        assert result[0]["function"]["name"] == "my-custom-tool"

    def test_tool_arguments_with_unicode(self) -> None:
        """Tool arguments containing unicode should survive round trip."""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_unicode_test",
                    "name": "Write",
                    "input": {
                        "file_path": "/home/user/projekt/daten.py",
                        "content": "# Ubersicht der Datenpunkte\nprint('Grusse aus Berlin')\n",
                    },
                }
            ],
        }
        result = translate_messages([msg])

        args = json.loads(result[0]["tool_calls"][0]["function"]["arguments"])
        assert "Ubersicht" in args["content"]
        assert "Grusse" in args["content"]

    def test_tool_arguments_with_json_special_chars(self) -> None:
        """Tool arguments with JSON-special characters (quotes, backslashes) survive."""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_json_special",
                    "name": "Bash",
                    "input": {
                        "command": 'echo "hello \\"world\\"" | grep -E "\\bworld\\b"',
                    },
                }
            ],
        }
        result = translate_messages([msg])

        args = json.loads(result[0]["tool_calls"][0]["function"]["arguments"])
        assert "echo" in args["command"]
        assert "\\" in args["command"]

    def test_tool_arguments_with_newlines(self) -> None:
        """Tool arguments with embedded newlines survive serialization."""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_newline_test",
                    "name": "Write",
                    "input": {
                        "file_path": "/tmp/test.py",
                        "content": "line1\nline2\nline3\n",
                    },
                }
            ],
        }
        result = translate_messages([msg])

        args = json.loads(result[0]["tool_calls"][0]["function"]["arguments"])
        assert args["content"] == "line1\nline2\nline3\n"
        assert args["content"].count("\n") == 3


class TestMissingOptionalFields:
    """Handling of requests/responses with missing optional fields."""

    def test_request_without_temperature(self) -> None:
        """A request without temperature should use a default."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [simple_text_message()],
        }
        result = anthropic_to_openai(request)

        # Should not crash; temperature should have a default
        assert "temperature" in result

    def test_request_without_stream(self) -> None:
        """A request without the stream flag should default to false."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [simple_text_message()],
        }
        result = anthropic_to_openai(request)

        assert result.get("stream", False) is False

    def test_request_without_tools(self) -> None:
        """A request with no tools should translate cleanly."""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [simple_text_message()],
        }
        result = anthropic_to_openai(request)

        # tools should be absent or None, not an empty list that confuses the API
        tools = result.get("tools")
        assert tools is None or tools == []

    def test_response_without_model_field(self) -> None:
        """LEGACY: A CC response missing the model field should still translate."""
        response = simple_completion()
        del response["model"]

        result = openai_to_anthropic(response)
        assert result["content"][0]["type"] == "text"

    def test_response_without_created_field(self) -> None:
        """LEGACY: A CC response missing the created field should still translate."""
        response = simple_completion()
        del response["created"]

        result = openai_to_anthropic(response)
        assert result["content"][0]["type"] == "text"

    def test_tool_call_without_type_field(self) -> None:
        """LEGACY: A CC tool_call entry missing the type field should still translate."""
        response = tool_call_completion()
        del response["choices"][0]["message"]["tool_calls"][0]["type"]

        result = openai_to_anthropic(response)
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1

    def test_responses_api_missing_model(self) -> None:
        """PRIMARY: Responses API response missing model field should still translate."""
        response = responses_simple()
        del response["model"]
        result = responses_to_anthropic(response)
        assert result["content"][0]["type"] == "text"

    def test_responses_api_missing_usage(self) -> None:
        """PRIMARY: Responses API response missing usage should provide defaults."""
        response = responses_simple()
        del response["usage"]
        result = responses_to_anthropic(response)
        assert result["usage"]["input_tokens"] >= 0
        assert result["usage"]["output_tokens"] >= 0

    def test_responses_api_invalid_arguments_json(self) -> None:
        """PRIMARY: Responses API function_call with invalid JSON arguments."""
        response = responses_function_call()
        response["output"][0]["arguments"] = "not valid json"
        result = responses_to_anthropic(response)
        tool = result["content"][0]
        assert tool["type"] == "tool_use"
        assert tool["input"] == {}


class TestUnknownFinishReason:
    """LEGACY: Handling of unexpected or future finish_reason values (CC path)."""

    def test_unknown_finish_reason_does_not_crash(self) -> None:
        """LEGACY: An unrecognized finish_reason should not crash the CC translator."""
        response = simple_completion()
        response["choices"][0]["finish_reason"] = "some_future_reason"

        result = openai_to_anthropic(response)
        assert "stop_reason" in result

    def test_null_finish_reason(self) -> None:
        """LEGACY: A null finish_reason (streaming intermediate) should be handled."""
        response = simple_completion()
        response["choices"][0]["finish_reason"] = None

        result = openai_to_anthropic(response)
        # Should not crash; stop_reason can be None for intermediate states
        assert "stop_reason" in result

    def test_empty_string_finish_reason(self) -> None:
        """LEGACY: An empty string finish_reason should be handled gracefully."""
        response = simple_completion()
        response["choices"][0]["finish_reason"] = ""

        result = openai_to_anthropic(response)
        assert "stop_reason" in result


class TestEmptyChoicesArray:
    """LEGACY: Handling of CC responses with no choices."""

    def test_empty_choices_handled(self) -> None:
        """LEGACY: A response with an empty choices array should not crash."""
        response = simple_completion()
        response["choices"] = []

        # Should raise an appropriate error or return a sensible default
        with pytest.raises((IndexError, KeyError, ValueError)):
            openai_to_anthropic(response)

    def test_multiple_choices_uses_first(self) -> None:
        """LEGACY: If multiple choices are present, the first is used."""
        response = simple_completion()
        response["choices"].append(
            {
                "index": 1,
                "message": {"role": "assistant", "content": "Second choice"},
                "finish_reason": "stop",
            }
        )

        result = openai_to_anthropic(response)
        # Should use first choice
        assert result["content"][0]["text"] == "Hello! How can I help you today?"


class TestContentBlockTypeVariants:
    """Handling of content block types beyond text and tool_use."""

    def test_image_content_block_passthrough(self) -> None:
        """An image content block should be handled (even if not fully supported)."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUg==",
                    },
                }
            ],
        }
        # Should not crash — may pass through, skip, or raise a clear error
        try:
            result = translate_messages([message])
            assert len(result) >= 1
        except (ValueError, NotImplementedError) as e:
            # Acceptable: raising a clear error for unsupported types
            assert "image" in str(e).lower() or "unsupported" in str(e).lower()

    def test_unknown_content_block_type(self) -> None:
        """An unknown content block type should be handled gracefully."""
        message = {
            "role": "user",
            "content": [
                {"type": "unknown_future_type", "data": "some data"},
            ],
        }
        try:
            result = translate_messages([message])
            assert len(result) >= 1
        except (ValueError, NotImplementedError, KeyError):
            # Acceptable: raising a clear error for unknown types
            pass


# ---------------------------------------------------------------------------
# PRIMARY: Responses API reverse edge cases
# ---------------------------------------------------------------------------


class TestResponsesApiStopReasonEdgeCases:
    """PRIMARY: Stop reason inference from Responses API output."""

    def test_text_only_stop_reason(self) -> None:
        """Text-only output infers stop_reason='end_turn'."""
        response = responses_simple()
        result = responses_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"

    def test_function_call_stop_reason(self) -> None:
        """Function call output infers stop_reason='tool_use'."""
        response = responses_function_call()
        result = responses_to_anthropic(response)
        assert result["stop_reason"] == "tool_use"

    def test_empty_output_stop_reason(self) -> None:
        """Empty output array infers stop_reason='end_turn'."""
        response = {"id": "resp_e", "output": [], "model": "grok-4-1-fast-reasoning", "usage": {}}
        result = responses_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"

    def test_unknown_output_type_does_not_crash(self) -> None:
        """Unknown output item type is skipped without crashing."""
        response = {
            "id": "resp_future",
            "output": [
                {"type": "some_future_type", "data": "test"},
                {"type": "message", "content": [{"type": "output_text", "text": "OK"}]},
            ],
            "model": "grok-4-1-fast-reasoning",
            "usage": {},
        }
        result = responses_to_anthropic(response)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "OK"


class TestResponsesApiErrorEdgeCases:
    """PRIMARY: Error response translation from Responses API."""

    def test_rate_limit_error(self) -> None:
        """429 error translates to Anthropic error format."""
        from tests.translation.fixtures.responses_api import error_response_429
        error = error_response_429()
        result = translate_responses_response(error, status_code=429)
        assert result["type"] == "error"
        assert result["error"]["type"] == "rate_limit_error"

    def test_server_error(self) -> None:
        """500 error translates to Anthropic error format."""
        from tests.translation.fixtures.responses_api import error_response_500
        error = error_response_500()
        result = translate_responses_response(error, status_code=500)
        assert result["type"] == "error"
        assert "error" in result

    def test_bad_request_error(self) -> None:
        """400 error translates correctly."""
        from tests.translation.fixtures.responses_api import error_response_400
        error = error_response_400()
        result = translate_responses_response(error, status_code=400)
        assert result["type"] == "error"
        assert result["error"]["type"] == "invalid_request_error"

    def test_error_includes_links(self) -> None:
        """Translated errors include _links (Agentic API Standard Pattern 2)."""
        from tests.translation.fixtures.responses_api import error_response_500
        error = error_response_500()
        result = translate_responses_response(error, status_code=500)
        links = result.get("_links") or result.get("error", {}).get("_links")
        assert links is not None

    def test_error_includes_suggestion(self) -> None:
        """Translated errors include suggestion (Agentic API Standard Pattern 3)."""
        from tests.translation.fixtures.responses_api import error_response_429
        error = error_response_429()
        result = translate_responses_response(error, status_code=429)
        suggestion = result["error"].get("suggestion") or result.get("suggestion", "")
        assert len(suggestion) > 0


class TestResponsesApiToolCallEdgeCases:
    """PRIMARY: Tool call edge cases in Responses API format."""

    def test_function_call_with_dict_arguments(self) -> None:
        """Function call with dict (not string) arguments should work."""
        response = {
            "id": "resp_dict_args",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_dict",
                    "name": "Bash",
                    "arguments": {"command": "ls -la"},
                },
            ],
            "model": "grok-4-1-fast-reasoning",
            "usage": {},
        }
        result = responses_to_anthropic(response)
        tool = result["content"][0]
        assert tool["type"] == "tool_use"
        assert tool["input"] == {"command": "ls -la"}

    def test_multiple_function_calls(self) -> None:
        """Multiple parallel function calls all translate correctly."""
        from tests.translation.fixtures.responses_api import multi_function_call_response
        response = multi_function_call_response()
        result = responses_to_anthropic(response)
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 2
        assert result["stop_reason"] == "tool_use"

    def test_text_and_function_call_coexist(self) -> None:
        """Text and function call in same response both translate."""
        from tests.translation.fixtures.responses_api import multi_function_call_response
        response = multi_function_call_response()
        result = responses_to_anthropic(response)
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert len(tool_blocks) == 2
        assert "compare" in text_blocks[0]["text"].lower()

    def test_reasoning_block_stripped(self) -> None:
        """Reasoning blocks in output are stripped."""
        from tests.translation.fixtures.responses_api import reasoning_response
        response = reasoning_response()
        result = responses_to_anthropic(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "The answer is 42."
