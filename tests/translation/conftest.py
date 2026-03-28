"""Translation-specific conftest -- parametrized fixtures for message type coverage.

Imports from fixture files and provides parametrized combinations
for exhaustive testing of the translation layer.

Fixtures are organized as:
- Anthropic message fixtures (used by all paths)
- LEGACY: OpenAI Chat Completions response fixtures (translation.reverse)
- PRIMARY: Responses API response fixtures (translation.responses_reverse)
- Streaming event fixtures (both legacy CC and primary Responses API)
"""

import pytest
from typing import Any

from tests.translation.fixtures.anthropic_messages import (
    simple_text_message,
    system_message_request,
    tool_use_response,
    tool_result_message,
    multi_turn_with_tools,
    parallel_tool_calls,
    full_request_with_tools,
)
from tests.translation.fixtures.openai_completions import (  # LEGACY CC fixtures
    simple_completion,
    tool_call_completion,
    multi_tool_call_completion,
    streaming_chunk,
    streaming_chunk_with_role,
    streaming_chunk_tool_call,
    streaming_chunk_finish,
    error_response_429,
    error_response_500,
    error_response_400,
)
from tests.translation.fixtures.responses_api import (  # PRIMARY Responses API fixtures
    simple_response as responses_simple,
    function_call_response as responses_function_call,
    multi_function_call_response as responses_multi_function_call,
    error_response_429 as responses_error_429,
    error_response_500 as responses_error_500,
    error_response_400 as responses_error_400,
)
from tests.translation.fixtures.streaming_events import (
    anthropic_message_start,
    anthropic_content_block_start,
    anthropic_content_block_delta,
    anthropic_tool_use_start,
    anthropic_tool_use_delta,
    anthropic_content_block_stop,
    anthropic_message_delta,
    anthropic_message_stop,
    anthropic_full_text_stream,
    openai_stream_chunks,
    openai_tool_call_stream_chunks,
)


# ---------------------------------------------------------------------------
# Anthropic message fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_text_msg() -> dict[str, Any]:
    """Single user text message."""
    return simple_text_message()


@pytest.fixture
def anthropic_system_request() -> dict[str, Any]:
    """Full request with top-level system field."""
    return system_message_request()


@pytest.fixture
def anthropic_tool_use_msg() -> dict[str, Any]:
    """Assistant message containing tool_use block."""
    return tool_use_response()


@pytest.fixture
def anthropic_tool_result_msg() -> dict[str, Any]:
    """User message containing tool_result block."""
    return tool_result_message()


@pytest.fixture
def anthropic_multi_turn() -> list[dict[str, Any]]:
    """Full multi-turn conversation with tool use cycle."""
    return multi_turn_with_tools()


@pytest.fixture
def anthropic_parallel_tools() -> dict[str, Any]:
    """Assistant message with multiple parallel tool_use blocks."""
    return parallel_tool_calls()


@pytest.fixture
def anthropic_full_request() -> dict[str, Any]:
    """Complete request body with tools defined."""
    return full_request_with_tools()


# ---------------------------------------------------------------------------
# LEGACY: OpenAI Chat Completions response fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_text_response() -> dict[str, Any]:
    """LEGACY: Simple CC text completion response."""
    return simple_completion()


@pytest.fixture
def openai_tool_response() -> dict[str, Any]:
    """Response with tool_calls in the message."""
    return tool_call_completion()


@pytest.fixture
def openai_multi_tool_response() -> dict[str, Any]:
    """Response with multiple parallel tool_calls."""
    return multi_tool_call_completion()


@pytest.fixture
def openai_chunk() -> dict[str, Any]:
    """Single streaming chunk with text delta."""
    return streaming_chunk()


@pytest.fixture
def openai_chunk_with_role() -> dict[str, Any]:
    """First streaming chunk with role in delta."""
    return streaming_chunk_with_role()


@pytest.fixture
def openai_tool_chunk() -> dict[str, Any]:
    """Streaming chunk with tool_call delta."""
    return streaming_chunk_tool_call()


@pytest.fixture
def openai_finish_chunk() -> dict[str, Any]:
    """Final streaming chunk with finish_reason."""
    return streaming_chunk_finish()


@pytest.fixture
def openai_rate_limit_error() -> dict[str, Any]:
    """429 rate limit error response."""
    return error_response_429()


@pytest.fixture
def openai_server_error() -> dict[str, Any]:
    """500 internal server error response."""
    return error_response_500()


@pytest.fixture
def openai_bad_request_error() -> dict[str, Any]:
    """400 bad request error response."""
    return error_response_400()


# ---------------------------------------------------------------------------
# PRIMARY: Responses API response fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def responses_api_text_response() -> dict[str, Any]:
    """PRIMARY: Simple Responses API text response."""
    return responses_simple()


@pytest.fixture
def responses_api_tool_response() -> dict[str, Any]:
    """PRIMARY: Responses API response with function_call."""
    return responses_function_call()


@pytest.fixture
def responses_api_multi_tool_response() -> dict[str, Any]:
    """PRIMARY: Responses API response with multiple function_calls."""
    return responses_multi_function_call()


# ---------------------------------------------------------------------------
# Streaming event fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_stream_events() -> list[dict[str, Any]]:
    """Complete Anthropic event sequence for a text response."""
    return anthropic_full_text_stream()


@pytest.fixture
def openai_stream_lines() -> list[str]:
    """LEGACY: Complete CC SSE data lines for a text response."""
    return openai_stream_chunks()


@pytest.fixture
def openai_tool_stream_lines() -> list[str]:
    """LEGACY: Complete CC SSE data lines for a tool call response."""
    return openai_tool_call_stream_chunks()


# ---------------------------------------------------------------------------
# Parametrized fixtures for exhaustive coverage
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("simple_text", id="text-message"),
        pytest.param("tool_use", id="tool-use-message"),
        pytest.param("tool_result", id="tool-result-message"),
        pytest.param("parallel_tools", id="parallel-tool-calls"),
    ]
)
def anthropic_message_variant(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrized fixture providing each Anthropic message type."""
    variants = {
        "simple_text": simple_text_message,
        "tool_use": tool_use_response,
        "tool_result": tool_result_message,
        "parallel_tools": parallel_tool_calls,
    }
    return variants[request.param]()


@pytest.fixture(
    params=[
        pytest.param("text", id="text-completion"),
        pytest.param("tool_call", id="tool-call-completion"),
        pytest.param("multi_tool", id="multi-tool-completion"),
    ]
)
def openai_response_variant(request: pytest.FixtureRequest) -> dict[str, Any]:
    """LEGACY: Parametrized fixture providing each CC response type."""
    variants = {
        "text": simple_completion,
        "tool_call": tool_call_completion,
        "multi_tool": multi_tool_call_completion,
    }
    return variants[request.param]()


@pytest.fixture(
    params=[
        pytest.param("text", id="responses-text"),
        pytest.param("function_call", id="responses-function-call"),
        pytest.param("multi_function_call", id="responses-multi-function-call"),
    ]
)
def responses_api_response_variant(request: pytest.FixtureRequest) -> dict[str, Any]:
    """PRIMARY: Parametrized fixture providing each Responses API response type."""
    variants = {
        "text": responses_simple,
        "function_call": responses_function_call,
        "multi_function_call": responses_multi_function_call,
    }
    return variants[request.param]()


@pytest.fixture(
    params=[
        pytest.param("rate_limit", id="429-rate-limit"),
        pytest.param("server_error", id="500-server-error"),
        pytest.param("bad_request", id="400-bad-request"),
    ]
)
def openai_error_variant(
    request: pytest.FixtureRequest,
) -> tuple[int, dict[str, Any]]:
    """LEGACY: Parametrized fixture providing each CC error type with status code."""
    variants: dict[str, tuple[int, Any]] = {
        "rate_limit": (429, error_response_429),
        "server_error": (500, error_response_500),
        "bad_request": (400, error_response_400),
    }
    status_code, factory = variants[request.param]
    return status_code, factory()


@pytest.fixture(
    params=[
        pytest.param("rate_limit", id="responses-429-rate-limit"),
        pytest.param("server_error", id="responses-500-server-error"),
        pytest.param("bad_request", id="responses-400-bad-request"),
    ]
)
def responses_api_error_variant(
    request: pytest.FixtureRequest,
) -> tuple[int, dict[str, Any]]:
    """PRIMARY: Parametrized fixture providing each Responses API error with status code."""
    variants: dict[str, tuple[int, Any]] = {
        "rate_limit": (429, responses_error_429),
        "server_error": (500, responses_error_500),
        "bad_request": (400, responses_error_400),
    }
    status_code, factory = variants[request.param]
    return status_code, factory()
