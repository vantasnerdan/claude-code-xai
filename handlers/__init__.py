"""Request handlers for the xAI bridge.

Responses API is the default handler for all models (issue #51).
Chat Completions handler is a thin wrapper that delegates to Responses (issue #52).
"""

from handlers.responses import handle_responses, stream_responses
from handlers.chat_completions import handle_chat_completions, stream_chat

__all__ = [
    "handle_responses",
    "stream_responses",
    "handle_chat_completions",
    "stream_chat",
]
