"""Request handlers for the xAI bridge.

Responses API is the default handler for all models (issue #51).
Chat Completions is retained as a legacy fallback.
"""

from handlers.chat_completions import handle_chat_completions, stream_chat
from handlers.responses import handle_responses, stream_responses

__all__ = [
    "handle_chat_completions",
    "stream_chat",
    "handle_responses",
    "stream_responses",
]
