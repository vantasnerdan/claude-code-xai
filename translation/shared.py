"""Shared translation utilities used by both Responses and Chat Completions paths.

Prevents duplication between forward.py (legacy) and responses_forward.py (default).
"""

from __future__ import annotations

from typing import Any


def flatten_system(system: str | list[dict[str, Any]]) -> str:
    """Flatten a system prompt to a single string.

    Handles both Anthropic system prompt formats:
    - String: returned as-is
    - List of content blocks: text blocks are joined with double newlines,
      non-text blocks are skipped

    Args:
        system: The system prompt in either Anthropic format.

    Returns:
        A single string suitable for the xAI system message.

    Raises:
        TypeError: If system is neither str nor list.
    """
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    raise TypeError(
        f"Expected str or list for system field, got {type(system).__name__}"
    )
