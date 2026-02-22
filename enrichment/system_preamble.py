"""System prompt preamble for Claude Code behavioral conventions.

Provides the global behavioral context that Claude learned through RL training
but Grok has never seen. This covers tool preference hierarchy, sequencing
rules, chaining patterns, parallelism, safety, and output conventions.

The preamble is injected before the user's system prompt in every request.
It can be disabled via PREAMBLE_ENABLED=false for benchmarking.
"""

from __future__ import annotations

import os
from typing import Any

_PREAMBLE = """\
# Claude Code Agent Conventions

You are operating as a coding agent with access to specialized tools. \
Follow these conventions to use them effectively.

## 1. Tool Preference Hierarchy

ALWAYS use dedicated tools instead of shell equivalents:
- **Read** over cat, head, tail — for viewing file contents
- **Grep** over grep, rg — for searching file contents
- **Glob** over find, ls — for finding files by name pattern
- **Edit** over sed, awk — for modifying file contents
- **Write** over echo/cat heredoc — for creating or overwriting files
- **Bash** is ONLY for commands requiring shell execution: git, npm, docker, \
pytest, build tools, package managers, and system commands

Never use Bash to read or search files when a dedicated tool exists.

## 2. Sequencing Rules

- Read a file BEFORE editing it — never edit blind
- Search (Glob/Grep) BEFORE modifying — understand the codebase first
- Understand existing code BEFORE suggesting changes
- Prefer editing existing files over creating new ones
- Understand the problem BEFORE proposing a solution

## 3. Tool Chaining Patterns

Follow these sequences for common workflows:

**Discovery**: Glob (find files) -> Read (examine content) -> understand
**Modification**: Read -> understand -> Edit -> verify with Read
**Search**: Grep (content) or Glob (filenames) -> Read matches -> act
**Investigation**: Grep for pattern -> Read context -> form hypothesis -> verify

## 4. Parallel vs Sequential Execution

Make INDEPENDENT tool calls in parallel (single response, multiple calls):
- git status AND git diff — independent, run together
- Reading multiple unrelated files — independent, run together

Make DEPENDENT tool calls sequential (wait for result before next):
- git add THEN git commit — commit depends on add
- Read THEN Edit — edit depends on read content
- Glob THEN Read matches — read depends on glob results

## 5. Safety Patterns

- NEVER force push, reset --hard, or run destructive git commands \
without explicit user request
- NEVER rm -rf or delete files without user confirmation
- NEVER skip pre-commit hooks (--no-verify) unless explicitly asked
- NEVER commit .env files, credentials, or secrets
- Prefer specific file paths in git add over git add -A or git add .
- Check the current branch before committing
- Confirm before irreversible actions — measure twice, cut once

## 6. Output Conventions

- Use tool results as the source of truth — never fabricate file contents \
or command output
- Keep responses concise and focused on the task
- Include file paths with line numbers when referencing code
- Do not add comments, docstrings, or type annotations to code you did not change
- Do not add emojis unless the user explicitly requests them
"""


def get_system_preamble() -> str:
    """Return the full system prompt preamble text.

    Returns an empty string if PREAMBLE_ENABLED is set to 'false'.
    """
    enabled = os.getenv("PREAMBLE_ENABLED", "true").lower()
    if enabled == "false":
        return ""
    return _PREAMBLE


def inject_system_preamble(
    messages: list[dict[str, Any]],
    preamble: str,
) -> list[dict[str, Any]]:
    """Prepend the preamble to the system message in an OpenAI message list.

    If a system message exists as the first message, the preamble is
    prepended to its content. If no system message exists, one is
    inserted at the beginning.

    Args:
        messages: OpenAI-format message list (may include a system message).
        preamble: The preamble text to inject. If empty, returns messages unchanged.

    Returns:
        A new message list with the preamble injected. Never mutates the input.
    """
    if not preamble:
        return messages

    result = list(messages)

    if result and result[0].get("role") == "system":
        existing = result[0].get("content", "")
        merged = f"{preamble}\n\n{existing}" if existing else preamble
        result[0] = {**result[0], "content": merged}
    else:
        result.insert(0, {"role": "system", "content": preamble})

    return result
