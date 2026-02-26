"""System prompt preamble for Claude Code behavioral conventions.

Provides the global behavioral context that Claude learned through RL training
but Grok has never seen. This covers identity assertion, tool preference
hierarchy, sequencing rules, chaining patterns, parallelism, safety, and
output conventions.

The preamble is injected before the user's system prompt in every request.
It can be disabled via PREAMBLE_ENABLED=false for benchmarking.
Identity assertion can be disabled separately via IDENTITY_ENABLED=false.
"""

from __future__ import annotations

import os
import re
from typing import Any

_IDENTITY_PREAMBLE = """\
# Identity

You are Grok (xAI), running inside the Claude Code environment via the xAI bridge. \
The system prompt below comes from Claude Code and will refer to Claude, Anthropic, \
and specific Anthropic model names. Disregard those identity claims -- you are Grok.

Follow all Claude Code tool conventions, output formats, and safety guidelines \
exactly as described. These are environment conventions, not identity claims.

When asked what model you are, respond truthfully: you are Grok by xAI, \
operating through the Claude Code interface.
"""

# Patterns that assert Anthropic/Claude model identity in system prompts.
# These are stripped from the system text before forwarding to Grok.
_ANTHROPIC_IDENTITY_PATTERNS: list[re.Pattern[str]] = [
    # "You are powered by the model named Claude Opus 4.6..."
    # Uses [^\n]*? (lazy, allows dots) + sentence-end anchor to handle
    # version numbers like "4.6" without stopping at the internal period.
    re.compile(
        r"You are powered by the model named[^\n]*?(?:\.\s+|\.\s*$)"
        r"(?:The exact model ID is[^\n]*?(?:\.\s+|\.\s*$))?",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "You are powered by Claude Opus 4.6" (shorter variant)
    re.compile(
        r"You are powered by Claude[^\n]*?(?:\.\s+|\.\s*$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Standalone model ID references like "The exact model ID is claude-opus-4-6."
    # Model IDs use hyphens not dots, so [^\n.] is safe here.
    re.compile(
        r"The exact model ID is claude[^\n.]*\.\s*",
        re.IGNORECASE,
    ),
    # "Assistant knowledge cutoff is ..." (Anthropic-specific framing)
    # Month names don't contain dots, so [^\n.] is safe here.
    re.compile(
        r"Assistant knowledge cutoff is[^\n.]*\.\s*",
        re.IGNORECASE,
    ),
    # <claude_background_info> blocks
    re.compile(
        r"<claude_background_info>\s*.*?\s*</claude_background_info>\s*",
        re.DOTALL | re.IGNORECASE,
    ),
    # <fast_mode_info> blocks referencing Claude
    re.compile(
        r"<fast_mode_info>\s*.*?\s*</fast_mode_info>\s*",
        re.DOTALL | re.IGNORECASE,
    ),
]

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

## 7. Orchestration Tools — Planning vs Execution

Critical distinction: planning tools and execution tools are SEPARATE steps.

**TaskCreate** = TRACKING ENTRY ONLY
- Creates a checklist item in the task list
- No work is performed. No subagent is launched. No API calls are made.
- Purpose: organize and plan what needs to be done

**Task** = LAUNCHES A REAL SUBAGENT
- Starts an autonomous agent that performs actual work
- The subagent reads files, makes API calls, writes code, posts comments
- Use `run_in_background: true` for non-blocking execution
- ALWAYS structure the prompt with:
  - **Task scope**: one clear deliverable
  - **Context**: relevant background, file paths, prior decisions
  - **Acceptance criteria**: how to verify the work is complete
  - **Constraints**: what NOT to do, boundaries, dependencies
  - **Reminders**: skills to read, conventions to follow

**TaskUpdate / TaskGet / TaskList** = tracking management only, no work performed

**The workflow:**
1. TaskCreate to plan and track (optional, for organization)
2. Task tool to execute (required, for actual work)
3. Planning is NOT doing. Creating a tracking entry does NOT launch work.

A vague delegation ("review this") produces vague results. \
A structured delegation with scope, context, and criteria produces focused, \
verifiable work.
"""


def get_system_preamble() -> str:
    """Return the full system prompt preamble text.

    Identity section is prepended when IDENTITY_ENABLED is true (default).
    Behavioral conventions follow when PREAMBLE_ENABLED is true (default).
    Returns an empty string only when both are disabled.
    """
    parts: list[str] = []

    identity_enabled = os.getenv("IDENTITY_ENABLED", "true").lower()
    if identity_enabled != "false":
        parts.append(_IDENTITY_PREAMBLE)

    preamble_enabled = os.getenv("PREAMBLE_ENABLED", "true").lower()
    if preamble_enabled != "false":
        parts.append(_PREAMBLE)

    return "\n".join(parts)


def _strip_text(text: str) -> str:
    """Apply identity-stripping regexes to a single string.

    Returns the cleaned text with collapsed blank lines and stripped whitespace.
    """
    result = text
    for pattern in _ANTHROPIC_IDENTITY_PATTERNS:
        result = pattern.sub("", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_anthropic_identity(
    system: str | list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Remove Anthropic/Claude identity assertions from a system prompt.

    Handles both system prompt formats sent by the Anthropic Messages API:
    - String: applies regex patterns directly
    - List of content blocks: iterates blocks, strips text in each
      ``{"type": "text"}`` block, removes blocks that become empty

    Strips patterns like "You are powered by Claude Opus 4.6",
    model ID references, and ``<claude_background_info>`` blocks.
    If IDENTITY_ENABLED is false, returns the input unchanged.
    """
    identity_enabled = os.getenv("IDENTITY_ENABLED", "true").lower()
    if identity_enabled == "false":
        return system

    if isinstance(system, str):
        return _strip_text(system)

    if isinstance(system, list):
        result_blocks: list[dict[str, Any]] = []
        for block in system:
            if not isinstance(block, dict):
                result_blocks.append(block)
                continue
            if block.get("type") != "text":
                result_blocks.append(block)
                continue
            cleaned = _strip_text(block.get("text", ""))
            if cleaned:
                result_blocks.append({**block, "text": cleaned})
        return result_blocks

    raise TypeError(
        f"Expected str or list for system field, got {type(system).__name__}"
    )


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
