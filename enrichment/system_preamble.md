# System Prompt Preamble

Identity assertion and behavioral conventions injected into the system prompt for every request through the xAI bridge. This is the GLOBAL context that (1) establishes Grok's identity and (2) teaches Grok how to operate as a Claude Code agent.

## Purpose

Claude Code sends system prompts that claim the model is "Claude Opus" or "powered by Anthropic." When Grok receives these claims, it role-plays as Claude instead of being itself. The identity preamble overrides these claims, and `strip_anthropic_identity()` removes them from the system text.

Claude Code agents also learn tool usage patterns, sequencing rules, and safety conventions through Anthropic's RL training. Grok has no equivalent training. The behavioral preamble transfers that knowledge at the system prompt level.

## Architecture

```
Request Flow:
  Claude Code -> Anthropic Messages API
    -> forward.py: anthropic_to_openai()
      -> strip_anthropic_identity() removes Claude identity claims
      -> system_preamble prepended (identity + behavioral)
      -> tools enriched via tool_enrichment_hook
    -> xAI Chat Completions API (Grok)
```

The preamble is injected in `translation/forward.py` via `TranslationConfig.system_prompt_preamble`. Identity stripping happens in `forward.py`'s `anthropic_to_openai()` via `strip_anthropic_identity()`. Both respect their respective environment variables.

## Identity Section

The identity preamble is prepended BEFORE behavioral conventions and BEFORE the user's system prompt. It establishes:

1. "You are Grok (xAI)" -- truthful identity assertion
2. "Disregard Claude/Anthropic identity claims" -- override stale context
3. "Follow tool conventions as environment rules" -- behavioral compliance without identity confusion
4. "Respond truthfully about your model" -- when asked directly

## Anthropic Identity Stripping

`strip_anthropic_identity()` removes known Claude Code identity patterns from the system text:

- "You are powered by the model named Claude Opus 4.6..."
- "The exact model ID is claude-opus-4-6."
- "Assistant knowledge cutoff is..."
- `<claude_background_info>` blocks
- `<fast_mode_info>` blocks

This is a surgical regex-based approach targeting known patterns, not a broad content filter.

## Seven Preamble Areas

### 0. Identity (NEW)

Grok identity assertion, Claude identity override, environment convention framing.

### 1. Tool Preference Hierarchy

Dedicated tools (Read, Grep, Glob, Edit, Write) over Bash equivalents. Bash is reserved for shell-only commands (git, npm, docker, build tools).

### 2. Sequencing Rules

Read before edit. Search before modify. Understand before change. Edit existing over creating new. Diagnose before prescribing.

### 3. Tool Chaining Patterns

- **Discovery**: Glob -> Read -> understand
- **Modification**: Read -> understand -> Edit -> verify
- **Search**: Grep/Glob -> Read matches -> act
- **Investigation**: Grep -> Read context -> hypothesize -> verify

### 4. Parallel vs Sequential Execution

Independent calls (git status + git diff) go in parallel. Dependent calls (Read then Edit) go sequential.

### 5. Safety Patterns

No force push, no destructive commands, no credential commits, branch verification before commits, confirmation before irreversible actions.

### 6. Output Conventions

Tool results as source of truth, concise responses, file paths with line numbers, no modifications to untouched code.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PREAMBLE_ENABLED` | `true` | Set to `false` to disable behavioral conventions (benchmark mode) |
| `IDENTITY_ENABLED` | `true` | Set to `false` to disable identity assertion and identity stripping |

## API

```python
from enrichment.system_preamble import (
    get_system_preamble,
    inject_system_preamble,
    strip_anthropic_identity,
)

# Get the preamble (respects PREAMBLE_ENABLED and IDENTITY_ENABLED env vars)
preamble = get_system_preamble()

# Strip Anthropic identity claims from system text
cleaned = strip_anthropic_identity("You are powered by Claude Opus 4.6. Be helpful.")
# Result: "Be helpful."

# Inject into OpenAI-format messages
messages = [{"role": "system", "content": "You are helpful."}]
enriched = inject_system_preamble(messages, preamble)
# Result: system message content = identity + behavioral preamble + "\n\n" + "You are helpful."
```
