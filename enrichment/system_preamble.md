# System Prompt Preamble

Behavioral conventions injected into the system prompt for every request through the xAI bridge. This is the GLOBAL context that teaches Grok how to operate as a Claude Code agent.

## Purpose

Claude Code agents learn tool usage patterns, sequencing rules, and safety conventions through Anthropic's RL training. Grok has no equivalent training. This preamble transfers that behavioral knowledge at the system prompt level, complementing the per-tool enrichment from the enrichment engine.

## Architecture

```
Request Flow:
  Claude Code -> Anthropic Messages API
    -> forward.py: anthropic_to_openai()
      -> system_preamble prepended to system message
      -> tools enriched via tool_enrichment_hook
    -> xAI Chat Completions API (Grok)
```

The preamble is injected in `translation/forward.py` via `TranslationConfig.system_prompt_preamble`. The `get_system_preamble()` function respects the `PREAMBLE_ENABLED` environment variable for A/B benchmarking.

## Six Behavioral Areas

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
| `PREAMBLE_ENABLED` | `true` | Set to `false` to disable preamble injection (benchmark mode) |

## API

```python
from enrichment.system_preamble import get_system_preamble, inject_system_preamble

# Get the preamble (respects PREAMBLE_ENABLED env var)
preamble = get_system_preamble()

# Inject into OpenAI-format messages
messages = [{"role": "system", "content": "You are helpful."}]
enriched = inject_system_preamble(messages, preamble)
# Result: system message content = preamble + "\n\n" + "You are helpful."
```
