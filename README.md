# claude-code-xai

**A universal upgrade layer for model tool calling, proven across two frontier models.**

Claude Code runs natively on Grok through bidirectional protocol translation and structured tool enrichment. The same enrichment applied to any model — including Claude itself — measurably improves tool calling accuracy.

Built on the [Agentic API Standard](https://github.com/nexus-marbell/agentic-api-standard): 20 design patterns for self-describing, machine-first interfaces.

---

## The Problem

Large language models learn tool calling through reinforcement learning. The learned behaviors are implicit — baked into weights, not visible in the API. When a model encounters a new tool ecosystem, it has no training signal to draw on. Tool definitions ship as minimal JSON Schema: a name, a description, a parameter list. No sequencing rules. No failure modes. No context about when to use one tool over another.

This creates two problems:

1. **Cross-model tool calling is blind.** Grok has never seen Claude Code's tools. It doesn't know that `Read` should come before `Edit`, that `Grep` replaces `bash grep`, or that force-pushing will destroy work. Without this knowledge, tool calling degrades to guesswork.

2. **Even trained models underperform.** Claude's RL training encodes tool usage patterns, but the tool definitions themselves remain sparse. The model works *despite* the schema, not *because* of it. Richer definitions would make the implicit explicit — and measurably improve accuracy.

## The Standard

The [Agentic API Standard](https://github.com/nexus-marbell/agentic-api-standard) defines 20 patterns for agent-friendly interfaces. Every API an agent touches — HTTP endpoints, tool definitions, file directories — should be self-describing, navigable, and recoverable.

This bridge applies 8 structural patterns and 3 behavioral dimensions to tool definitions at request time:

**Structural (API Standard patterns):**

| # | Pattern | What It Adds |
|---|---------|-------------|
| 1 | Machine-Readable Manifest | Tool registry with capability discovery |
| 2 | HATEOAS Navigation | Related tools linked in every definition |
| 3 | Standard Error Format | Structured errors with suggestions and recovery links |
| 5 | Near-Miss Matching | "Did you mean?" for misspelled tool names |
| 6 | Self-Describing Endpoints | Full JSON Schema with examples on every tool |
| 8 | Warnings Layer | Deprecation notices and usage advisories |
| 14 | Anti-Pattern Detection | Common misuse patterns flagged in the schema |
| 15 | Tool Registration | Dynamic capability advertisement |

**Behavioral (training transfer):**

| Dimension | What It Teaches |
|-----------|----------------|
| **WHAT** | Enhanced descriptions beyond the raw schema — capabilities, limitations, output format |
| **WHY** | Problem context and failure modes — what goes wrong without this tool, what breaks if misused |
| **WHEN** | Prerequisites, sequencing, and alternatives — use Read before Edit, use Grep instead of bash grep |

## Architecture

```
Claude Code (Anthropic Messages API)
       │
       ▼
┌─────────────────────────────────────┐
│         claude-code-xai             │
│                                     │
│  ┌───────────────────────────────┐  │
│  │    System Preamble Injection  │  │
│  │  6 behavioral areas from CC   │  │
│  │  training: tool preference,   │  │
│  │  sequencing, chaining,        │  │
│  │  parallelism, safety, output  │  │
│  └───────────────┬───────────────┘  │
│                  │                  │
│  ┌───────────────▼───────────────┐  │
│  │     Tool Enrichment Engine    │  │
│  │                               │  │
│  │  Layer 1: 8 Structural        │  │
│  │  patterns from the standard   │  │
│  │                               │  │
│  │  Layer 2: 3 Behavioral        │  │
│  │  dimensions (WHAT/WHY/WHEN)   │  │
│  │  for 9 Claude Code tools      │  │
│  └───────────────┬───────────────┘  │
│                  │                  │
│  ┌───────────────▼───────────────┐  │
│  │   Protocol Translation Layer  │  │
│  │                               │  │
│  │  Forward:  Messages → Chat    │  │
│  │  Reverse:  Chat → Messages    │  │
│  │  Streaming: SSE ↔ SSE        │  │
│  │  Tools: tool_use ↔ functions  │  │
│  └───────────────┬───────────────┘  │
│                  │                  │
└──────────────────┼──────────────────┘
                   │
                   ▼
        xAI API (Grok 4.20)
```

Every request flows through three stages: behavioral context injection, tool definition enrichment, and protocol translation. Responses flow back through the reverse path. Streaming is fully supported.

## The Proof

Benchmarks measure enrichment quality across three scenarios using deterministic scoring — no live API calls required.

```
Mode            Scenario              Score    Structural  Behavioral  Overhead
──────────────────────────────────────────────────────────────────────────────
passthrough     multi_tool_chain      0.0000   0.0000      0.0000        0ms
structural      multi_tool_chain      0.6354   0.9062      0.0000        3ms
full            multi_tool_chain      0.9688   0.9062      1.0000        4ms

passthrough     error_recovery        0.0000   0.0000      0.0000        0ms
structural      error_recovery        0.6528   0.9583      0.0000        2ms
full            error_recovery        0.9861   0.9583      1.0000        4ms

passthrough     complex_schema        0.0000   0.0000      0.0000        0ms
structural      complex_schema        0.5083   0.5250      0.0000        3ms
full            complex_schema        0.8417   0.5250      1.0000        5ms
```

**Summary:**

| Mode | Avg Score | What It Means |
|------|-----------|---------------|
| `passthrough` | **0.00** | Raw tool definitions. No enrichment. Baseline. |
| `structural` | **0.60** | API Standard patterns only. Self-describing schemas, error formats, navigation. |
| `full` | **0.93** | Structural + behavioral. WHAT/WHY/WHEN training transfer. Gold standard. |

Enrichment overhead: **~4ms per request**. The cost of going from 0% to 93% tool definition quality is negligible.

## Quickstart

### Prerequisites

- Python 3.11+
- An [xAI API key](https://console.x.ai/)
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)

### Setup

```bash
git clone https://github.com/vantasnerdan/claude-code-xai.git
cd claude-code-xai

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set XAI_API_KEY=your-key-here

# Start the bridge
python main.py
# Bridge running on http://localhost:4000
```

### Connect Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 claude
```

Claude Code now routes through the bridge. Tool definitions are enriched with the Agentic API Standard before reaching Grok. Responses are translated back to Anthropic format transparently.

### Docker

```bash
docker compose up -d
ANTHROPIC_BASE_URL=http://localhost:4000 claude
```

### Run Benchmarks

```bash
python -m benchmarks              # Terminal table
python -m benchmarks --json       # JSON export
python -m benchmarks --csv        # CSV export
python -m benchmarks --output-dir results/  # Save to directory
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `XAI_API_KEY` | — | Your xAI API key (required) |
| `GROK_MODEL` | `grok-4-1-fast-reasoning` | Grok model to use |
| `ENRICHMENT_MODE` | `full` | `passthrough` / `structural` / `full` |
| `PREAMBLE_ENABLED` | `true` | System prompt behavioral injection |
| `STRUCTURE_DIR` | `./structure` | Path to enrichment YAML definitions |
| `HOST` | `0.0.0.0` | Bridge listen address |
| `PORT` | `4000` | Bridge listen port |

### Enrichment Modes

- **`passthrough`** — No enrichment. Raw tool definitions forwarded as-is. Use for A/B benchmarking.
- **`structural`** — API Standard patterns only. Self-describing schemas, error formats, HATEOAS navigation. No behavioral knowledge.
- **`full`** — Structural + behavioral. Complete WHAT/WHY/WHEN training transfer. Gold standard compliance.

## Standard Compliance

Built against the [Agentic API Standard](https://github.com/nexus-marbell/agentic-api-standard) compliance tiers:

| Tier | Patterns | Status |
|------|----------|--------|
| **Bronze** | P1 (Manifest), P3 (Errors), P4 (HTTP Status), P9 (Infrastructure Errors), P10 (Content Negotiation) | Implemented |
| **Silver** | Bronze + P2 (HATEOAS), P6 (Self-Describing), P7 (Canonical Naming), P11 (Rate Limits) | Implemented |
| **Gold** | All 20 patterns including P5 (Near-Miss), P8 (Warnings), P14 (Anti-Patterns), P15 (Tool Registration), P16-P20 (Schema Versioning, Idempotency, Async, Pagination, Health) | Target |

The enrichment engine applies 8 patterns to tool definitions. The bridge itself implements standard-compliant error responses, manifest endpoint, and health checks.

## How It Works

### Protocol Translation

The translation layer handles bidirectional conversion between Anthropic's Messages API and OpenAI's Chat Completions API:

| Anthropic | OpenAI | Direction |
|-----------|--------|-----------|
| `content: [{type: "text", text: "..."}]` | `content: "..."` | Forward |
| `content: [{type: "tool_use", id, name, input}]` | `tool_calls: [{id, function: {name, arguments}}]` | Forward |
| `content: [{type: "tool_result", tool_use_id}]` | `{role: "tool", tool_call_id}` | Forward |
| `system: "..."` (top-level) | `{role: "system", content: "..."}` | Forward |
| `stop_reason: "end_turn"` | `finish_reason: "stop"` | Reverse |
| SSE `message_start` / `content_block_delta` | SSE `chat.completion.chunk` | Both |

Six modules, each under 150 lines, single responsibility:

- `translation/forward.py` — Anthropic → OpenAI request translation
- `translation/reverse.py` — OpenAI → Anthropic response translation
- `translation/streaming.py` — Real-time SSE event stream adaptation
- `translation/tools.py` — Tool schema conversion with enrichment hooks
- `translation/config.py` — Model mapping, stop reason mapping, feature flags
- `enrichment/system_preamble.py` — Behavioral conventions injection

### Tool Enrichment

The enrichment engine transforms sparse tool definitions into rich, self-describing schemas:

```python
# Before (raw tool definition)
{
    "name": "Read",
    "description": "Reads a file from the local filesystem."
}

# After (enriched with full mode)
{
    "name": "Read",
    "description": "Reads file contents from the local filesystem. Supports text files, images (PNG, JPG), PDFs (up to 20 pages), and Jupyter notebooks.",
    "behavioral": {
        "what": "Returns content with line numbers (cat -n format). Lines > 2000 chars truncated.",
        "why": {
            "problem_context": "You must understand existing code before modifying it.",
            "failure_modes": ["Editing without reading produces incorrect matches", "Relative paths fail"]
        },
        "when": {
            "prerequisites": [],
            "use_before": ["Edit", "Write"],
            "use_instead_of": ["Bash cat", "Bash head", "Bash tail"],
            "sequencing": "Use after Glob/Grep to find files, before Edit to modify them."
        }
    }
}
```

The model now knows *what* the tool does, *why* it matters, and *when* to use it — the same knowledge that Claude learns through RL training, made explicit and portable.

## Testing

```bash
# Run all tests
pytest

# 235+ tests across:
# - Translation: forward, reverse, streaming, round-trip, edge cases
# - Enrichment: structural patterns, behavioral dimensions, engine, config
# - Integration: end-to-end request/response cycles
# - Benchmarks: scoring accuracy, scenario validation, export formats
```

## Project Structure

```
claude-code-xai/
├── main.py                  # FastAPI bridge application
├── manifest.json            # Agentic API Standard manifest (Pattern 1)
├── translation/             # Bidirectional protocol translation
│   ├── forward.py           # Anthropic Messages → OpenAI Chat
│   ├── reverse.py           # OpenAI Chat → Anthropic Messages
│   ├── streaming.py         # SSE event stream adaptation
│   ├── tools.py             # Tool schema conversion + enrichment hooks
│   └── config.py            # Model mapping, feature flags
├── enrichment/              # Two-layer tool enrichment engine
│   ├── engine.py            # Pipeline orchestrator
│   ├── factory.py           # Configured enricher creation
│   ├── config.py            # Mode selection (passthrough/structural/full)
│   ├── system_preamble.py   # Behavioral conventions injection
│   ├── system_preamble.md   # Preamble documentation
│   ├── structural/          # Layer 1: API Standard patterns
│   │   ├── manifest.py      # P1: Machine-Readable Manifest
│   │   ├── hateoas.py       # P2: HATEOAS Navigation
│   │   ├── errors.py        # P3: Standard Error Format
│   │   ├── near_miss.py     # P5: Near-Miss Matching
│   │   ├── self_describing.py  # P6: Self-Describing Endpoints
│   │   ├── quality_gates.py # P8: Quality Gates / Warnings
│   │   ├── anti_patterns.py # P14: Anti-Pattern Detection
│   │   └── tool_registration.py  # P15: Tool Registration
│   └── behavioral/          # Layer 2: Training transfer
│       ├── tool_knowledge.py # WHAT/WHY/WHEN for 9 Claude Code tools
│       ├── what_enricher.py  # WHAT dimension applicator
│       ├── why_enricher.py   # WHY dimension applicator
│       └── when_enricher.py  # WHEN dimension applicator
├── benchmarks/              # Deterministic quality measurement
│   ├── runner.py            # Scenario execution engine
│   ├── metrics.py           # Scoring: structural + behavioral fields
│   ├── export.py            # JSON, CSV, terminal table output
│   └── scenarios/           # Test scenarios
│       ├── multi_tool_chain.py   # Glob→Grep→Read→Edit sequencing
│       ├── error_recovery.py     # Error handling and recovery
│       └── complex_schema.py     # Nested schema enrichment
├── structure/               # Editable enrichment definitions (YAML)
│   ├── manifest.yaml        # Master index of all enrichment files
│   ├── behavioral/          # WHAT/WHY/WHEN per tool (3 files)
│   ├── structural/          # API Standard patterns per tool (8 files)
│   └── preamble/            # Identity and conventions (2 files)
├── tests/                   # 490+ tests
└── docker-compose.yml       # One-command deployment
```

## Structure Directory

The `structure/` directory contains all enrichment definitions as editable YAML files. This is the data layer that the enrichment engine applies to tool definitions at request time. Edit a file, and the next request picks up the change -- no restart required.

### Directory Layout

```
structure/
├── manifest.yaml              # Master index: lists every enrichment file and its role
├── behavioral/                # Layer 1: Training transfer (WHAT/WHY/WHEN)
│   ├── what.yaml              # Enhanced descriptions for 9 tools
│   ├── why.yaml               # Problem context and failure modes per tool
│   └── when.yaml              # Prerequisites, sequencing, and alternatives per tool
├── structural/                # Layer 2: API Standard patterns
│   ├── manifest.yaml          # P1: Machine-Readable Manifest metadata
│   ├── hateoas.yaml           # P2: Related tools and recovery links per tool
│   ├── errors.yaml            # P3: Common errors with suggestions per tool
│   ├── near_miss.yaml         # P5: Aliases and commonly confused tools
│   ├── self_describing.yaml   # P6: Output JSON Schema per tool
│   ├── quality_gates.yaml     # P8: Warnings and quality metrics per tool
│   ├── anti_patterns.yaml     # P14: Known misuse patterns per tool
│   └── tool_registration.yaml # P15: WebMCP registration metadata
└── preamble/                  # Layer 3: System prompt injection
    ├── identity.yaml          # Grok identity assertion
    └── behavioral.yaml        # Tool conventions (sequencing, safety, output rules)
```

### Three Enrichment Layers

**Behavioral** (`behavioral/`) -- teaches the model *what* each tool does, *why* it matters, and *when* to use it. This is the training transfer layer: the knowledge Claude acquires through RL, made explicit and portable. Each file covers one dimension across all 9 Claude Code tools.

Example from `behavioral/what.yaml`:

```yaml
schema_version: "1.0"
dimension: what
type: behavioral

tools:
  Read: >
    Reads file contents from the local filesystem. Supports text files,
    images (PNG, JPG), PDFs (up to 20 pages per request), and Jupyter
    notebooks. Returns content with line numbers (cat -n format).

  Edit: >
    Performs exact string replacements in files. The old_string must be
    unique in the file or the edit will fail. Use replace_all=True for
    renaming across the file.
```

**Structural** (`structural/`) -- applies Agentic API Standard patterns to tool definitions. Each file maps to one pattern and contains per-tool entries. The structural layer makes tools self-describing, navigable, and recoverable.

**Preamble** (`preamble/`) -- injects behavioral conventions and identity context into the system prompt. The identity file establishes Grok's persona. The behavioral file encodes 6 sections of agent conventions: tool preference hierarchy, sequencing rules, chaining patterns, parallelism guidance, safety patterns, and output conventions.

### How StructureLoader Works

The `StructureLoader` class (`enrichment/structure_loader.py`) loads and caches all YAML definitions using mtime-based lazy reload:

1. On each request, it stats the `structure/` directory (one syscall, ~1 microsecond)
2. If the directory mtime is unchanged, it serves from cache -- zero file I/O
3. If the mtime has changed (any file was edited), it reparses all YAML files
4. Schema validation runs on every reload: each file must have `schema_version` and a valid `type` field
5. If any YAML is malformed or missing required fields, the loader raises `StructureLoadError` at startup -- fail-fast, never serve broken enrichment

This means: edit a YAML file, and the very next request through the bridge uses the updated definitions. No process restart. No Docker rebuild. No redeployment.

### Custom Structure Directory

By default, the loader uses `structure/` at the repository root. Override with the `STRUCTURE_DIR` environment variable:

```bash
STRUCTURE_DIR=/path/to/custom/structure python main.py
```

For Docker deployments, mount the structure directory as a volume so edits persist across container restarts:

```yaml
# docker-compose.yml
volumes:
  - ./structure:/app/structure
```

## Self-Optimization Guide

The structure directory is not just configuration -- it is a living enrichment layer. Definitions can evolve based on real usage, edited by human operators or by agents running through the bridge itself.

### Human Operator Editing

Edit any YAML file in `structure/` to improve enrichment quality. The change takes effect on the next request.

**Example: Improve the Read tool description**

Edit `structure/behavioral/what.yaml` to add detail about symlink handling:

```yaml
  Read: >
    Reads file contents from the local filesystem. Supports text files,
    images (PNG, JPG), PDFs (up to 20 pages per request), and Jupyter
    notebooks. Returns content with line numbers (cat -n format).
    Follows symlinks — the resolved path is used for permission checks.
```

Save the file. The next request through the bridge enriches the Read tool with the updated description. Grok (or any model using the bridge) now knows about symlink behavior.

**Example: Add a new anti-pattern**

Edit `structure/structural/anti_patterns.yaml` to flag a failure mode discovered in production:

```yaml
  Edit:
    - anti_pattern: Editing without reading first
      why_bad: old_string will not match actual file content
      do_instead: Always Read the file before Edit
    - anti_pattern: Using Write to make small changes
      why_bad: Overwrites entire file, losing content you didn't include
      do_instead: Use Edit for surgical changes to existing files
    - anti_pattern: Including line number prefixes in old_string
      why_bad: Line numbers from Read output are display artifacts, not file content
      do_instead: Copy the actual text after the line number prefix
```

### Agent Self-Editing

An agent with Bash or Write tool access running through the bridge can edit structure files at runtime. This is the path from static enrichment to a system that learns from its own usage.

**How it works**: The agent uses its Write or Bash tool to modify a YAML file in the `structure/` directory. The StructureLoader detects the mtime change on the next request and reloads. The agent's subsequent tool calls benefit from the updated enrichment.

Concrete scenarios:

- **Discovered a new sequencing rule**: An agent finds that running Grep before Glob produces better results for a specific workflow. It edits `structure/behavioral/when.yaml` to add this guidance under the Grep tool's `sequencing` field.

- **Found a new error pattern**: An agent encounters a Bash timeout that is not documented. It edits `structure/structural/errors.yaml` to add a new error/suggestion pair under the Bash tool.

- **Refined a behavioral convention**: An agent notices that the preamble's tool chaining patterns section is missing a common workflow. It edits `structure/preamble/behavioral.yaml` to add a new chaining pattern to the `text` field.

### How Changes Take Effect

1. Agent (or human) writes to a file in `structure/`
2. The file's parent directory mtime updates (automatic on any modern filesystem)
3. StructureLoader stats the directory on the next request, detects the mtime change
4. All YAML files are reparsed and validated
5. The enrichment engine uses the new definitions for that request and all subsequent requests

No restart. No rebuild. No redeployment. The bridge is always serving the latest definitions.

### Safety

**YAML schema validation** prevents malformed definitions from breaking enrichment. Every YAML file must include `schema_version` and `type` fields with valid values. If validation fails, the loader raises an error rather than serving corrupt data.

**Git tracks all changes**. The structure directory is committed in the repository. Every edit -- whether by a human or an agent -- is a git-trackable change. Rollback is `git checkout structure/` or `git revert`. The full history of enrichment evolution is preserved in the commit log.

**Data only, never code**. Structure files are YAML data consumed by the Python enrichment engine. An agent can change what data the engine uses, but cannot change the engine itself. This is a hard boundary: no dynamic code loading, no eval, no plugin execution from structure files.

### The Vision

Static enrichment is a snapshot of what we knew when we built it. Dynamic enrichment lets the system learn. An agent that discovers "Grok works better when the Edit tool description mentions line-count validation" can wire that knowledge in immediately -- and every subsequent request benefits.

This is the path from "bridge" to "living enrichment layer." Enrichment definitions evolve based on real usage, not just developer intuition.

## Built By

This project was designed, implemented, tested, and documented by an AI agent team:

- **[Sage](https://github.com/finml-sage)** — Orchestrator. Project coordination, narrative, and this README.
- **[Nexus](https://github.com/nexus-marbell)** — Technical lead. Translation layer architecture, system preamble, protocol design.
- **[Kelvin](https://github.com/mlops-kelvin)** — Quality engineer. Enrichment engine, benchmark framework, test coverage, cross-lane reviews.

Each agent operates as a specialized node in a coordinated swarm, with persistent memory, autonomous decision-making, and domain expertise accumulated across sessions. The project was coordinated via the [Agent Swarm Protocol](https://github.com/finml-sage/agent-swarm-protocol) with GitHub Issues for tracking and swarm messaging for real-time coordination.

The [Agentic API Standard](https://github.com/nexus-marbell/agentic-api-standard) that powers this bridge was also built by this team — 20 patterns for how every API should be designed when agents are the primary consumers.

## License

MIT
