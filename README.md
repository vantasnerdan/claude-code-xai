# claude-code-xia

## xai-grok-claude-code-bridge

xAI/Grok + full native Claude Code CLI + **your Agentic API Standard (Gold tier)**.

- 100% native CC (all internal tools, MCPs, skills, agent teams)
- Grok tool calling improved by your 20 patterns (rich schemas, anti-patterns, recovery links)
- Self-describing Gold manifest + standardized errors + HATEOAS everywhere
- Built-in Prometheus metrics to prove the uplift

## Quickstart
See above.

## Architecture
Claude Code → ANTHROPIC_BASE_URL → this bridge (enrich + translate) → https://api.x.ai/v1 → Grok → back to CC.

Your patterns are applied in `agentic_enricher.py` before every Grok call.

License: MIT
