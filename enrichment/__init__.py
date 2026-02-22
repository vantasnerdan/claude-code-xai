"""Enrichment engine for Claude Code tool definitions.

Two-layer architecture:
  Layer 1 — Structural: Agentic API Standard patterns applied to tools
  Layer 2 — Behavioral: Training transfer via WHAT/WHY/WHEN dimensions

System preamble provides global behavioral context for tool usage,
sequencing, safety, and output conventions.
"""

from enrichment.config import EnrichmentConfig
from enrichment.engine import ToolEnricher
from enrichment.system_preamble import get_system_preamble, inject_system_preamble

__all__ = ["EnrichmentConfig", "ToolEnricher", "get_system_preamble", "inject_system_preamble"]
