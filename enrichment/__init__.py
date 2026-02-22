"""Enrichment engine for Claude Code tool definitions.

Two-layer architecture:
  Layer 1 — Structural: Agentic API Standard patterns applied to tools
  Layer 2 — Behavioral: Training transfer via WHAT/WHY/WHEN dimensions
"""

from enrichment.config import EnrichmentConfig
from enrichment.engine import ToolEnricher

__all__ = ["EnrichmentConfig", "ToolEnricher"]
