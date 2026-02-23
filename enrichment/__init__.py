"""Enrichment engine for Claude Code tool definitions.

Two-layer architecture:
  Layer 1 — Structural: Agentic API Standard patterns applied to tools
  Layer 2 — Behavioral: Training transfer via WHAT/WHY/WHEN dimensions

System preamble provides global behavioral context for tool usage,
sequencing, safety, and output conventions.

Structure definitions are loaded from YAML files in the structure/ directory
via StructureLoader with lazy mtime-based reload.
"""

from enrichment.config import EnrichmentConfig
from enrichment.engine import ToolEnricher
from enrichment.structure_loader import StructureLoader, StructureLoadError
from enrichment.system_preamble import get_system_preamble, inject_system_preamble

__all__ = [
    "EnrichmentConfig",
    "ToolEnricher",
    "StructureLoader",
    "StructureLoadError",
    "get_system_preamble",
    "inject_system_preamble",
]
