"""Integration test: enrichment engine wired into the translation pipeline."""
import pytest
from enrichment.factory import create_enricher
from translation.tools import set_tool_enrichment_hook, translate_tools


class TestEnrichmentIntegration:
    """Verify enrichment engine integrates with translation layer."""

    def setup_method(self):
        """Reset the enrichment hook before each test."""
        set_tool_enrichment_hook(None)

    def test_enricher_wires_into_translation_hook(self):
        """ToolEnricher.enrich can be used as the enrichment hook."""
        enricher = create_enricher(mode="full")
        set_tool_enrichment_hook(enricher.enrich)

        tools = [{"name": "Read", "description": "Reads files", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]
        result = translate_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "Read"

    def test_passthrough_mode_does_not_modify_tools(self):
        """In passthrough mode, tools pass through unchanged."""
        enricher = create_enricher(mode="passthrough")
        set_tool_enrichment_hook(enricher.enrich)

        tools = [{"name": "Read", "description": "Reads files", "input_schema": {"type": "object"}}]
        result = translate_tools(tools)

        assert result[0]["function"]["description"] == "Reads files"

    def test_full_mode_enriches_tool_definition(self):
        """In full mode, tool definitions are enhanced with behavioral knowledge."""
        enricher = create_enricher(mode="full")

        tools = [{"name": "Read", "description": "Reads files", "input_schema": {"type": "object"}}]
        result = enricher.enrich(tools)

        # Full mode should add behavioral knowledge keys
        assert "behavioral_what" in result[0]
        assert "behavioral_why" in result[0]
        assert "behavioral_when" in result[0]
        # And structural metadata
        assert "_manifest" in result[0]

    def test_structural_mode_adds_patterns_without_behavioral(self):
        """Structural mode applies API standard patterns but not behavioral enrichment."""
        enricher = create_enricher(mode="structural")
        set_tool_enrichment_hook(enricher.enrich)

        tools = [{"name": "Read", "description": "Reads files", "input_schema": {"type": "object"}}]

        # Verify structural data is applied at enrichment level
        enriched = enricher.enrich(tools)
        assert "_manifest" in enriched[0]
        assert "behavioral_what" not in enriched[0]

        # Verify translation still works after enrichment
        result = translate_tools(tools)
        assert result[0]["type"] == "function"

    def test_factory_reads_environment_variable(self, monkeypatch):
        """Factory defaults to ENRICHMENT_MODE env var."""
        monkeypatch.setenv("ENRICHMENT_MODE", "passthrough")
        enricher = create_enricher()
        assert enricher.config.is_passthrough

    def test_factory_defaults_to_full(self, monkeypatch):
        """Without env var, factory defaults to full mode."""
        monkeypatch.delenv("ENRICHMENT_MODE", raising=False)
        enricher = create_enricher()
        assert enricher.config.mode == "full"

    def test_three_modes_produce_different_results(self):
        """Passthrough, structural, and full produce measurably different enrichment."""
        tools = [{"name": "Edit", "description": "Edits files", "input_schema": {"type": "object"}}]

        passthrough = create_enricher(mode="passthrough").enrich(tools)
        structural = create_enricher(mode="structural").enrich(tools)
        full = create_enricher(mode="full").enrich(tools)

        # Passthrough should be identical to input
        assert passthrough == tools
        # Structural and full should be different from passthrough
        assert structural != passthrough
        # Full should have behavioral data that structural doesn't
        assert full != structural

    def test_enrichment_does_not_mutate_input(self):
        """Enrichment must not modify the original tool list."""
        tools = [{"name": "Read", "description": "Reads files", "input_schema": {"type": "object"}}]
        original = [dict(t) for t in tools]

        enricher = create_enricher(mode="full")
        enricher.enrich(tools)

        assert tools == original

    def test_translation_after_enrichment_preserves_core_fields(self):
        """Core fields (name, description, parameters) survive enrichment + translation."""
        enricher = create_enricher(mode="full")
        set_tool_enrichment_hook(enricher.enrich)

        tools = [
            {"name": "Bash", "description": "Executes commands", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ]
        result = translate_tools(tools)

        func = result[0]["function"]
        assert func["name"] == "Bash"
        assert func["description"] == "Executes commands"
        assert func["parameters"]["required"] == ["command"]

    def test_multiple_tools_all_enriched(self):
        """All tools in a batch receive enrichment, not just the first."""
        enricher = create_enricher(mode="full")

        tools = [
            {"name": "Read", "description": "Read", "input_schema": {"type": "object"}},
            {"name": "Edit", "description": "Edit", "input_schema": {"type": "object"}},
            {"name": "Bash", "description": "Bash", "input_schema": {"type": "object"}},
        ]
        result = enricher.enrich(tools)

        assert len(result) == 3
        for tool in result:
            assert "_manifest" in tool
