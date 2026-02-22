"""Tests for EnrichmentConfig."""
import pytest

from enrichment.config import EnrichmentConfig


class TestEnrichmentConfig:
    """Tests for the EnrichmentConfig frozen dataclass."""

    def test_default_config_is_full_mode(self) -> None:
        """Default configuration is full mode with all enrichment layers."""
        config = EnrichmentConfig()
        assert config.mode == "full"
        assert config.include_behavioral is True
        assert config.is_passthrough is False
        assert config.enable_what is True
        assert config.enable_why is True
        assert config.enable_when is True

    def test_passthrough_mode(self) -> None:
        """Passthrough mode disables all enrichment."""
        config = EnrichmentConfig(mode="passthrough")
        assert config.is_passthrough is True
        assert config.include_behavioral is False

    def test_structural_mode_excludes_behavioral(self) -> None:
        """Structural mode enables patterns but excludes behavioral enrichment."""
        config = EnrichmentConfig(mode="structural")
        assert config.is_passthrough is False
        assert config.include_behavioral is False

    def test_enabled_structural_patterns(self) -> None:
        """Default structural patterns include all 8 implemented patterns."""
        config = EnrichmentConfig()
        expected = frozenset({1, 2, 3, 5, 6, 8, 14, 15})
        assert config.enabled_structural == expected

    def test_config_is_frozen(self) -> None:
        """Config is immutable — attributes cannot be changed after creation."""
        config = EnrichmentConfig()
        with pytest.raises(AttributeError):
            config.mode = "passthrough"  # type: ignore[misc]

    def test_custom_structural_patterns(self) -> None:
        """Can specify a custom set of structural patterns."""
        config = EnrichmentConfig(enabled_structural=frozenset({6}))
        assert config.enabled_structural == frozenset({6})

    def test_selective_behavioral_dimensions(self) -> None:
        """Can enable/disable individual behavioral dimensions."""
        config = EnrichmentConfig(
            enable_what=True,
            enable_why=False,
            enable_when=True,
        )
        assert config.enable_what is True
        assert config.enable_why is False
        assert config.enable_when is True

    def test_full_mode_includes_behavioral(self) -> None:
        """Full mode reports include_behavioral as True."""
        config = EnrichmentConfig(mode="full")
        assert config.include_behavioral is True
