"""Tests for app.config constants."""

from pathlib import Path

from app.config import Config


def test_supported_languages_contains_all_yaml_configs():
    """All 12 language YAMLs from pocket-tts v2.0.0 must be listed."""
    expected = {
        'english', 'english_2026-01', 'english_2026-04',
        'french_24l',
        'german', 'german_24l',
        'italian', 'italian_24l',
        'portuguese', 'portuguese_24l',
        'spanish', 'spanish_24l',
    }
    assert set(Config.SUPPORTED_LANGUAGES) == expected


def test_legacy_aliases_canonicalize_english_variants():
    """english and english_2026-01 both resolve to english_2026-04 for cache tagging."""
    assert Config.LEGACY_MODEL_ALIASES == {
        'english': 'english_2026-04',
        'english_2026-01': 'english_2026-04',
    }


def test_alias_targets_are_supported_languages():
    """Every alias target must itself be a valid supported language."""
    for target in Config.LEGACY_MODEL_ALIASES.values():
        assert target in Config.SUPPORTED_LANGUAGES


def test_voice_cache_dir_defaults_to_base_path_subdir():
    """Default cache dir sits under BASE_PATH."""
    cache_dir = Path(Config.VOICE_CACHE_DIR)
    assert cache_dir.name == 'voice_cache'
