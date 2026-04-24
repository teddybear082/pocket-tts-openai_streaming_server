"""Tests for voice cache filename parsing and resolution."""

from pathlib import Path

import pytest

from app.services.voice_cache import (
    active_model_tag,
    known_model_tags,
    parse_safetensors_name,
    resolve_voice_path,
)


def test_active_model_tag_returns_raw_when_not_aliased():
    assert active_model_tag('german_24l') == 'german_24l'


def test_active_model_tag_canonicalizes_english_alias():
    assert active_model_tag('english') == 'english_2026-04'


def test_active_model_tag_canonicalizes_english_2026_01():
    assert active_model_tag('english_2026-01') == 'english_2026-04'


def test_known_model_tags_includes_all_supported_plus_alias_targets():
    tags = known_model_tags()
    assert 'english_2026-04' in tags
    assert 'german_24l' in tags
    assert 'french_24l' in tags
    # Alias keys are also included so users can name files with them.
    assert 'english' in tags


def test_parse_safetensors_name_recognizes_tagged_file():
    tags = known_model_tags()
    stem, tag = parse_safetensors_name('Emma Watson.english_2026-04.safetensors', tags)
    assert stem == 'Emma Watson'
    assert tag == 'english_2026-04'


def test_parse_safetensors_name_legacy_unlabeled():
    tags = known_model_tags()
    stem, tag = parse_safetensors_name('legacy.safetensors', tags)
    assert stem == 'legacy'
    assert tag is None


def test_parse_safetensors_name_dot_in_stem_but_unknown_tag():
    """A filename like 'my.voice.safetensors' has 'voice' as the final segment —
    but 'voice' isn't a known tag, so we treat the whole thing as the stem."""
    tags = known_model_tags()
    stem, tag = parse_safetensors_name('my.voice.safetensors', tags)
    assert stem == 'my.voice'
    assert tag is None


def test_resolve_voice_path_prefers_tagged_cache(tmp_voices, tmp_cache):
    """Preference: cache_dir tagged > voices_dir tagged > raw audio > legacy."""
    (tmp_voices / 'emma.wav').write_bytes(b'fake-audio')
    (tmp_cache / 'emma.english_2026-04.safetensors').write_bytes(b'fake-st')

    result = resolve_voice_path(
        'emma', active_model='english_2026-04',
        voices_dir=tmp_voices, cache_dir=tmp_cache,
    )
    assert result == tmp_cache / 'emma.english_2026-04.safetensors'


def test_resolve_voice_path_falls_back_to_raw_audio(tmp_voices, tmp_cache):
    (tmp_voices / 'emma.wav').write_bytes(b'fake-audio')
    result = resolve_voice_path(
        'emma', active_model='english_2026-04',
        voices_dir=tmp_voices, cache_dir=tmp_cache,
    )
    assert result == tmp_voices / 'emma.wav'


def test_resolve_voice_path_legacy_unlabeled(tmp_voices, tmp_cache):
    (tmp_voices / 'emma.safetensors').write_bytes(b'fake-st')
    result = resolve_voice_path(
        'emma', active_model='english_2026-04',
        voices_dir=tmp_voices, cache_dir=tmp_cache,
    )
    assert result == tmp_voices / 'emma.safetensors'


def test_resolve_voice_path_passthrough_for_builtin_name(tmp_voices, tmp_cache):
    """Built-in names (no matching file anywhere) pass through untouched —
    pocket-tts handles them via HuggingFace."""
    result = resolve_voice_path(
        'alba', active_model='english_2026-04',
        voices_dir=tmp_voices, cache_dir=tmp_cache,
    )
    assert result == 'alba'


def test_resolve_voice_path_respects_alias(tmp_voices, tmp_cache):
    """Asking for 'english' should find a cache tagged 'english_2026-04'."""
    (tmp_cache / 'emma.english_2026-04.safetensors').write_bytes(b'fake-st')
    result = resolve_voice_path(
        'emma', active_model='english',  # alias
        voices_dir=tmp_voices, cache_dir=tmp_cache,
    )
    assert result == tmp_cache / 'emma.english_2026-04.safetensors'


def test_resolve_voice_path_voices_dir_tagged_cache(tmp_voices, tmp_cache):
    """A tagged cache dropped directly into voices_dir (e.g. by WingmanAI) is honored."""
    (tmp_voices / 'emma.german_24l.safetensors').write_bytes(b'fake-st')
    result = resolve_voice_path(
        'emma', active_model='german_24l',
        voices_dir=tmp_voices, cache_dir=tmp_cache,
    )
    assert result == tmp_voices / 'emma.german_24l.safetensors'
