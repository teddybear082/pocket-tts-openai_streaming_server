"""Tests for version helpers."""

from unittest.mock import patch

from app.services.versions import get_versions


def test_get_versions_returns_both():
    result = get_versions()
    assert 'server' in result
    assert 'pocket_tts' in result


def test_get_versions_handles_missing_package():
    """When importlib metadata is unavailable, server falls back to the
    hardcoded `app.__version__` constant; pocket_tts has no such fallback
    and reports 'unknown'."""
    from importlib.metadata import PackageNotFoundError

    from app import __version__ as server_fallback

    get_versions.cache_clear()
    with patch('app.services.versions.version', side_effect=PackageNotFoundError):
        result = get_versions()
    assert result == {'server': server_fallback, 'pocket_tts': 'unknown'}
