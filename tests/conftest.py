"""Shared pytest fixtures."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_voices(tmp_path: Path) -> Path:
    """Temporary voices directory (simulates user-provided audio)."""
    d = tmp_path / 'voices'
    d.mkdir()
    return d


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """Temporary voice cache directory (simulates writable cache volume)."""
    d = tmp_path / 'voice_cache'
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def reset_tts_singleton():
    """Reset the TTSService singleton between tests."""
    import app.services.tts as tts_module
    tts_module._tts_service = None
    yield
    tts_module._tts_service = None


@pytest.fixture
def mock_tts_model():
    """Mock pocket-tts TTSModel for unit tests."""
    model = MagicMock()
    model.sample_rate = 24000
    model.device = 'cpu'
    return model
