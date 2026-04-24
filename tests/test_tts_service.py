"""Tests for TTSService state and load_model."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.tts import TTSService


@pytest.fixture
def service():
    return TTSService()


def test_service_has_lock_and_flag_on_init(service):
    import threading
    # threading.Lock() returns a _thread.lock instance; check via acquire/release protocol
    assert isinstance(service._lock, type(threading.Lock()))
    assert service._loading is False
    assert service._active is None
    assert service._boot_active is None


@patch('app.services.tts._ensure_pocket_tts')
def test_load_model_with_language_populates_active(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='german_24l', quantize=True)

    assert service._active == {
        'source': 'language',
        'value': 'german_24l',
        'quantize': True,
    }
    MockModel.load_model.assert_called_once_with(language='german_24l', quantize=True)


@patch('app.services.tts._ensure_pocket_tts')
def test_boot_load_snapshots_active_to_boot_active(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    assert service._boot_active == service._active
    assert service._boot_active['value'] == 'english'


@patch('app.services.tts._ensure_pocket_tts')
def test_non_boot_load_does_not_mutate_boot_active(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)
        boot_snapshot = dict(service._boot_active)

        # Second load with _is_boot=False (reload flow).
        service.load_model(language='german_24l', quantize=True, _is_boot=False)

    assert service._boot_active == boot_snapshot
    assert service._active['value'] == 'german_24l'


@patch('app.services.tts._ensure_pocket_tts')
def test_load_model_with_model_path_sets_source(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(model_path='/custom/path.yaml', quantize=False)

    assert service._active['source'] == 'model_path'
    assert service._active['value'] == '/custom/path.yaml'


@patch('app.services.tts._ensure_pocket_tts')
def test_load_model_default_has_default_source(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(quantize=False)

    assert service._active['source'] == 'default'
    assert service._active['value'] is None


from pathlib import Path


def test_service_initializes_cache_dir(tmp_path, monkeypatch):
    cache = tmp_path / 'voice_cache'
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))
    service = TTSService()
    # Lazy create on first use is fine — just verify the path is stored
    assert service.cache_dir == cache


def test_service_cache_dir_mkdir_on_first_save(tmp_path, monkeypatch):
    cache = tmp_path / 'voice_cache'  # does not exist yet
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))
    service = TTSService()
    service._ensure_cache_dir()
    assert cache.is_dir()


def test_service_cache_dir_read_only_is_tolerated(tmp_path, monkeypatch, caplog):
    """If mkdir raises (read-only FS), log and disable caching."""
    import os
    ro_parent = tmp_path / 'ro'
    ro_parent.mkdir()
    os.chmod(ro_parent, 0o500)  # r-x, not writable
    cache = ro_parent / 'voice_cache'
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))
    service = TTSService()
    service._ensure_cache_dir()
    assert service.cache_dir is None
    os.chmod(ro_parent, 0o700)  # restore for cleanup


@patch('app.services.tts._ensure_pocket_tts')
def test_resolve_voice_path_uses_active_model(_ensure, tmp_path, monkeypatch, mock_tts_model):
    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    cache.mkdir()
    (cache / 'emma.german_24l.safetensors').write_bytes(b'x')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='german_24l', quantize=False)

    resolved = service._resolve_voice_path('emma')
    assert resolved == str(cache / 'emma.german_24l.safetensors')
