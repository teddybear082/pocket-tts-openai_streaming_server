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


@patch('app.services.tts._ensure_pocket_tts')
def test_resolve_voice_path_accepts_filename_with_extension(
    _ensure, tmp_path, monkeypatch, mock_tts_model
):
    """Backwards-compat: passing 'emma.wav' (with extension) should resolve to
    the existing file, not get joined with another extension."""
    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    cache.mkdir()
    (voices / 'emma.wav').write_bytes(b'fake-audio')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    resolved = service._resolve_voice_path('emma.wav')
    assert resolved == str(voices / 'emma.wav')


@patch('app.services.tts._ensure_pocket_tts')
def test_reload_model_swaps_and_clears_voice_cache(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)
        service.voice_cache['stale'] = {'fake': 'state'}

        new_model = MagicMock()
        new_model.sample_rate = 24000
        new_model.device = 'cpu'
        MockModel.load_model.return_value = new_model
        service.reload_model(language='german_24l', quantize=True)

    assert service.model is new_model
    assert service.voice_cache == {}
    assert service._active['value'] == 'german_24l'
    assert service._active['quantize'] is True
    assert service._boot_active['value'] == 'english'  # unchanged


@patch('app.services.tts._ensure_pocket_tts')
def test_reload_model_rejects_if_boot_used_model_path(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(model_path='/custom/x.yaml', quantize=False)

    with pytest.raises(RuntimeError, match='model_path'):
        service.reload_model(language='german_24l', quantize=False)


@patch('app.services.tts._ensure_pocket_tts')
def test_reload_model_rejects_unknown_language(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    with pytest.raises(ValueError, match='klingon'):
        service.reload_model(language='klingon', quantize=False)


@patch('app.services.tts._ensure_pocket_tts')
def test_reload_model_rejects_if_already_loading(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    service._loading = True
    with pytest.raises(RuntimeError, match='already loading'):
        service.reload_model(language='german_24l', quantize=False)


@patch('app.services.tts._ensure_pocket_tts')
def test_reload_model_async_returns_false_when_already_loading(_ensure, service, mock_tts_model):
    """Async API uses an atomic check-and-set so concurrent callers can't both
    win the claim. The loser gets False instead of an exception so the route
    can map it directly to 409."""
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    service._loading = True
    started = service.reload_model_async(language='german_24l', quantize=False)
    assert started is False


@patch('app.services.tts._ensure_pocket_tts')
def test_reload_model_async_validation_still_raises(_ensure, service, mock_tts_model):
    """Validation errors are still raised synchronously so the route can
    return 400/403 instead of 409 — only the in-progress check is silent."""
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    with pytest.raises(ValueError, match='klingon'):
        service.reload_model_async(language='klingon', quantize=False)


@patch('app.services.tts._ensure_pocket_tts')
def test_reload_model_restores_previous_on_failure(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)
        original_model = service.model

        MockModel.load_model.side_effect = RuntimeError('weights corrupted')
        with pytest.raises(RuntimeError, match='weights corrupted'):
            service.reload_model(language='german_24l', quantize=False)

    assert service.model is original_model
    assert service._active['value'] == 'english'
    assert service._loading is False


@patch('app.services.tts._ensure_pocket_tts')
def test_generate_audio_raises_when_loading(_ensure, service, mock_tts_model):
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)
    service._loading = True
    with pytest.raises(RuntimeError, match='model reloading'):
        service.generate_audio(voice_state={}, text='hi')


@patch('app.services.tts._ensure_pocket_tts')
def test_get_voice_state_saves_clone_to_cache(_ensure, tmp_path, monkeypatch, mock_tts_model):
    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    (voices / 'emma.wav').write_bytes(b'fake-audio')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    mock_tts_model.get_state_for_audio_prompt.return_value = {'fake': 'state'}
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    with patch('app.services.tts.export_model_state') as mock_export:
        state = service.get_voice_state('emma')

    assert state == {'fake': 'state'}
    mock_export.assert_called_once()
    # Called with state and expected cache path
    args = mock_export.call_args.args
    assert args[0] == {'fake': 'state'}
    assert str(args[1]).endswith('emma.english_2026-04.safetensors')
    assert (cache).is_dir()


@patch('app.services.tts._ensure_pocket_tts')
def test_get_voice_state_saves_clone_under_custom_model_path(
    _ensure, tmp_path, monkeypatch, mock_tts_model
):
    """Issue #13: when started with --model-path "<full file path>", the cache
    filename must not include the raw path. Otherwise safetensors raises a
    serialization I/O error on Windows because `:` and `\\` are illegal in
    filenames. The cache filename must be safe and the call must succeed."""
    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    (voices / 'Aadi.wav').write_bytes(b'fake-audio')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    mock_tts_model.get_state_for_audio_prompt.return_value = {'fake': 'state'}
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        # Simulate the user starting the server with --model-path on Windows.
        service.load_model(
            model_path=r'C:\PocketTTS-Server\model\languages\english\english.yaml',
            quantize=False,
        )

    with patch('app.services.tts.export_model_state') as mock_export:
        state = service.get_voice_state('Aadi')

    assert state == {'fake': 'state'}
    mock_export.assert_called_once()
    target_path = mock_export.call_args.args[1]
    name = target_path.name if hasattr(target_path, 'name') else str(target_path).rsplit('/', 1)[-1]
    # The filename must contain no characters disallowed in Windows filenames.
    assert not any(c in name for c in r'\/:*?"<>|'), (
        f'cache filename {name!r} contains path-illegal chars'
    )
    # Stem prefix preserved; suffix is .safetensors.
    assert name.startswith('Aadi.')
    assert name.endswith('.safetensors')


@patch('app.services.tts._ensure_pocket_tts')
def test_get_voice_state_tolerates_safetensor_serialize_error(
    _ensure, tmp_path, monkeypatch, mock_tts_model
):
    """Issue #13: SafetensorError does not inherit from OSError, so a serialize
    failure used to bubble up and break voice loading entirely. Cache writes
    are best-effort: any failure must be logged and the voice must still load."""
    from safetensors import SafetensorError

    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    (voices / 'emma.wav').write_bytes(b'fake-audio')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    mock_tts_model.get_state_for_audio_prompt.return_value = {'fake': 'state'}
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    with patch('app.services.tts.export_model_state') as mock_export:
        mock_export.side_effect = SafetensorError('I/O error: simulated failure')
        # Must NOT raise — caching is best-effort.
        state = service.get_voice_state('emma')

    assert state == {'fake': 'state'}


@patch('app.services.tts._ensure_pocket_tts')
def test_save_cloned_state_failure_logs_traceback(
    _ensure, tmp_path, monkeypatch, mock_tts_model, caplog
):
    """Cache write failures must include the traceback in logs (via exc_info)
    so the underlying cause stays diagnosable even though the warning is swallowed."""
    import logging

    from safetensors import SafetensorError

    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    (voices / 'emma.wav').write_bytes(b'fake-audio')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    mock_tts_model.get_state_for_audio_prompt.return_value = {'fake': 'state'}
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    with patch('app.services.tts.export_model_state') as mock_export:
        mock_export.side_effect = SafetensorError('I/O error: simulated failure')
        with caplog.at_level(logging.WARNING, logger='PocketTTS.tts'):
            service.get_voice_state('emma')

    cache_warnings = [r for r in caplog.records if 'Could not save voice cache' in r.message]
    assert cache_warnings, 'expected a warning about the failed cache save'
    # exc_info must be set so the traceback is captured.
    assert cache_warnings[0].exc_info is not None


@patch('app.services.tts._ensure_pocket_tts')
def test_get_voice_state_regenerates_when_source_newer(
    _ensure, tmp_path, monkeypatch, mock_tts_model
):
    import os
    import time

    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    cache.mkdir()
    stale = cache / 'emma.english_2026-04.safetensors'
    stale.write_bytes(b'old')
    os.utime(stale, (time.time() - 100, time.time() - 100))
    (voices / 'emma.wav').write_bytes(b'new-audio')  # mtime = now
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    mock_tts_model.get_state_for_audio_prompt.return_value = {'fresh': 'state'}
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    with patch('app.services.tts.export_model_state') as mock_export:
        state = service.get_voice_state('emma')

    assert state == {'fresh': 'state'}
    # Should have called get_state_for_audio_prompt with the .wav file, not the stale cache
    call_arg = mock_tts_model.get_state_for_audio_prompt.call_args.args[0]
    assert str(call_arg).endswith('emma.wav')
    mock_export.assert_called_once()


@patch('app.services.tts._ensure_pocket_tts')
def test_get_voice_state_stem_with_dot_regenerates_correctly(
    _ensure, tmp_path, monkeypatch, mock_tts_model
):
    """A voice stem containing dots (e.g. 'John.Doe') must be parsed via the
    known_model_tags helper so the staleness check finds the right source
    file. Naive split('.', 1)[0] would yield 'John' and miss 'John.Doe.wav'."""
    import os
    import time

    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    cache.mkdir()
    stale = cache / 'John.Doe.english_2026-04.safetensors'
    stale.write_bytes(b'old')
    os.utime(stale, (time.time() - 100, time.time() - 100))
    (voices / 'John.Doe.wav').write_bytes(b'new-audio')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    mock_tts_model.get_state_for_audio_prompt.return_value = {'fresh': 'state'}
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    with patch('app.services.tts.export_model_state'):
        service.get_voice_state('John.Doe')

    call_arg = mock_tts_model.get_state_for_audio_prompt.call_args.args[0]
    assert str(call_arg).endswith('John.Doe.wav')


@patch('app.services.tts._ensure_pocket_tts')
def test_list_voices_collapses_per_stem(_ensure, tmp_path, monkeypatch, mock_tts_model):
    voices = tmp_path / 'voices'
    voices.mkdir()
    cache = tmp_path / 'voice_cache'
    cache.mkdir()
    (voices / 'emma.wav').write_bytes(b'a')
    (voices / 'emma.safetensors').write_bytes(b'a')
    (cache / 'emma.english_2026-04.safetensors').write_bytes(b'a')
    (cache / 'emma.german_24l.safetensors').write_bytes(b'a')
    (voices / 'morgan.mp3').write_bytes(b'a')
    monkeypatch.setattr('app.config.Config.VOICE_CACHE_DIR', str(cache))

    service = TTSService()
    service.set_voices_dir(str(voices))
    with patch('app.services.tts.TTSModel') as MockModel:
        MockModel.load_model.return_value = mock_tts_model
        service.load_model(language='english', quantize=False)

    voices_list = service.list_voices()
    custom_ids = [v['id'] for v in voices_list if v['type'] == 'custom']
    assert custom_ids == ['emma', 'morgan']
