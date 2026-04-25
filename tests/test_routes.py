"""Tests for /v1/model routes."""

from unittest.mock import MagicMock

import pytest

from app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def mock_tts_service():
    """Install a mock TTSService for route tests."""
    import app.services.tts as tts_module

    service = MagicMock()
    service.is_loaded = True
    service.sample_rate = 24000
    service.device = 'cpu'
    service.voices_dir = None
    service._active = {'source': 'language', 'value': 'english', 'quantize': False}
    service._boot_active = {'source': 'language', 'value': 'english', 'quantize': False}
    service._loading = False
    service._last_reload_error = None
    service._loading_target = None
    service.validate_voice.return_value = (True, 'ok')
    tts_module._tts_service = service
    yield service
    tts_module._tts_service = None


def test_get_model_returns_active_state(client, mock_tts_service):
    resp = client.get('/v1/model')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['active'] == {'source': 'language', 'value': 'english', 'quantize': False}
    assert body['boot'] == {'source': 'language', 'value': 'english', 'quantize': False}
    assert body['differs_from_boot'] is False
    assert body['loading'] is False
    assert body['model_path_locked'] is False
    assert 'english' in body['available_languages']
    assert len(body['available_languages']) == 12
    assert 'server_version' in body
    assert 'pocket_tts_version' in body


def test_get_model_reports_differs_from_boot(client, mock_tts_service):
    mock_tts_service._active = {'source': 'language', 'value': 'german_24l', 'quantize': True}
    resp = client.get('/v1/model')
    assert resp.get_json()['differs_from_boot'] is True


def test_get_model_reports_model_path_locked(client, mock_tts_service):
    mock_tts_service._boot_active = {'source': 'model_path', 'value': '/x.yaml', 'quantize': False}
    mock_tts_service._active = mock_tts_service._boot_active
    resp = client.get('/v1/model')
    assert resp.get_json()['model_path_locked'] is True


def test_post_model_returns_202_and_starts_load(client, mock_tts_service):
    resp = client.post('/v1/model', json={'language': 'german_24l', 'quantize': True})
    assert resp.status_code == 202
    body = resp.get_json()
    assert body['loading_target'] == {'value': 'german_24l', 'quantize': True}
    mock_tts_service.reload_model_async.assert_called_once_with(
        language='german_24l',
        quantize=True,
    )


def test_post_model_rejects_unknown_language(client, mock_tts_service):
    resp = client.post('/v1/model', json={'language': 'klingon', 'quantize': False})
    assert resp.status_code == 400
    assert 'klingon' in resp.get_json()['error']


def test_post_model_409_when_already_loading(client, mock_tts_service):
    """Route relies on reload_model_async's atomic claim returning False
    when a reload is already in progress, not on its own pre-check."""
    mock_tts_service.reload_model_async.return_value = False
    resp = client.post('/v1/model', json={'language': 'german_24l', 'quantize': False})
    assert resp.status_code == 409


def test_post_model_403_when_model_path_locked(client, mock_tts_service):
    mock_tts_service._boot_active = {'source': 'model_path', 'value': '/x', 'quantize': False}
    resp = client.post('/v1/model', json={'language': 'german_24l', 'quantize': False})
    assert resp.status_code == 403


def test_post_model_400_on_missing_language(client, mock_tts_service):
    resp = client.post('/v1/model', json={'quantize': True})
    assert resp.status_code == 400


def test_post_model_400_on_empty_body(client, mock_tts_service):
    resp = client.post('/v1/model', json={})
    assert resp.status_code == 400


def test_post_model_400_on_non_dict_body(client, mock_tts_service):
    """A JSON value that isn't an object (e.g. a number, string, list) must be
    rejected without raising AttributeError on .get()."""
    resp = client.post('/v1/model', data='42', content_type='application/json')
    assert resp.status_code == 400
    assert 'object' in resp.get_json()['error'].lower()


def test_speech_400_on_non_dict_body(client, mock_tts_service):
    resp = client.post('/v1/audio/speech', data='"hello"', content_type='application/json')
    assert resp.status_code == 400
    assert 'object' in resp.get_json()['error'].lower()


def test_post_model_400_on_non_bool_quantize(client, mock_tts_service):
    """JSON booleans only — string 'false' would be truthy under bool() coercion."""
    resp = client.post('/v1/model', json={'language': 'german_24l', 'quantize': 'false'})
    assert resp.status_code == 400
    assert 'boolean' in resp.get_json()['error'].lower()


def test_post_model_accepts_omitted_quantize(client, mock_tts_service):
    """Omitted quantize defaults to False without triggering the type check."""
    resp = client.post('/v1/model', json={'language': 'german_24l'})
    assert resp.status_code == 202
    mock_tts_service.reload_model_async.assert_called_once_with(
        language='german_24l', quantize=False
    )


def test_speech_returns_503_when_loading(client, mock_tts_service):
    mock_tts_service._loading = True
    resp = client.post('/v1/audio/speech', json={'input': 'hi', 'voice': 'alba'})
    assert resp.status_code == 503
    assert 'reloading' in resp.get_json()['error'].lower()


def test_speech_does_not_500_when_resolve_raises_in_mismatch_path(client, mock_tts_service):
    """If get_voice_state raises ValueError AND _resolve_voice_path itself
    raises (e.g. SSRF protection on http://), the mismatch detection must
    not propagate the second exception — it should fall through to a clean
    400 with the original error message."""
    mock_tts_service.validate_voice.return_value = (True, 'ok')
    mock_tts_service.get_voice_state.side_effect = ValueError('Voice could not be loaded: nope')
    mock_tts_service._resolve_voice_path.side_effect = ValueError('URL scheme not allowed')

    resp = client.post(
        '/v1/audio/speech', json={'input': 'hi', 'voice': 'http://evil.example/voice.wav'}
    )
    assert resp.status_code == 400
    assert resp.get_json()['error'].startswith('Voice could not be loaded')


def test_speech_voice_model_mismatch_returns_400_with_code(client, mock_tts_service, tmp_path):
    """When the resolved voice is a legacy unlabeled .safetensors and pocket-tts
    raises a shape-mismatch during load, the route surfaces a helpful error."""
    # Simulate: validate_voice succeeds, get_voice_state raises.
    legacy = tmp_path / 'emma.safetensors'
    legacy.write_bytes(b'x')
    mock_tts_service.validate_voice.return_value = (True, 'ok')
    mock_tts_service._resolve_voice_path.return_value = str(legacy)
    mock_tts_service.get_voice_state.side_effect = ValueError(
        "Voice 'emma' could not be loaded: RuntimeError: size mismatch for layer.0.weight"
    )
    resp = client.post('/v1/audio/speech', json={'input': 'hi', 'voice': 'emma'})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['error'] == 'voice_model_mismatch'
    assert body['voice'] == 'emma'
    assert body['active_model'] == 'english'


def test_health_includes_active_model(client, mock_tts_service):
    resp = client.get('/health')
    body = resp.get_json()
    assert body['active_model'] == {
        'source': 'language',
        'value': 'english',
        'quantize': False,
    }


def test_home_passes_versions_to_template(client, mock_tts_service):
    resp = client.get('/')
    assert resp.status_code == 200
    html = resp.data.decode()
    assert 'data-server-version=' in html
