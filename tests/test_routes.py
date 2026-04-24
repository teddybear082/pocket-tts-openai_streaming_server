"""Tests for /v1/model routes."""

from unittest.mock import MagicMock, patch

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
