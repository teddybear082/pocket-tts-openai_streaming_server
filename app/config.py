"""
Configuration management for PocketTTS OpenAI Server.
Loads settings from environment variables with sensible defaults.
"""

import os
import sys
from pathlib import Path


def get_base_path() -> Path:
    """Get the base path for the application, handling PyInstaller frozen state."""
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            # One-file mode
            return Path(sys._MEIPASS)
        else:
            # One-dir mode
            return Path(sys.executable).parent
    return Path(__file__).parent.parent


class Config:
    """Application configuration loaded from environment variables."""

    # Base paths
    BASE_PATH = get_base_path()
    IS_FROZEN = getattr(sys, 'frozen', False)

    # Server settings
    HOST = os.environ.get('POCKET_TTS_HOST', '0.0.0.0')
    PORT = int(os.environ.get('POCKET_TTS_PORT', '49112'))

    # Model settings
    MODEL_PATH = os.environ.get('POCKET_TTS_MODEL_PATH', None)
    LANGUAGE = os.environ.get('POCKET_TTS_LANGUAGE', None)
    QUANTIZE = os.environ.get('POCKET_TTS_QUANTIZE', 'false').lower() == 'true'

    # Supported languages (pocket-tts v2.0.0 predefined model YAMLs)
    SUPPORTED_LANGUAGES = [
        'english',  # alias for english_2026-04 (default)
        'english_2026-01',
        'english_2026-04',
        'french_24l',  # no bare `french` — upstream raises
        'german',
        'german_24l',
        'italian',
        'italian_24l',
        'portuguese',
        'portuguese_24l',
        'spanish',
        'spanish_24l',
    ]

    # Canonicalize equivalent model IDs so tagged caches dedupe.
    LEGACY_MODEL_ALIASES = {
        'english': 'english_2026-04',
        'english_2026-01': 'english_2026-04',
    }

    # Writable voice cache dir for tagged .safetensors clones.
    VOICE_CACHE_DIR = os.environ.get(
        'POCKET_TTS_VOICE_CACHE_DIR',
        str(BASE_PATH / 'voice_cache'),
    )

    DEFAULT_VOICE = os.environ.get(
        'POCKET_TTS_DEFAULT_VOICE', 'hf://kyutai/tts-voices/alba-mackenna/casual.wav'
    )

    # Voice directory
    VOICES_DIR = os.environ.get('POCKET_TTS_VOICES_DIR', None)

    # Streaming default
    STREAM_DEFAULT = os.environ.get('POCKET_TTS_STREAM_DEFAULT', 'false').lower() == 'true'

    # Text preprocessing default
    TEXT_PREPROCESS_DEFAULT = (
        os.environ.get('POCKET_TTS_TEXT_PREPROCESS_DEFAULT', 'false').lower() == 'true'
    )

    # Docker detection
    @staticmethod
    def _is_docker() -> bool:
        """Detect if running in a Docker container."""
        # Check for .dockerenv file (most reliable)
        if os.path.exists('/.dockerenv'):
            return True
        # Check cgroup for docker/containerd references
        try:
            with open('/proc/1/cgroup') as f:
                return any('docker' in line or 'containerd' in line for line in f)
        except (FileNotFoundError, PermissionError):
            return False

    IS_DOCKER = _is_docker.__func__()

    # Logging
    LOG_LEVEL = os.environ.get('POCKET_TTS_LOG_LEVEL', 'INFO')
    LOG_DIR = os.environ.get('POCKET_TTS_LOG_DIR', str(BASE_PATH / 'logs'))
    LOG_FILE = os.environ.get('POCKET_TTS_LOG_FILE', 'pocket_tts.log')
    LOG_MAX_BYTES = int(os.environ.get('POCKET_TTS_LOG_MAX_BYTES', str(10 * 1024 * 1024)))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('POCKET_TTS_LOG_BACKUP_COUNT', '5'))

    # Built-in voice mappings (these are resolved by pocket-tts internally)
    BUILTIN_VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']

    # Supported audio extensions for custom voices
    VOICE_EXTENSIONS = ('.wav', '.mp3', '.flac', '.safetensors')

    @classmethod
    def get_bundle_paths(cls) -> tuple:
        """Get bundled paths for frozen executables."""
        if cls.IS_FROZEN:
            voices_dir = cls.BASE_PATH / 'voices'
            model_path = cls.BASE_PATH / 'model' / 'b6369a24.yaml'
            return (
                str(voices_dir) if voices_dir.is_dir() else None,
                str(model_path) if model_path.is_file() else None,
            )
        return None, None

    @classmethod
    def get_template_folder(cls) -> str:
        """Get the templates folder path."""
        return str(cls.BASE_PATH / 'templates')

    @classmethod
    def get_static_folder(cls) -> str:
        """Get the static files folder path."""
        return str(cls.BASE_PATH / 'static')
