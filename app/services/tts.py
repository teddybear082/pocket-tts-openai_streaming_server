"""
TTS Service - handles model loading, voice management, and audio generation.
"""

import os
import time
from pathlib import Path

from app.config import Config
from app.logging_config import get_logger

logger = get_logger('tts')

# Lazy import pocket_tts to allow for better error handling
TTSModel = None
export_model_state = None


def _ensure_pocket_tts():
    """Ensure pocket-tts is imported."""
    global TTSModel, export_model_state
    if TTSModel is None:
        try:
            from pocket_tts import TTSModel as _TTSModel
            from pocket_tts.models.tts_model import export_model_state as _export_state

            TTSModel = _TTSModel
            export_model_state = _export_state
        except ImportError as exc:
            raise ImportError('pocket-tts not found. Install with: pip install pocket-tts') from exc


class TTSService:
    """
    Service class for Text-to-Speech operations.
    Manages model loading, voice caching, and audio generation.
    """

    def __init__(self):
        import threading
        from collections import OrderedDict

        self.model = None
        self.voice_cache: OrderedDict = OrderedDict()
        self.voices_dir: str | None = None
        self._model_loaded = False

        # Concurrency + reload state
        self._lock = threading.Lock()
        self._loading = False  # fast-path flag; read without lock
        self._active: dict | None = None
        self._boot_active: dict | None = None

        from pathlib import Path
        self.cache_dir: Path | None = Path(Config.VOICE_CACHE_DIR)

    def _ensure_cache_dir(self) -> None:
        """Create the voice cache directory on first need. Tolerate read-only FS."""
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                f'Voice cache dir {self.cache_dir} is not writable ({e}); '
                f'cache persistence disabled.'
            )
            self.cache_dir = None

    def _save_cloned_state(self, state: dict, audio_path) -> None:
        """Persist a freshly-cloned state as <stem>.<active_tag>.safetensors."""
        from pathlib import Path

        from app.services.voice_cache import active_model_tag

        self._ensure_cache_dir()
        if self.cache_dir is None:
            return

        audio_path = Path(audio_path)
        tag = active_model_tag((self._active or {}).get('value') or 'english')
        target = self.cache_dir / f'{audio_path.stem}.{tag}.safetensors'
        try:
            export_model_state(state, target)
            logger.info(f'Saved cloned voice state to {target}')
        except OSError as e:
            logger.warning(f'Could not save voice cache to {target}: {e}')

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded and self.model is not None

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        if self.model:
            return self.model.sample_rate
        return 24000  # Default pocket-tts sample rate

    @property
    def device(self) -> str:
        """Get the model's device."""
        if self.model:
            return str(self.model.device)
        return 'unknown'

    def load_model(
        self,
        model_path: str | None = None,
        language: str | None = None,
        quantize: bool = False,
        _is_boot: bool = True,
    ) -> None:
        """
        Load the TTS model.

        Args:
            model_path: Optional path to model config file (.yaml)
            language: Optional language identifier (e.g., english, french_24l).
                      Incompatible with model_path.
            quantize: If True, apply dynamic int8 quantization to reduce memory.
            _is_boot: Internal; True only for the first boot-time load. Controls
                      whether _boot_active is initialized.
        """
        _ensure_pocket_tts()

        logger.info('Loading Pocket TTS model...')
        t0 = time.time()

        effective_path = model_path

        if not effective_path:
            _, bundle_model = Config.get_bundle_paths()
            if bundle_model and os.path.isfile(bundle_model):
                effective_path = bundle_model
                logger.info(f'Using bundled model: {effective_path}')

        try:
            if effective_path:
                logger.info(f'Loading model from: {effective_path}')
                self.model = TTSModel.load_model(config=effective_path, quantize=quantize)
                active = {'source': 'model_path', 'value': effective_path, 'quantize': quantize}
            elif language:
                logger.info(f'Loading model with language: {language}')
                self.model = TTSModel.load_model(language=language, quantize=quantize)
                active = {'source': 'language', 'value': language, 'quantize': quantize}
            else:
                logger.info('Loading default model from HuggingFace...')
                self.model = TTSModel.load_model(quantize=quantize)
                active = {'source': 'default', 'value': None, 'quantize': quantize}

            self._model_loaded = True
            self._active = active
            if _is_boot:
                self._boot_active = dict(active)

            load_time = time.time() - t0
            logger.info(
                f'Model loaded in {load_time:.2f}s. '
                f'Device: {self.device}, Sample Rate: {self.sample_rate}'
            )

        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise

    def reload_model(self, language: str, quantize: bool) -> None:
        """Reload the model with a new language / quantize setting.

        Blocks until any in-flight generation finishes. Serialized via the
        single TTSService lock (pocket-tts v2 TTSModel is not thread-safe).
        """
        if self._boot_active and self._boot_active['source'] == 'model_path':
            raise RuntimeError(
                'Cannot switch language: server was started with a custom model_path.'
            )

        if language not in Config.SUPPORTED_LANGUAGES:
            raise ValueError(f'Unsupported language: {language!r}')

        if self._loading:
            raise RuntimeError('already loading')

        self._loading = True
        try:
            with self._lock:
                previous_model = self.model
                previous_active = self._active
                try:
                    self.load_model(language=language, quantize=quantize, _is_boot=False)
                    self.voice_cache.clear()
                except Exception:
                    # Restore previous state on failure so the server remains usable.
                    self.model = previous_model
                    self._active = previous_active
                    raise
        finally:
            self._loading = False

    def set_voices_dir(self, voices_dir: str | None) -> None:
        """
        Set the directory for custom voice files.

        Args:
            voices_dir: Path to directory containing voice files
        """
        if voices_dir and os.path.isdir(voices_dir):
            self.voices_dir = voices_dir
            logger.info(f'Voices directory set to: {voices_dir}')
        elif voices_dir:
            logger.warning(f'Voices directory not found: {voices_dir}')
            self.voices_dir = None
        else:
            self.voices_dir = None

    def get_voice_state(self, voice_id_or_path: str) -> dict:
        """Resolve a voice ID to a cached model state.

        When the resolved path is raw audio, encode it against the active model
        and persist the result as <stem>.<active_tag>.safetensors in cache_dir.
        If a tagged cache exists but its source audio is newer, regenerate.
        """
        from pathlib import Path

        from app.services.voice_cache import (
            AUDIO_EXTENSIONS,
            cache_is_stale,
        )

        if not self.is_loaded:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        resolved_key = self._resolve_voice_path(voice_id_or_path)

        # Cache check with LRU touch.
        if resolved_key in self.voice_cache:
            self.voice_cache.move_to_end(resolved_key)
            logger.debug(f'Using in-memory voice state for: {resolved_key}')
            return self.voice_cache[resolved_key]

        # If resolved to a tagged cache, check staleness against raw-audio source.
        resolved_path = Path(resolved_key) if os.path.isabs(resolved_key) else None
        regenerate_from_source: Path | None = None

        if resolved_path and resolved_path.suffix == '.safetensors' and self.voices_dir:
            stem = resolved_path.stem.split('.', 1)[0]  # strip tag if present
            for ext in AUDIO_EXTENSIONS:
                source = Path(self.voices_dir) / f'{stem}{ext}'
                if cache_is_stale(cache_path=resolved_path, source_path=source):
                    regenerate_from_source = source
                    break

        logger.info(f'Loading voice: {resolved_key}')
        t0 = time.time()

        try:
            if regenerate_from_source:
                logger.info(f'Regenerating stale cache from {regenerate_from_source}')
                state = self.model.get_state_for_audio_prompt(regenerate_from_source, truncate=True)
                self._save_cloned_state(state, regenerate_from_source)
            elif resolved_path and resolved_path.suffix.lower() in AUDIO_EXTENSIONS:
                state = self.model.get_state_for_audio_prompt(resolved_path, truncate=True)
                self._save_cloned_state(state, resolved_path)
            else:
                # Pre-made .safetensors OR built-in OR hf:// — let pocket-tts handle it.
                state = self.model.get_state_for_audio_prompt(resolved_key)

            # LRU insert.
            self.voice_cache[resolved_key] = state
            if len(self.voice_cache) > 32:
                self.voice_cache.popitem(last=False)

            load_time = time.time() - t0
            logger.info(f'Voice loaded in {load_time:.2f}s: {resolved_key}')
            return state

        except Exception as e:
            logger.error(f"Failed to load voice '{voice_id_or_path}': {e}")
            raise ValueError(f"Voice '{voice_id_or_path}' could not be loaded: {e}") from e

    def _resolve_voice_path(self, voice_id_or_path: str) -> str:
        """Resolve a voice identifier using per-model cache preference.

        Raises ValueError on unsafe URL schemes (retained from previous behavior).
        """
        from pathlib import Path

        from app.services.voice_cache import resolve_voice_path

        # Retain SSRF protection.
        if voice_id_or_path.startswith(('http://', 'https://')):
            raise ValueError(
                f'URL scheme not allowed for security reasons: {voice_id_or_path[:50]}. '
                "Use 'hf://' for HuggingFace models or provide a local file path."
            )

        if voice_id_or_path.startswith('hf://'):
            return voice_id_or_path

        # Built-in names pass through untouched (pocket-tts handles resolution).
        if voice_id_or_path.lower() in Config.BUILTIN_VOICES:
            return voice_id_or_path.lower()

        # Absolute path hit.
        if os.path.isabs(voice_id_or_path) and os.path.exists(voice_id_or_path):
            return voice_id_or_path

        voices_path = Path(self.voices_dir) if self.voices_dir else None
        active_model = (self._active or {}).get('value') or 'english'

        resolved = resolve_voice_path(
            voice_id=voice_id_or_path,
            active_model=active_model,
            voices_dir=voices_path,
            cache_dir=self.cache_dir,
        )
        return str(resolved) if isinstance(resolved, Path) else resolved

    def validate_voice(self, voice_id_or_path: str) -> tuple[bool, str]:
        """
        Validate if a voice can be loaded (fast check without full loading).

        Args:
            voice_id_or_path: Voice identifier

        Returns:
            Tuple of (is_valid, message)
        """
        # Block unsafe URL schemes first
        if voice_id_or_path.startswith(('http://', 'https://')):
            return (
                False,
                'HTTP/HTTPS URLs are not allowed for security reasons. Use hf:// for HuggingFace models.',
            )

        try:
            resolved = self._resolve_voice_path(voice_id_or_path)
        except ValueError as e:
            return False, str(e)

        # Built-in voices are always valid
        if resolved.lower() in Config.BUILTIN_VOICES:
            return True, f'Built-in voice: {resolved}'

        # HuggingFace URLs - assume valid
        if resolved.startswith('hf://'):
            return True, f'HuggingFace voice: {resolved}'

        # Local file - check existence
        if os.path.exists(resolved):
            return True, f'Local voice file: {resolved}'

        return False, f'Voice not found: {voice_id_or_path}'

    def generate_audio(self, voice_state: dict, text: str):
        """Generate complete audio for given text."""
        import torch  # noqa: F401 — kept for return-type doc

        if self._loading:
            raise RuntimeError('model reloading')
        if not self.is_loaded:
            raise RuntimeError('Model not loaded')

        with self._lock:
            t0 = time.time()
            audio = self.model.generate_audio(voice_state, text)
            gen_time = time.time() - t0

        logger.info(f'Generated {len(text)} chars in {gen_time:.2f}s')
        return audio

    def generate_audio_stream(self, voice_state: dict, text: str):
        """Generate audio in streaming chunks. Holds the lock for the entire stream."""
        if self._loading:
            raise RuntimeError('model reloading')
        if not self.is_loaded:
            raise RuntimeError('Model not loaded')

        logger.info(f'Starting streaming generation for {len(text)} chars')
        with self._lock:
            yield from self.model.generate_audio_stream(voice_state, text)

    def list_voices(self) -> list[dict]:
        """
        List all available voices.

        Returns:
            List of voice dictionaries with 'id' and 'name' keys
        """
        voices = []

        # Built-in voices (sorted alphabetically)
        builtin_sorted = sorted(Config.BUILTIN_VOICES)
        for voice in builtin_sorted:
            voices.append({'id': voice, 'name': voice.capitalize(), 'type': 'builtin'})

        # Custom voices from directory
        custom_voices = []
        if self.voices_dir and os.path.isdir(self.voices_dir):
            voice_dir = Path(self.voices_dir)

            # Collect all valid files
            voice_files = []
            for ext in Config.VOICE_EXTENSIONS:
                voice_files.extend(voice_dir.glob(f'*{ext}'))

            # Sort alphabetically by filename
            voice_files.sort(key=lambda f: f.name.lower())

            for voice_file in voice_files:
                # Format name: "bobby_mcfern" -> "Bobby Mcfern"
                clean_name = voice_file.stem.replace('_', ' ').replace('-', ' ').title()

                custom_voices.append(
                    {
                        'id': voice_file.name,
                        'name': clean_name,
                        'type': 'custom',
                    }
                )

        voices.extend(custom_voices)
        return voices


# Global service instance
_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    """Get the global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
