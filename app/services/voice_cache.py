"""Pure filename and path logic for per-model cached voice states.

Kept separate from tts.py so it can be exercised without loading pocket-tts.
"""

from pathlib import Path

from app.config import Config


def active_model_tag(raw_model: str) -> str:
    """Normalize a language identifier for use as a filename tag.

    Equivalent names (english, english_2026-01, english_2026-04) map to a
    single canonical tag so cache files don't duplicate.
    """
    return Config.LEGACY_MODEL_ALIASES.get(raw_model, raw_model)


def known_model_tags() -> set[str]:
    """Filename tags we treat as model identifiers during parsing.

    Includes both the raw supported languages and the alias-target canonicals.
    """
    return set(Config.SUPPORTED_LANGUAGES) | set(Config.LEGACY_MODEL_ALIASES.values())


def parse_safetensors_name(filename: str, tags: set[str]) -> tuple[str, str | None]:
    """Split `stem.model_tag.safetensors` into (stem, tag).

    Returns (stem, None) when the filename is unlabeled or the final segment
    is not a recognized tag.
    """
    base = Path(filename).stem  # strips .safetensors
    if '.' in base:
        stem, tag = base.rsplit('.', 1)
        if tag in tags:
            return stem, tag
    return base, None


def resolve_voice_path(
    voice_id: str,
    active_model: str,
    voices_dir: Path | None,
    cache_dir: Path | None,
) -> Path | str:
    """Resolve a voice identifier to its on-disk source.

    Preference order:
      1. cache_dir/<voice_id>.<active_tag>.safetensors
      2. voices_dir/<voice_id>.<active_tag>.safetensors
      3. voices_dir/<voice_id>.{wav,mp3,flac}
      4. voices_dir/<voice_id>.safetensors  (legacy unlabeled)
      5. The bare voice_id string — pocket-tts will resolve (e.g. built-ins).
    """
    tag = active_model_tag(active_model)
    tagged_filename = f'{voice_id}.{tag}.safetensors'

    if cache_dir:
        p = cache_dir / tagged_filename
        if p.exists():
            return p

    if voices_dir:
        p = voices_dir / tagged_filename
        if p.exists():
            return p

        for ext in ('.wav', '.mp3', '.flac'):
            p = voices_dir / f'{voice_id}{ext}'
            if p.exists():
                return p

        p = voices_dir / f'{voice_id}.safetensors'
        if p.exists():
            return p

    return voice_id
