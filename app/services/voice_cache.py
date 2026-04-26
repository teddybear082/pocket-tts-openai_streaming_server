"""Pure filename and path logic for per-model cached voice states.

Kept separate from tts.py so it can be exercised without loading pocket-tts.
"""

import re
from pathlib import Path

from app.config import Config

_ILLEGAL_FS_CHARS = re.compile(r'[\\/:*?"<>|]')
_CONFIG_EXTENSIONS = ('.yaml', '.yml', '.json')


def active_model_tag(raw_model: str) -> str:
    """Normalize a language identifier for use as a filename tag.

    Handles three input shapes:
      - plain language identifier (e.g. 'english_2026-04') → unchanged
      - aliased identifier (e.g. 'english') → canonical alias target
      - custom --model-path value (e.g. r'C:\\models\\english.yaml') →
        reduced to the file stem ('english'), since path separators and
        Windows drive colons are illegal in filenames and would crash
        safetensors serialization (issue #13).
    """
    canonical = Config.LEGACY_MODEL_ALIASES.get(raw_model, raw_model)

    # If the value looks like a file path (separators present, or it ends in
    # a known config extension), reduce it to a filesystem-safe stem.
    if '/' in canonical or '\\' in canonical or canonical.lower().endswith(_CONFIG_EXTENSIONS):
        last = canonical.replace('\\', '/').rsplit('/', 1)[-1]
        for ext in _CONFIG_EXTENSIONS:
            if last.lower().endswith(ext):
                last = last[: -len(ext)]
                break
        canonical = _ILLEGAL_FS_CHARS.sub('_', last)

    return canonical


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

    Preference order (within each directory, the canonical-tagged filename
    is checked first; if `active_model` is an alias, the alias-tagged
    filename is checked as a fallback so files written under either name
    resolve correctly):
      1. cache_dir/<voice_id>.<canonical_tag>.safetensors
      2. cache_dir/<voice_id>.<active_model>.safetensors  (alias fallback)
      3. voices_dir/<voice_id>.<canonical_tag>.safetensors
      4. voices_dir/<voice_id>.<active_model>.safetensors  (alias fallback)
      5. voices_dir/<voice_id>.{wav,mp3,flac}
      6. voices_dir/<voice_id>.safetensors  (legacy unlabeled)
      7. The bare voice_id string — pocket-tts will resolve (e.g. built-ins).

    New caches are always written using the canonical tag (see
    `_save_cloned_state` in tts.py), so the alias-tagged paths exist only
    for files placed by external tools or by users running an older version.
    """
    canonical_tag = active_model_tag(active_model)
    candidates = [f'{voice_id}.{canonical_tag}.safetensors']
    if active_model != canonical_tag:
        candidates.append(f'{voice_id}.{active_model}.safetensors')

    for directory in (cache_dir, voices_dir):
        if not directory:
            continue
        for name in candidates:
            p = directory / name
            if p.exists():
                return p

    if voices_dir:
        for ext in ('.wav', '.mp3', '.flac'):
            p = voices_dir / f'{voice_id}{ext}'
            if p.exists():
                return p

        p = voices_dir / f'{voice_id}.safetensors'
        if p.exists():
            return p

    return voice_id


AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac')


def list_voice_stems(
    voices_dir: Path | None,
    cache_dir: Path | None,
) -> list[str]:
    """Return the unique voice stems present across voices_dir and cache_dir."""
    stems: set[str] = set()
    tags = known_model_tags()

    for directory in (voices_dir, cache_dir):
        if not directory or not directory.is_dir():
            continue
        for ext in AUDIO_EXTENSIONS:
            for f in directory.glob(f'*{ext}'):
                stems.add(f.stem)
        for f in directory.glob('*.safetensors'):
            stem, _tag = parse_safetensors_name(f.name, tags)
            stems.add(stem)

    return sorted(stems)


def cache_is_stale(cache_path: Path, source_path: Path) -> bool:
    """True when `source_path` exists and is newer than `cache_path`."""
    if not source_path.exists():
        return False
    if not cache_path.exists():
        return False
    return source_path.stat().st_mtime > cache_path.stat().st_mtime
