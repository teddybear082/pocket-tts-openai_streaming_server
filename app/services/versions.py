"""Cached version lookup for the web UI."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from app import __version__ as _SERVER_VERSION_FALLBACK


@lru_cache(maxsize=1)
def get_versions() -> dict[str, str]:
    """Return {'server': ..., 'pocket_tts': ...}.

    Falls back to the hardcoded `app.__version__` for the server when the
    package isn't installed via pip. `pocket_tts` falls back to 'unknown'.
    """
    try:
        server = version('pocket-tts-openai-server')
    except PackageNotFoundError:
        server = _SERVER_VERSION_FALLBACK

    try:
        pocket_tts = version('pocket-tts')
    except PackageNotFoundError:
        pocket_tts = 'unknown'

    return {'server': server, 'pocket_tts': pocket_tts}
