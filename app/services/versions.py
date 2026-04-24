"""Cached version lookup for the web UI."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=1)
def get_versions() -> dict[str, str]:
    """Return {'server': ..., 'pocket_tts': ...} with 'unknown' fallback."""
    result = {}
    for key, pkg in (('server', 'pocket-tts-openai-server'), ('pocket_tts', 'pocket-tts')):
        try:
            result[key] = version(pkg)
        except PackageNotFoundError:
            result[key] = 'unknown'
    return result
