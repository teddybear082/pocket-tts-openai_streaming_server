"""
Audio conversion and streaming utilities.
"""

import io
import struct
import subprocess
import threading
from typing import Iterator

import torch
import torchaudio

from app.logging_config import get_logger

logger = get_logger('audio')

# Valid audio formats
VALID_FORMATS = {'mp3', 'wav', 'opus', 'aac', 'flac', 'pcm'}

# OpenAI's documented range for the `speed` parameter on /v1/audio/speech.
SPEED_MIN = 0.25
SPEED_MAX = 4.0

# ffmpeg's `atempo` filter only accepts factors in [0.5, 2.0] per stage. Speeds
# outside that band are realised by chaining stages (e.g. 0.25 → 0.5,0.5).
_ATEMPO_STAGE_MIN = 0.5
_ATEMPO_STAGE_MAX = 2.0


def validate_format(fmt: str) -> str:
    """
    Normalize and validate the requested audio format.

    Args:
        fmt: Requested format string

    Returns:
        Validated format string
    """
    fmt = fmt.lower()

    # OpenAI sometimes sends 'mpeg' for mp3
    if fmt == 'mpeg':
        return 'mp3'

    if fmt not in VALID_FORMATS:
        logger.warning(f"Unknown format '{fmt}', falling back to wav")
        return 'wav'

    return fmt


def convert_audio(
    audio_tensor: torch.Tensor, sample_rate: int, target_format: str = 'wav'
) -> io.BytesIO:
    """
    Convert a raw audio tensor to a byte buffer in the specified format.

    Args:
        audio_tensor: The audio waveform (1D or 2D)
        sample_rate: The sample rate of the audio
        target_format: The target audio format

    Returns:
        Buffer containing the encoded audio data
    """
    buffer = io.BytesIO()

    # Ensure tensor is CPU
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()

    # Ensure 2D (channels, time)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Handle PCM raw bytes (no container)
    if target_format == 'pcm':
        try:
            pcm_bytes = tensor_to_pcm_bytes(audio_tensor)
            buffer.write(pcm_bytes)
            buffer.seek(0)
            return buffer
        except Exception as e:
            logger.error(f'Error converting audio to PCM: {e}')
            raise

    # Map OpenAI format names to torchaudio/backend supported format names
    # torchaudio uses 'ogg' as the container for 'opus'
    # 'aac' usually requires 'adts' or 'm4a'
    actual_format = target_format
    if actual_format == 'opus':
        actual_format = 'ogg'
    elif actual_format == 'aac':
        actual_format = 'adts'

    try:
        torchaudio.save(buffer, audio_tensor, sample_rate, format=actual_format)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(
            f'Error converting audio to {target_format} (backend format: {actual_format}): {e}'
        )
        raise


def write_wav_header(
    sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, num_frames: int = 0
) -> bytes:
    """
    Generate a WAV header for streaming.

    If num_frames is 0, set to max value (streaming/unknown length).

    Args:
        sample_rate: Audio sample rate
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample
        num_frames: Number of frames (0 for unknown/streaming)

    Returns:
        WAV header bytes
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Data size: if unknown, max uint32
    data_size = num_frames * block_align
    if num_frames == 0:
        data_size = 0xFFFFFFFF - 36

    chunk_size = 36 + data_size

    header = io.BytesIO()
    header.write(b'RIFF')
    header.write(struct.pack('<I', chunk_size))
    header.write(b'WAVE')
    header.write(b'fmt ')
    header.write(struct.pack('<I', 16))  # Subchunk1Size (16 for PCM)
    header.write(struct.pack('<H', 1))  # AudioFormat (1 for PCM)
    header.write(struct.pack('<H', num_channels))
    header.write(struct.pack('<I', sample_rate))
    header.write(struct.pack('<I', byte_rate))
    header.write(struct.pack('<H', block_align))
    header.write(struct.pack('<H', bits_per_sample))
    header.write(b'data')
    header.write(struct.pack('<I', data_size))

    return header.getvalue()


def tensor_to_pcm_bytes(chunk_tensor: torch.Tensor) -> bytes:
    """
    Convert audio tensor chunk to 16-bit PCM bytes.

    Args:
        chunk_tensor: Audio tensor chunk

    Returns:
        PCM audio bytes
    """
    if chunk_tensor.is_cuda:
        chunk_tensor = chunk_tensor.cpu()

    if chunk_tensor.dim() == 1:
        chunk_tensor = chunk_tensor.unsqueeze(0)

    # Convert to 16-bit PCM
    pcm = (chunk_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
    return pcm.numpy().tobytes()


def _build_atempo_chain(speed: float) -> str:
    """
    Build an ffmpeg `-af` filter chain that scales playback speed by `speed`
    while preserving pitch. ffmpeg's `atempo` filter only accepts factors in
    [0.5, 2.0] per stage, so factors outside that band are realised by chaining
    stages (e.g. 0.25 → `atempo=0.5,atempo=0.5`).

    Caller is responsible for ensuring `speed != 1.0` and that `speed` lies in
    [SPEED_MIN, SPEED_MAX]; this function is not invoked on the default path.
    """
    filters: list[str] = []
    remaining = speed
    while remaining < _ATEMPO_STAGE_MIN:
        filters.append(f'atempo={_ATEMPO_STAGE_MIN}')
        remaining /= _ATEMPO_STAGE_MIN
    while remaining > _ATEMPO_STAGE_MAX:
        filters.append(f'atempo={_ATEMPO_STAGE_MAX}')
        remaining /= _ATEMPO_STAGE_MAX
    filters.append(f'atempo={remaining:.6f}')
    return ','.join(filters)


def apply_atempo_buffer(
    audio_buffer: io.BytesIO, target_format: str, speed: float
) -> io.BytesIO:
    """
    Run a fully-encoded audio buffer through ffmpeg's `atempo` filter and
    return a new buffer in the same `target_format`.

    Used for non-streaming responses. ffmpeg auto-detects the input container
    from the byte stream; the output muxer is selected explicitly so the
    response payload matches the advertised MIME type.

    Caller must guarantee `speed != 1.0` — the default path skips this entirely
    so that speed=1.0 requests have zero overhead vs. the pre-`speed` behaviour.
    """
    af_chain = _build_atempo_chain(speed)

    # Mirror the OpenAI-name → ffmpeg-muxer mapping used in `convert_audio`.
    mux_format = target_format
    if mux_format == 'opus':
        mux_format = 'ogg'
    elif mux_format == 'aac':
        mux_format = 'adts'
    elif mux_format == 'pcm':
        mux_format = 's16le'

    audio_buffer.seek(0)
    audio_bytes = audio_buffer.read()

    try:
        proc = subprocess.run(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', 'pipe:0',
                '-af', af_chain,
                '-f', mux_format,
                'pipe:1',
            ],
            input=audio_bytes,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg is required to apply the 'speed' parameter but was not "
            'found on PATH'
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8', errors='replace')[:500]
        raise RuntimeError(f'ffmpeg atempo failed: {stderr}') from e

    return io.BytesIO(proc.stdout)


def apply_atempo_to_pcm_stream(
    pcm_chunks: Iterator[bytes], sample_rate: int, speed: float
) -> Iterator[bytes]:
    """
    Pipe an iterator of raw 16-bit mono PCM chunks through a long-running
    ffmpeg `atempo` subprocess and yield rate-altered PCM chunks.

    A producer thread writes incoming chunks to ffmpeg's stdin and closes
    the pipe when the source iterator is exhausted; the main generator
    reads from ffmpeg's stdout until EOF. atempo produces fewer (or more)
    output samples than input, so chunk boundaries on the output do not
    correspond to those on the input — that's fine, the bytes are still
    valid contiguous PCM.

    Caller must guarantee `speed != 1.0` — the default streaming path does
    not invoke this and stays byte-for-byte identical to the pre-`speed`
    behaviour.
    """
    af_chain = _build_atempo_chain(speed)

    try:
        proc = subprocess.Popen(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-f', 's16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                '-i', 'pipe:0',
                '-af', af_chain,
                '-f', 's16le',
                'pipe:1',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg is required to apply the 'speed' parameter but was not "
            'found on PATH'
        ) from e

    feed_error: list[BaseException] = []

    def _feed() -> None:
        try:
            for chunk in pcm_chunks:
                proc.stdin.write(chunk)
        except BrokenPipeError:
            # ffmpeg exited early; the consumer side will surface stderr.
            pass
        except BaseException as exc:  # noqa: BLE001 — re-raised on main thread
            feed_error.append(exc)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    feeder = threading.Thread(target=_feed, name='atempo-feeder', daemon=True)
    feeder.start()

    read_size = 4096
    try:
        while True:
            data = proc.stdout.read(read_size)
            if not data:
                break
            yield data
    finally:
        feeder.join(timeout=5)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    if feed_error:
        raise feed_error[0]

    if proc.returncode not in (0, None):
        stderr = proc.stderr.read().decode('utf-8', errors='replace')[:500]
        logger.warning('ffmpeg atempo exited %d: %s', proc.returncode, stderr)


def get_mime_type(fmt: str) -> str:
    """
    Get the MIME type for an audio format.

    Args:
        fmt: Audio format string

    Returns:
        MIME type string
    """
    mime_types = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'pcm': 'audio/L16',
        'opus': 'audio/opus',
        'aac': 'audio/aac',
        'flac': 'audio/flac',
    }
    return mime_types.get(fmt, f'audio/{fmt}')
