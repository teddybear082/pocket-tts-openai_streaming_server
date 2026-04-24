"""
Audio conversion and streaming utilities.
"""

import io
import struct

import torch
import torchaudio

from app.logging_config import get_logger

logger = get_logger('audio')

# Valid audio formats
VALID_FORMATS = {'mp3', 'wav', 'opus', 'aac', 'flac', 'pcm'}


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
        logger.error(f"Error converting audio to {target_format} (backend format: {actual_format}): {e}")
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
