"""
Flask routes for the OpenAI-compatible TTS API.
"""

import atexit
import os
import shutil
import time
import uuid

import librosa
import soundfile as sf
from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

from app.logging_config import get_logger
from app.services.audio import (
    convert_audio,
    get_mime_type,
    tensor_to_pcm_bytes,
    validate_format,
    write_wav_header,
)
from app.services.tts import get_tts_service

logger = get_logger('routes')

# Temporary directory for modified voices
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp_audio')
os.makedirs(TEMP_DIR, exist_ok=True)


def cleanup_temp_files():
    """Delete all temporary audio files."""
    try:
        if os.path.exists(TEMP_DIR):
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f'Failed to delete {file_path}. Reason: {e}')
            logger.info('Temporary files cleaned up')
    except Exception as e:
        logger.error(f'Error during cleanup: {e}')


# Register cleanup on exit
atexit.register(cleanup_temp_files)

# Register cleanup on startup (call it immediately)
cleanup_temp_files()



# Create blueprint
api = Blueprint('api', __name__)


@api.route('/')
def home():
    """Serve the web interface."""
    from app.config import Config

    return render_template('index.html', is_docker=Config.IS_DOCKER)


@api.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for container orchestration.

    Returns service status and basic model info.
    """
    tts = get_tts_service()

    # Validate a built-in voice quickly
    voice_valid, voice_msg = tts.validate_voice('alba')

    return jsonify(
        {
            'status': 'healthy' if tts.is_loaded else 'unhealthy',
            'model_loaded': tts.is_loaded,
            'device': tts.device if tts.is_loaded else None,
            'sample_rate': tts.sample_rate if tts.is_loaded else None,
            'voices_dir': tts.voices_dir,
            'voice_check': {'valid': voice_valid, 'message': voice_msg},
        }
    ), 200 if tts.is_loaded else 503


@api.route('/v1/voices', methods=['GET'])
def list_voices():
    """
    List available voices.

    Returns OpenAI-compatible voice list format.
    """
    tts = get_tts_service()
    voices = tts.list_voices()

    return jsonify(
        {
            'object': 'list',
            'data': [{'id': v['id'], 'name': v['name'], 'object': 'voice'} for v in voices],
        }
    )


@api.route('/delete_temp_files', methods=['POST'])
def delete_temp_files():
    """
    Force deletion of all temporary files.
    """
    cleanup_temp_files()
    return jsonify({'status': 'success', 'message': 'Temporary files deleted'}), 200


@api.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible speech generation endpoint.

    Request body:
        model: string (ignored, for compatibility)
        input: string (required) - Text to synthesize
        voice: string (optional) - Voice ID or path
        response_format: string (optional) - Audio format
        stream: boolean (optional) - Enable streaming

    Returns:
        Audio file or streaming audio response
    """
    from flask import current_app

    data = request.json

    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400

    text = data.get('input')
    if not text:
        return jsonify({'error': "Missing 'input' text"}), 400

    voice = data.get('voice', 'alba')
    speed = float(data.get('speed', 1.0))
    stream_request = data.get('stream', False)

    # Determine format based on streaming
    if stream_request:
        response_format = data.get('response_format', 'pcm')
    else:
        response_format = data.get('response_format', 'mp3')

    target_format = validate_format(response_format)

    tts = get_tts_service()

    # Handle speed adjustment
    temp_voice_path = None
    if speed != 1.0:
        try:
            # Resolve original voice path
            original_voice_path = tts.resolve_voice_path(voice)

            # Load audio with librosa
            # target_sr=None preserves native sampling rate
            y, sr = librosa.load(original_voice_path, sr=None)

            # Time-stretch
            # rate > 1.0 speeds up, rate < 1.0 slows down
            y_stretched = librosa.effects.time_stretch(y, rate=speed)

            # Save to temp file
            unique_filename = f"{uuid.uuid4()}_{os.path.basename(original_voice_path)}"
            if not unique_filename.endswith('.wav'):
                 unique_filename += '.wav'

            temp_voice_path = os.path.join(TEMP_DIR, unique_filename)
            sf.write(temp_voice_path, y_stretched, sr)

            # Use temp file as voice
            voice = temp_voice_path

        except Exception as e:
            logger.warning(f"Failed to apply speed modifier {speed}: {e}. Using original voice.")
            # If we created a file but failed later, cleanup
            if temp_voice_path and os.path.exists(temp_voice_path):
                os.unlink(temp_voice_path)
            temp_voice_path = None

    # Validate voice first
    is_valid, msg = tts.validate_voice(voice)
    if not is_valid:
        available = [v['id'] for v in tts.list_voices()]
        return jsonify(
            {
                'error': f"Voice '{voice}' not found",
                'available_voices': available[:10],  # Limit to first 10
                'hint': 'Use /v1/voices to see all available voices',
            }
        ), 400

    try:
        voice_state = tts.get_voice_state(voice)

        # Check if streaming should be used
        use_streaming = stream_request or current_app.config.get('STREAM_DEFAULT', False)

        if use_streaming:
            return _stream_audio(tts, voice_state, text, target_format)
        else:
            return _generate_file(tts, voice_state, text, target_format)

    except ValueError as e:
        logger.warning(f'Voice loading failed: {e}')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Generation failed')
        return jsonify({'error': str(e)}), 500


def _generate_file(tts, voice_state, text: str, fmt: str):
    """Generate complete audio and return as file."""
    t0 = time.time()
    audio_tensor = tts.generate_audio(voice_state, text)
    generation_time = time.time() - t0

    logger.info(f'Generated {len(text)} chars in {generation_time:.2f}s')

    audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
    mimetype = get_mime_type(fmt)

    return send_file(
        audio_buffer, mimetype=mimetype, as_attachment=True, download_name=f'speech.{fmt}'
    )


def _stream_audio(tts, voice_state, text: str, fmt: str):
    """Stream audio chunks."""
    # Normalize streaming format: we always emit PCM bytes, optionally wrapped
    # in a WAV container. For non-PCM/WAV formats (e.g. mp3, opus), coerce to
    # raw PCM to avoid mismatched content-type vs. payload.
    stream_fmt = fmt
    if stream_fmt not in ('pcm', 'wav'):
        logger.warning(
            "Requested streaming format '%s' is not supported for streaming; "
            "falling back to 'pcm'.",
            stream_fmt,
        )
        stream_fmt = 'pcm'

    def generate():
        stream = tts.generate_audio_stream(voice_state, text)
        for chunk_tensor in stream:
            yield tensor_to_pcm_bytes(chunk_tensor)

    def stream_with_header():
        # Yield WAV header first if streaming as WAV
        if stream_fmt == 'wav':
            yield write_wav_header(tts.sample_rate, num_channels=1, bits_per_sample=16)
        yield from generate()

    mimetype = get_mime_type(stream_fmt)

    return Response(stream_with_context(stream_with_header()), mimetype=mimetype)
