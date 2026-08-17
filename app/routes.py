"""
Flask routes for the OpenAI-compatible TTS API.
"""

import threading
import time

from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

from app.config import Config
from app.logging_config import get_logger
from app.services.audio import (
    SPEED_MAX,
    SPEED_MIN,
    apply_atempo_buffer,
    apply_atempo_to_pcm_stream,
    convert_audio,
    encode_pcm_stream,
    ffmpeg_available,
    get_mime_type,
    streaming_mime_type,
    tensor_to_pcm_bytes,
    validate_format,
    write_wav_header,
)
from app.services.preprocess import TextPreprocessor
from app.services.tts import get_tts_service
from app.services.versions import get_versions

logger = get_logger('routes')

# Create blueprint
api = Blueprint('api', __name__)

# Per-request audio stats storage.  Keyed by a client-generated UUID; values
# are dicts with 'generation_time' (float, seconds) and 'audio_duration'
# (float, seconds).  Cleaned up after the client polls once.
_audio_stats_lock = threading.Lock()
_audio_stats: dict[str, dict] = {}

# Create text preprocessor instance, some options changed from defaults
text_preprocessor = TextPreprocessor(
    remove_urls=False,
    remove_emails=False,
    remove_html=True,
    remove_hashtags=True,
    remove_mentions=False,
    remove_punctuation=False,
    remove_stopwords=False,
    remove_extra_whitespace=False,
)


@api.route('/')
def home():
    """Serve the web interface."""
    from app.config import Config

    return render_template(
        'index.html',
        is_docker=Config.IS_DOCKER,
        versions=get_versions(),
    )


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
            'active_model': tts._active,
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
            'data': [
                {
                    'id': v['id'],
                    'name': v['name'],
                    'object': 'voice',
                    'type': v.get('type', 'builtin'),
                }
                for v in voices
            ],
        }
    )


@api.route('/v1/model', methods=['GET'])
def get_model():
    """Return the active model state, boot snapshot, and supported languages."""
    tts = get_tts_service()

    active = tts._active or {'source': 'default', 'value': None, 'quantize': False}
    boot = tts._boot_active or active
    differs = active != boot
    model_path_locked = boot.get('source') == 'model_path'
    versions = get_versions()

    return jsonify(
        {
            'active': active,
            'boot': boot,
            'differs_from_boot': differs,
            'loading': tts._loading,
            'loading_target': getattr(tts, '_loading_target', None),
            'last_error': getattr(tts, '_last_reload_error', None),
            'model_path_locked': model_path_locked,
            'available_languages': list(Config.SUPPORTED_LANGUAGES),
            'server_version': versions['server'],
            'pocket_tts_version': versions['pocket_tts'],
        }
    )


@api.route('/v1/model', methods=['POST'])
def post_model():
    """Request a runtime model switch. Returns 202; UI polls GET for completion."""
    data = request.json
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    language = data.get('language')

    # Reject non-bool `quantize` rather than coercing — `bool('false')` is True,
    # which would silently enable quantization for any client sending a string.
    quantize = False
    if 'quantize' in data:
        if not isinstance(data['quantize'], bool):
            return jsonify({'error': "Field 'quantize' must be a boolean"}), 400
        quantize = data['quantize']

    if not language:
        return jsonify({'error': "Missing required field 'language'"}), 400

    if language not in Config.SUPPORTED_LANGUAGES:
        return jsonify(
            {
                'error': f"Unknown language: '{language}'",
                'available': list(Config.SUPPORTED_LANGUAGES),
            }
        ), 400

    tts = get_tts_service()

    if tts._boot_active and tts._boot_active.get('source') == 'model_path':
        return jsonify(
            {
                'error': 'Language switching disabled: server started with --model-path.',
            }
        ), 403

    # `reload_model_async` does the atomic check-and-claim, so the 409 race
    # window between `if tts._loading` and `start()` is gone.
    started = tts.reload_model_async(language=language, quantize=quantize)
    if not started:
        return jsonify({'error': 'A model reload is already in progress.'}), 409

    return jsonify(
        {
            'status': 'accepted',
            'loading_target': {'value': language, 'quantize': quantize},
        }
    ), 202


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
        speed: float (optional) - Playback speed multiplier in
            [0.25, 4.0], default 1.0. When omitted or equal to 1.0 the
            default code path is unchanged. Other values run the audio
            through ffmpeg's `atempo` filter (preserves pitch).

    Returns:
        Audio file or streaming audio response
    """
    from flask import current_app

    data = request.json

    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    text = data.get('input')
    if not text:
        return jsonify({'error': "Missing 'input' text"}), 400

    voice = data.get('voice', 'alba')
    stream_request = data.get('stream', False)

    response_format = data.get('response_format', 'mp3')
    target_format = validate_format(response_format)

    # `speed` is intentionally only validated when present — clients that
    # never send the field hit the same code path as before.
    speed = 1.0
    if 'speed' in data:
        try:
            speed = float(data['speed'])
        except (TypeError, ValueError):
            return jsonify({'error': "'speed' must be a number"}), 400
        if not (SPEED_MIN <= speed <= SPEED_MAX):
            return jsonify(
                {
                    'error': (f"'speed' must be between {SPEED_MIN} and {SPEED_MAX}"),
                    'received': speed,
                }
            ), 400

    # atempo needs ffmpeg. When it's missing, degrade gracefully rather than
    # failing the request: serve normal-speed audio and warn so the operator
    # knows why `speed` had no effect.
    if speed != 1.0 and not ffmpeg_available():
        logger.warning(
            "Ignoring 'speed=%s': ffmpeg not found on PATH. Install ffmpeg to "
            'enable the speed parameter.',
            speed,
        )
        speed = 1.0

    tts = get_tts_service()

    if tts._loading:
        return jsonify({'error': 'Model is reloading; retry shortly.'}), 503

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

        # Check if text preprocessing should be used
        use_text_preprocess = current_app.config.get('TEXT_PREPROCESS_DEFAULT', False)
        # Preprocess text
        if use_text_preprocess:
            # logger.info(f'Preprocessing text: {text}')
            text = text_preprocessor.process(text)
            # logger.info(f'Preprocessed text: {text}')
        if use_streaming:
            stats_id = data.get('stats_id')
            return _stream_audio(
                tts, voice_state, text, target_format, speed=speed, stats_id=stats_id
            )
        return _generate_file(tts, voice_state, text, target_format, speed=speed)

    except ValueError as e:
        msg = str(e)
        # Detect the legacy-unlabeled-safetensors mismatch pattern. Re-resolving
        # can itself raise (e.g. SSRF protection on http:// URLs); treat any
        # failure here as "not a mismatch" and fall through to the generic 400.
        try:
            resolved = tts._resolve_voice_path(voice) if not tts._loading else ''
        except Exception:
            resolved = ''
        is_legacy_st = resolved.endswith('.safetensors') and not any(
            resolved.endswith(f'.{tag}.safetensors') for tag in Config.SUPPORTED_LANGUAGES
        )
        mismatch_markers = ('size mismatch', 'Error(s) in loading state_dict', 'shape')
        if is_legacy_st and any(m in msg for m in mismatch_markers):
            return jsonify(
                {
                    'error': 'voice_model_mismatch',
                    'message': (
                        f"Voice '{voice}' appears to have been cloned for a different "
                        f'model. Upload the original audio (.wav/.mp3/.flac) to '
                        f're-clone for the active model, or switch to the model it '
                        f'was generated for.'
                    ),
                    'voice': voice,
                    'active_model': (tts._active or {}).get('value'),
                }
            ), 400

        logger.warning(f'Voice loading failed: {e}')
        return jsonify({'error': msg}), 400
    except Exception as e:
        logger.exception('Generation failed')
        return jsonify({'error': str(e)}), 500


@api.route('/v1/audio/speech', methods=['GET'])
def generate_speech_get():
    """GET version of speech generation for browser ``<audio>`` element playback.

    Accepts the same parameters as the POST endpoint but as query strings.
    Always streams the response so the browser can begin playback immediately.

    Parameters:
        input: string (required) - Text to synthesize
        voice: string (optional) - Voice ID or path (default: alba)
        response_format: string (optional) - Audio format (default: mp3)
        speed: float (optional) - Playback speed multiplier in [0.25, 4.0]
        stats_id: string (optional) - Client-generated UUID; server stores
            generation timing under this key so the client can poll
            ``/v1/audio/speech/stats?id=<stats_id>`` after playback.
    """
    from flask import current_app

    text = request.args.get('input')
    if not text:
        return jsonify({'error': "Missing 'input' query parameter"}), 400

    voice = request.args.get('voice', 'alba')
    response_format = request.args.get('response_format', 'mp3')
    target_format = validate_format(response_format)

    speed = request.args.get('speed', 1.0, type=float)
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        return jsonify(
            {
                'error': f"'speed' must be between {SPEED_MIN} and {SPEED_MAX}",
                'received': speed,
            }
        ), 400

    if speed != 1.0 and not ffmpeg_available():
        logger.warning(
            "Ignoring 'speed=%s': ffmpeg not found on PATH. Install ffmpeg to "
            'enable the speed parameter.',
            speed,
        )
        speed = 1.0

    tts = get_tts_service()

    if tts._loading:
        return jsonify({'error': 'Model is reloading; retry shortly.'}), 503

    is_valid, msg = tts.validate_voice(voice)
    if not is_valid:
        available = [v['id'] for v in tts.list_voices()]
        return jsonify(
            {
                'error': f"Voice '{voice}' not found",
                'available_voices': available[:10],
                'hint': 'Use /v1/voices to see all available voices',
            }
        ), 400

    try:
        voice_state = tts.get_voice_state(voice)

        use_text_preprocess = current_app.config.get('TEXT_PREPROCESS_DEFAULT', False)
        if use_text_preprocess:
            text = text_preprocessor.process(text)

        # GET always streams — the <audio> element needs a continuous response.
        stats_id = request.args.get('stats_id')
        return _stream_audio(
            tts, voice_state, text, target_format, speed=speed, stats_id=stats_id
        )

    except ValueError as e:
        logger.warning(f'Voice loading failed: {e}')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Generation failed')
        return jsonify({'error': str(e)}), 500


@api.route('/v1/audio/speech/stats', methods=['GET'])
def audio_speech_stats():
    """Return generation stats for a streaming request, then delete them.

    The client passes a ``stats_id`` query parameter that it generated before
    starting the streaming request.  The server stores timing data under that
    key once generation completes and removes it on the first poll.
    """
    stats_id = request.args.get('id')
    if not stats_id:
        return jsonify({'error': "Missing 'id' query parameter"}), 400

    with _audio_stats_lock:
        stats = _audio_stats.pop(stats_id, None)

    if stats is None:
        return jsonify({'error': 'Stats not found (may have already been polled)'}), 404

    return jsonify(stats)


def _generate_file(tts, voice_state, text: str, fmt: str, speed: float = 1.0):
    """Generate complete audio and return as file."""
    t0 = time.time()
    audio_tensor = tts.generate_audio(voice_state, text)
    generation_time = time.time() - t0

    logger.info(f'Generated {len(text)} chars in {generation_time:.2f}s')

    audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
    if speed != 1.0:
        audio_buffer = apply_atempo_buffer(audio_buffer, fmt, speed)
    mimetype = get_mime_type(fmt)

    return send_file(
        audio_buffer, mimetype=mimetype, as_attachment=True, download_name=f'speech.{fmt}'
    )


def _stream_audio(
    tts, voice_state, text: str, fmt: str, speed: float = 1.0, stats_id: str | None = None
):
    """Stream audio chunks.

    For PCM and WAV the data is emitted directly (WAV prepends a header).
    All other formats are encoded on the fly by piping PCM through ffmpeg.
    Opus uses an OGG container so the browser ``<audio>`` element can play
    it natively.

    When *stats_id* is provided the generator tracks wall-clock generation
    time and total PCM sample count, then stores the resulting stats in
    ``_audio_stats`` so the client can poll ``/v1/audio/speech/stats``.
    """

    sample_rate = tts.sample_rate
    total_pcm_bytes = 0
    t0 = time.monotonic()

    def pcm_chunks():
        nonlocal total_pcm_bytes
        stream = tts.generate_audio_stream(voice_state, text)
        for chunk_tensor in stream:
            pcm = tensor_to_pcm_bytes(chunk_tensor)
            total_pcm_bytes += len(pcm)
            yield pcm

    def stream_with_header():
        if fmt == 'pcm':
            yield from pcm_chunks()
        elif fmt == 'wav':
            yield write_wav_header(sample_rate, num_channels=1, bits_per_sample=16)
            if speed == 1.0:
                yield from pcm_chunks()
            else:
                yield from apply_atempo_to_pcm_stream(
                    pcm_chunks(), sample_rate, speed
                )
        else:
            # mp3 / opus / aac / flac — encode on the fly via ffmpeg.
            yield from encode_pcm_stream(
                pcm_chunks(), sample_rate, fmt, speed
            )

        # Store stats once the generator is exhausted (all data sent).
        if stats_id:
            generation_time = time.monotonic() - t0
            # total_pcm_bytes is 16-bit mono samples → 2 bytes per sample.
            audio_duration = total_pcm_bytes / (sample_rate * 2) if sample_rate else 0
            with _audio_stats_lock:
                _audio_stats[stats_id] = {
                    'generation_time': round(generation_time, 4),
                    'audio_duration': round(audio_duration, 4),
                }

    mimetype = streaming_mime_type(fmt)

    return Response(stream_with_context(stream_with_header()), mimetype=mimetype)
