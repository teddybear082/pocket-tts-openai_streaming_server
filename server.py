#!/usr/bin/env python3
"""
PocketTTS OpenAI-Compatible Server

A drop-in replacement for OpenAI's TTS API using the pocket-tts model.
Supports streaming, custom voices, and runs on CPU.

Usage:
    python server.py [OPTIONS]

    # Or with environment variables:
    POCKET_TTS_PORT=8080 python server.py
"""

import argparse
import os
import sys

from app import create_app, init_tts_service
from app.config import Config
from app.logging_config import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PocketTTS OpenAI-Compatible Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with defaults
    python server.py

    # Custom port and voices directory
    python server.py --port 8080 --voices-dir ./my_voices

    # Load French model with quantization
    python server.py --language french_24l --quantize

Environment Variables:
    POCKET_TTS_HOST         Server host (default: 0.0.0.0)
    POCKET_TTS_PORT         Server port (default: 49112)
    POCKET_TTS_MODEL_PATH   Path to model config file
    POCKET_TTS_LANGUAGE     Model language (e.g., english, french_24l)
    POCKET_TTS_QUANTIZE     Enable int8 quantization (default: false)
    POCKET_TTS_VOICES_DIR   Path to voices directory
    POCKET_TTS_STREAM_DEFAULT Enable streaming by default
    POCKET_TTS_TEXT_PREPROCESS_DEFAULT Enable text preprocessing by default
    POCKET_TTS_LOG_DIR      Log directory path
        """,
    )

    parser.add_argument(
        '--host', type=str, default=Config.HOST, help=f'Host to bind to (default: {Config.HOST})'
    )
    parser.add_argument(
        '--port', type=int, default=Config.PORT, help=f'Port to listen on (default: {Config.PORT})'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=Config.MODEL_PATH,
        dest='model_path',
        help='Path to model config file (.yaml) or variant name',
    )
    parser.add_argument(
        '--voices-dir',
        type=str,
        default=Config.VOICES_DIR,
        dest='voices_dir',
        help='Directory containing voice files',
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        default=Config.STREAM_DEFAULT,
        help='Enable streaming by default for all requests',
    )
    parser.add_argument(
        '--text-preprocess',
        action='store_true',
        default=Config.TEXT_PREPROCESS_DEFAULT,
        help='Enable text preprocessing for all requests',
    )
    parser.add_argument(
        '--language',
        type=str,
        default=Config.LANGUAGE,
        dest='language',
        help='Model language (e.g., english, french_24l, german_24l, portuguese, italian, spanish_24l). Incompatible with --model-path.',
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        default=Config.QUANTIZE,
        help='Apply dynamic int8 quantization to reduce memory usage and improve speed.',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=Config.LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        dest='log_level',
        help='Logging level',
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Update config from args (environment takes precedence via Config class)
    os.environ.setdefault('POCKET_TTS_LOG_LEVEL', args.log_level)

    # Create app
    app = create_app(
        {'STREAM_DEFAULT': args.stream, 'TEXT_PREPROCESS_DEFAULT': args.text_preprocess}
    )

    logger = get_logger()

    # Validate mutually exclusive options
    if args.language and args.model_path:
        logger.error('--language and --model-path are mutually exclusive. Use one or the other.')
        sys.exit(1)

    # Initialize TTS service
    try:
        init_tts_service(
            model_path=args.model_path,
            voices_dir=args.voices_dir,
            language=args.language,
            quantize=args.quantize,
        )
    except Exception as e:
        logger.error(f'Failed to initialize TTS service: {e}')
        sys.exit(1)

    # Start server with Waitress (production WSGI server)
    try:
        from waitress import serve

        logger.info(f'Starting PocketTTS server on http://{args.host}:{args.port}')
        logger.info('Press Ctrl+C to stop')

        serve(app, host=args.host, port=args.port, threads=4, url_scheme='http')

    except ImportError:
        logger.warning('Waitress not installed, falling back to Flask dev server')
        logger.warning('Install waitress for production: pip install waitress')
        app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
