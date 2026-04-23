"""
PocketTTS OpenAI-Compatible Server

Flask application factory and initialization.
"""

from flask import Flask

from app.config import Config
from app.logging_config import get_logger, setup_logging


def create_app(config_overrides: dict = None) -> Flask:
    """
    Application factory for creating the Flask app.

    Args:
        config_overrides: Optional dictionary of config values to override

    Returns:
        Configured Flask application
    """
    # Setup logging first
    setup_logging()
    logger = get_logger()

    # Create Flask app with correct paths
    app = Flask(
        __name__,
        template_folder=Config.get_template_folder(),
        static_folder=Config.get_static_folder(),
    )

    # Apply default config
    app.config['STREAM_DEFAULT'] = Config.STREAM_DEFAULT

    # Apply overrides
    if config_overrides:
        app.config.update(config_overrides)

    # Register blueprints
    from app.routes import api

    app.register_blueprint(api)

    logger.info('Flask application created')

    return app


def init_tts_service(
    model_path: str = None,
    voices_dir: str = None,
    language: str = None,
    quantize: bool = False,
) -> None:
    """
    Initialize the TTS service with model and voices.

    Args:
        model_path: Optional path to model config file
        voices_dir: Optional path to voices directory
        language: Optional language identifier (e.g., english, french_24l)
        quantize: Whether to apply dynamic int8 quantization
    """
    from app.services.tts import get_tts_service

    logger = get_logger()
    tts = get_tts_service()

    # Load model
    tts.load_model(model_path=model_path, language=language, quantize=quantize)

    # Set voices directory
    if voices_dir:
        tts.set_voices_dir(voices_dir)
    else:
        # Check for bundled voices
        bundle_voices, _ = Config.get_bundle_paths()
        if bundle_voices:
            tts.set_voices_dir(bundle_voices)

    logger.info('TTS service initialized')
