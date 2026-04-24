# PocketTTS OpenAI-Compatible Server

An OpenAI-compatible Text-to-Speech API server powered by [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts). Drop-in replacement for OpenAI's TTS API with support for streaming, custom voices, and voice cloning.

Tested and working fully with [WingmanAI by Shipbit](https://www.wingman-ai.com/). Due to low resource use, can be used for real time local text to speech even while playing intensive video games (even in VR!) with WingmanAI.

**Key Features:**

- 🎯 **OpenAI API Compatible** - Works with any OpenAI TTS client
- 🚀 **Real-time Streaming** - Low-latency audio generation
- 🎤 **150+ Community Voices** - Ready-to-use voice library included
- 🎭 **Voice Cloning** - Clone any voice from a short audio sample
- 🐳 **Docker Ready** - One-command deployment
- 💻 **Cross-platform** - Runs on Windows, macOS, and Linux
- ⚡ **CPU Optimized** - No GPU required
- 🎤 **Text pre-processing** - Clean text for words and symbols TTS usually has difficulty with, automatically

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/teddybear082/pocket-tts-openai_streaming_server.git
cd pocket-tts-openai_streaming_server

# Start the server
docker compose up -d

# View logs
docker compose logs -f
```

The server will be available at `http://localhost:49112`

**Custom Configuration:**

```bash
# Change port
POCKET_TTS_PORT=8080 docker compose up -d

# Use custom voices directory
POCKET_TTS_VOICES_DIR=/path/to/my/voices docker compose up -d
```

### Option 2: Python (from source)

```bash
# Clone the repository
git clone https://github.com/teddybear082/pocket-tts-openai_streaming_server.git
cd pocket-tts-openai_streaming_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py
```

**Command Line Options:**

```bash
python server.py --help

# Custom port and voices
python server.py --port 8080 --voices-dir ./my_voices

# Enable streaming by default
python server.py --stream

# Enable text preprocessing
python server.py --text-preprocess
```

### Option 3: Windows Executable

1. Download the latest release from [Releases](https://github.com/teddybear082/pocket-tts-openai_streaming_server/releases)
2. Extract the ZIP file
3. Double-click `PocketTTS-Server.exe` to run with defaults
4. Or run `run_pocket_tts_server_exe.bat` for custom configuration

## Web Interface

Open `http://localhost:49112` in your browser to access the built-in web UI:

- Select from available voices
- Enter text to synthesize
- Listen to generated audio directly

## API Usage

### Generate Speech

**Endpoint:** `POST /v1/audio/speech`

```bash
curl http://localhost:49112/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello world! This is a test.",
    "voice": "alba"
  }' \
  --output speech.mp3
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:49112/v1",
    api_key="not-needed"  # No authentication required
)

# Generate and save audio
response = client.audio.speech.create(
    model="tts-1",
    voice="alba",
    input="Hello world! This is a test."
)
response.stream_to_file("output.mp3")

# Streaming
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alba",
    input="This is streaming audio.",
    response_format="pcm"
) as response:
    for chunk in response.iter_bytes():
        # Process audio chunks in real-time
        pass
```

### API Reference

| Endpoint           | Method | Description                              |
| ------------------ | ------ | ---------------------------------------- |
| `/`                | GET    | Web interface                            |
| `/health`          | GET    | Health check for container orchestration |
| `/v1/voices`       | GET    | List available voices                    |
| `/v1/audio/speech` | POST   | Generate speech audio                    |

**Speech Parameters:**

| Parameter         | Type    | Required | Default | Description                                        |
| ----------------- | ------- | -------- | ------- | -------------------------------------------------- |
| `model`           | string  | No       | -       | Ignored (for OpenAI compatibility)                 |
| `input`           | string  | Yes      | -       | Text to synthesize                                 |
| `voice`           | string  | No       | `alba`  | Voice ID (see `/v1/voices`)                        |
| `response_format` | string  | No       | `mp3`   | Output format: `mp3`, `wav`, `pcm`, `opus`, `aac`, `flac` |
| `stream`          | boolean | No       | `false` | Enable streaming response                          |

## Custom Voices

### Using Custom Voice Files

1. **Create a voices directory** with your audio files (`.wav`, `.mp3`, `.flac`)
2. **Configure the server** to use your directory:

   **Docker:**

   ```bash
   POCKET_TTS_VOICES_DIR=/path/to/voices docker compose up -d
   ```

   **Python:**

   ```bash
   python server.py --voices-dir /path/to/voices
   ```

   **Windows EXE:**
   Use the batch launcher and specify the voices directory when prompted.

3. **Use your voice** by filename:
   ```json
   { "voice": "my_voice.wav", "input": "Hello!" }
   ```

### Voice File Guidelines

- **Duration:** 3-15 seconds of clear speech works best
- **Quality:** Clean audio without background noise
- **Format:** WAV, MP3, or FLAC
- **Tip:** Use [Adobe Podcast Enhance](https://podcast.adobe.com/enhance) to clean noisy samples

### Built-in Voices

The following voices are available by default:
`alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

The `voices/` directory includes 150+ community-contributed voices.

## Configuration

### Environment Variables

| Variable                            | Default    | Description                            |
| ------------------------------------| ---------- | -------------------------------------- |
| `POCKET_TTS_HOST`                   | `0.0.0.0`  | Server bind address                    |
| `POCKET_TTS_PORT`                   | `49112`    | Server port                            |
| `POCKET_TTS_VOICES_DIR`             | `./voices` | Custom voices directory                |
| `POCKET_TTS_MODEL_PATH`             | -          | Custom model path                      |
| `POCKET_TTS_STREAM_DEFAULT`         | `true`     | Enable streaming by default            |
| `POCKET_TTS_TEXT_PREPROCESS_DEFAULT`| `true`     | Enable text preprocessing by default   |
| `POCKET_TTS_LOG_LEVEL`              | `INFO`     | Log level: DEBUG, INFO, WARNING, ERROR |
| `POCKET_TTS_LOG_DIR`                | `./logs`   | Log files directory                    |
| `HF_TOKEN`                          | -          | Hugging Face token (for voice cloning) |

### Docker Compose Options

See [docker-compose.yml](docker-compose.yml) for all available options including:

- Volume mounts for custom voices
- Resource limits
- Health check configuration
- HuggingFace cache persistence

## Project Structure

```
pocket-tts-openai_streaming_server/
├── app/                    # Application modules
│   ├── __init__.py        # Flask app factory
│   ├── config.py          # Configuration management
│   ├── logging_config.py  # Logging setup
│   ├── routes.py          # API endpoints
│   └── services/          # Business logic
│       ├── audio.py       # Audio conversion
│       └── tts.py         # TTS service
|       |-- preprocess.py  # Text preprocessor
├── static/                 # Web UI assets
├── templates/              # HTML templates
├── voices/                 # Voice files
├── server.py              # Main entry point
├── Dockerfile             # Container build
├── docker-compose.yml     # Container orchestration
└── requirements.txt       # Python dependencies
```

## Development

### Dependencies

| File                   | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| `requirements.txt`     | Runtime dependencies only (Flask, torch, pocket-tts) |
| `requirements-dev.txt` | Adds dev tools: ruff (linting), pytest (testing)     |

### Running Locally

```bash
# Install runtime dependencies only
pip install -r requirements.txt

# Or install with dev tools (recommended for contributors)
pip install -r requirements-dev.txt

# Run with debug logging
python server.py --log-level DEBUG
```

### Linting

```bash
pip install ruff
ruff check .
ruff format .
```

### Building Windows EXE

```bash
pip install pyinstaller
pyinstaller --onefile --name PocketTTS-Server \
  --add-data "static;static" \
  --add-data "templates;templates" \
  --add-data "voices;voices" \
  --add-data "app;app" \
  server.py
```

## Troubleshooting

### Model Loading Takes Long

First run downloads the model (~500MB). Subsequent runs use cached model.

**Docker:** Model cache is persisted in a Docker volume.

### Voice Cloning Requires HF Token

For voice cloning, you may need a Hugging Face token:

1. Get token from https://huggingface.co/settings/tokens
2. Set `HF_TOKEN` environment variable

### Port Already in Use

```bash
# Use a different port
python server.py --port 8080

# Or with Docker
POCKET_TTS_PORT=8080 docker compose up -d
```

## Credits

- [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai Labs
- Community voice contributors (see [voices/credits.txt](voices/credits.txt))

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

Pocket-TTS is subject to its own license terms.
