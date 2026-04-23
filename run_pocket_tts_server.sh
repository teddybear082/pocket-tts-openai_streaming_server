#!/usr/bin/env bash
# Pocket TTS Server Launcher (Linux/macOS)
# This script provides an interactive configuration menu before starting the server.

set -euo pipefail

# Resolve the script directory (handles symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Banner
echo ""
echo "========================================"
echo "   Pocket TTS OpenAI Streaming Server"
echo "========================================"
echo ""

# --- 0. Hugging Face Token ---
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    read -r -p "Hugging Face Token (leave blank if already logged in): " INPUT_HF
    if [ -n "$INPUT_HF" ]; then
        HF_TOKEN="$INPUT_HF"
        export HF_TOKEN
        echo "[INFO] Hugging Face Token set."
    fi
fi

# --- 1. Activate Virtual Environment ---
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[WARNING] 'venv' not found. Using system Python..."
fi

echo ""
echo "Configure the server (press Enter to accept defaults):"
echo ""

# --- 2. Host ---
read -r -p "Host IP [0.0.0.0]: " INPUT_HOST
HOST="${INPUT_HOST:-0.0.0.0}"

# --- 3. Port ---
read -r -p "Port [49112]: " INPUT_PORT
PORT="${INPUT_PORT:-49112}"

# --- 4. Model Path ---
echo "Model Config Path (.yaml) or variant name (leave blank for built-in):"
read -r INPUT_MODEL
MODEL_ARG=()
if [ -n "$INPUT_MODEL" ]; then
    MODEL_ARG+=("--model-path" "$INPUT_MODEL")
fi

# --- 5. Voices Directory ---
DEFAULT_VOICES="$SCRIPT_DIR/voices"
if [ ! -d "$DEFAULT_VOICES" ]; then
    DEFAULT_VOICES=""
fi
PROMPT_VOICES=""
if [ -n "$DEFAULT_VOICES" ]; then
    PROMPT_VOICES="$DEFAULT_VOICES"
fi
[ -n "$DEFAULT_VOICES" ] && PROMPT_VOICES="$DEFAULT_VOICES"
read -r -p "Voices Directory [$(if [ -n "$DEFAULT_VOICES" ]; then echo "$DEFAULT_VOICES"; else echo "None"; fi)]: " INPUT_VOICES
VOICES_ARG=()
if [ -n "$INPUT_VOICES" ]; then
    VOICES_ARG+=("--voices-dir" "$INPUT_VOICES")
fi

# --- 6. Streaming ---
read -r -p "Enable Streaming? (y/n) [Y]: " INPUT_STREAM
STREAM_ARG=()
if [ "${INPUT_STREAM,,}" != "n" ]; then
    STREAM_ARG+=("--stream")
fi

# --- 7. Text Preprocessing ---
read -r -p "Enable Text Preprocessing? (y/n) [Y]: " INPUT_PREPROCESS
PREPROCESS_ARG=()
if [ "${INPUT_PREPROCESS,,}" != "n" ]; then
    PREPROCESS_ARG+=("--text-preprocess")
fi

# --- 8. Language ---
echo "Language (english, french_24l, german_24l, portuguese, italian, spanish_24l - leave blank for default):"
read -r INPUT_LANGUAGE
LANGUAGE_ARG=()
if [ -n "$INPUT_LANGUAGE" ]; then
    LANGUAGE_ARG+=("--language" "$INPUT_LANGUAGE")
fi

# --- 9. Quantization ---
read -r -p "Enable int8 Quantization? (y/n) [N]: " INPUT_QUANTIZE
QUANTIZE_ARG=()
if [ "${INPUT_QUANTIZE,,}" = "y" ]; then
    QUANTIZE_ARG+=("--quantize")
fi

# --- Summary ---
echo ""
echo "========================================"
echo "Starting Pocket TTS Server..."
echo "  Host         : $HOST"
echo "  Port         : $PORT"
[ -n "$INPUT_MODEL" ] && echo "  Model        : $INPUT_MODEL"
[ -n "$INPUT_VOICES" ] && echo "  Voices       : $INPUT_VOICES"
echo "  Streaming    : $([ "${INPUT_STREAM,,}" = "n" ] && echo "Disabled" || echo "Enabled")"
echo "  Preprocessing: $([ "${INPUT_PREPROCESS,,}" = "n" ] && echo "Disabled" || echo "Enabled")"
[ -n "$INPUT_LANGUAGE" ] && echo "  Language     : $INPUT_LANGUAGE"
[ "${INPUT_QUANTIZE,,}" = "y" ] && echo "  Quantization : Enabled"
echo "========================================"
echo ""

# --- Run ---
python server.py \
    --host "$HOST" \
    --port "$PORT" \
    "${MODEL_ARG[@]+"${MODEL_ARG[@]}"}" \
    "${VOICES_ARG[@]+"${VOICES_ARG[@]}"}" \
    "${STREAM_ARG[@]+"${STREAM_ARG[@]}"}" \
    "${PREPROCESS_ARG[@]+"${PREPROCESS_ARG[@]}"}" \
    "${LANGUAGE_ARG[@]+"${LANGUAGE_ARG[@]}"}" \
    "${QUANTIZE_ARG[@]+"${QUANTIZE_ARG[@]}"}"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[ERROR] Server exited with error code $EXIT_CODE."
fi
exit $EXIT_CODE
