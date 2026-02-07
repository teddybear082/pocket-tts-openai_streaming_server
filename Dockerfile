# Dockerfile for PocketTTS OpenAI-Compatible Server
# Optimized for CPU inference (pocket-tts runs efficiently on CPU)
# Uses CPU-only PyTorch for smaller image size (~700MB vs ~2GB)

FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (requirements.txt specifies CPU-only PyTorch)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt


# Production image
FROM python:3.10-slim

# Install runtime dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash pockettts
WORKDIR /app

# Copy application code
COPY --chown=pockettts:pockettts app/ ./app/
COPY --chown=pockettts:pockettts static/ ./static/
COPY --chown=pockettts:pockettts templates/ ./templates/
COPY --chown=pockettts:pockettts voices/ ./voices/
COPY --chown=pockettts:pockettts server.py ./

# Create logs directory, and ensure app directory is owned by user
RUN chown pockettts:pockettts /app && mkdir -p /app/logs && chown pockettts:pockettts /app/logs

# Create HuggingFace cache directory (for volume mount)
RUN mkdir -p /home/pockettts/.cache/huggingface && \
    chown -R pockettts:pockettts /home/pockettts/.cache

# Switch to non-root user
USER pockettts

# Environment variables with defaults
ENV POCKET_TTS_HOST=0.0.0.0 \
    POCKET_TTS_PORT=49112 \
    POCKET_TTS_VOICES_DIR=/app/voices \
    POCKET_TTS_LOG_DIR=/app/logs \
    POCKET_TTS_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 49112

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:49112/health')" || exit 1

# Run server
CMD ["python", "server.py"]
