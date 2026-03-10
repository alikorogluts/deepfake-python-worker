# ══════════════════════════════════════════════════════════════════
# DeepFake Python Worker – Dockerfile
# ══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="alikoroglu <https://github.com/alikorogluts>"
LABEL description="DeepFake Detection – RabbitMQ Async Worker"
LABEL version="3.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# ── YENİ: HuggingFace cache klasörü ──────────────────────────────
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache
# ─────────────────────────────────────────────────────────────────

COPY src/ ./src/

RUN useradd --no-create-home --shell /bin/false worker

# ── YENİ: cache klasörü worker'a ver ─────────────────────────────
RUN chown -R worker:worker /app/.cache
# ─────────────────────────────────────────────────────────────────

USER worker

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "\
import os, socket; \
s = socket.create_connection(\
    (os.getenv('RABBITMQ_HOST','rabbitmq'), int(os.getenv('RABBITMQ_PORT','5672'))),\
    timeout=5\
); s.close(); print('OK')"

CMD ["python", "-m", "src.worker.main"]

