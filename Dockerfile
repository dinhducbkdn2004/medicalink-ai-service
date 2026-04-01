# medicalink-ai-service — worker RabbitMQ + RAG Qdrant
# Build: docker build -t medicalink-ai:latest .
# Run:   docker run --env-file .env medicalink-ai:latest
#
# Bật ghi log đánh giá: mount thư mục chứa RAG_EVAL_LOG_PATH (vd. /app/data).

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY medicalink_ai ./medicalink_ai
RUN pip install --no-cache-dir . --no-deps

# Cache FlashRank / FastEmbed trong container; production nên mount volume
ENV FLASHRANK_CACHE_DIR=/app/.cache/flashrank
RUN mkdir -p /app/.cache/flashrank /app/data

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Worker mặc định; job đồng bộ batch: docker run ... medicalink-ai-sync
CMD ["medicalink-ai-worker"]
