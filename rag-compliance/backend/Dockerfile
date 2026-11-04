# ── Build args ───────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies layer (cached unless requirements.txt changes) ───────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application layer ─────────────────────────────────────────────────────────
FROM deps AS production

ARG EMBED_MODEL
ENV EMBED_MODEL=${EMBED_MODEL}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

COPY . .

# Pre-download the embedding model so the container starts fast
# (runs at build time; fails silently so offline builds still work)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
import os; \
model_name = os.environ.get('EMBED_MODEL', 'BAAI/bge-small-en-v1.5'); \
print(f'Pre-loading embedder: {model_name}'); \
SentenceTransformer(model_name); \
print('Embedder cached.')" || echo "Embedder pre-load skipped (no network)"

# Create non-root user for security
RUN useradd -m -u 1001 appuser \
    && mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Production: 2 workers × 2 threads, graceful 30s timeout
# Adjust --workers based on available CPU cores (rule of thumb: 2 × CPUs + 1)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info", \
     "--access-log"]
