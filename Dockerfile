# syntax=docker/dockerfile:1
#
# Intelligent Recommendation Platform - API image (final delivery).
#
# Build:  docker build -t recommendation-platform .
# Run:    docker run -p 8000:8000 recommendation-platform
#
# The source tree is not pip-installed: config/settings.py derives every
# artifact path from its own location (REPO_ROOT), so the code runs from
# /app/src and the artifacts live in /app/artifacts.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1

# libgomp1: OpenMP runtime required by the implicit (ALS) wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not invalidate this layer.
# The runtime list is read from pyproject.toml (single source of truth),
# plus the optional backends the image serves: ALS, the neural model, and the
# Redis / RabbitMQ / MongoDB clients used by the "full" compose profile (without
# them the adapters would silently fall back to their in-process doubles).
# PyTorch comes from the CPU-only index (~200 MB instead of ~2 GB with CUDA).
COPY pyproject.toml ./
RUN python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb'))['project']; o=d['optional-dependencies']; \
print('\n'.join(d['dependencies'] + o['als'] + o['cache'] + o['broker'] + o['nosql']))" > requirements.txt \
    && pip install -r requirements.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY src ./src
COPY artifacts ./artifacts
# Spanish titles/posters and the evaluation reports shown in the metrics view.
COPY datasets/peliculas_es.csv ./datasets/
COPY reports/arranque_en_frio.json reports/evaluation_report.json ./reports/

# Accounts and the profiles of people who signed up in the demo (ids above the
# 610 MovieLens users) are personal data: they must not ship inside the image.
# The app recreates the empty accounts table on startup.
RUN python -c "import sqlite3; c = sqlite3.connect('artifacts/platform.db'); c.execute('DROP TABLE IF EXISTS cuentas'); c.execute('DELETE FROM user_profiles WHERE user_id > 610'); c.commit()"

# Writable runtime state goes to /app/data (mount a volume there to keep it).
ENV PYTHONPATH=/app/src \
    RECO_DOCUMENT_STORE_ROOT=/app/data/documents \
    RECO_RETRY_BUFFER_PATH=/app/data/retry_buffer.jsonl \
    RECO_ERROR_LOG_PATH=/app/data/logs/errors.jsonl \
    WORKERS=1

# The history file shipped in artifacts/ seeds the writable copy.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/documents /app/data/logs \
    && if [ -f artifacts/documents/interaction_history.jsonl ]; then \
         cp artifacts/documents/interaction_history.jsonl /app/data/documents/; \
       fi \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# application.server wraps uvicorn so that multi-worker mode keeps TCP_NODELAY
# on every connection (see its docstring: ~48 ms -> ~2 ms per request).
CMD ["python", "-m", "application.server"]
