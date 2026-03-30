# ── Build stage ────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  libpq5 curl && rm -rf /var/lib/apt/lists/*

# Run as non-root user (security best practice)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

COPY --from=builder /install /usr/local

COPY src/ src/
COPY alembic.ini .

# Copy model artifacts into the container
COPY src/models/artifacts/ src/models/artifacts/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Single entry point.  Override CMD for different modes:
#   API:    python -m src serve    (default)
#   Worker: python -m src work
#   Train:  python -m src train
CMD ["python", "-m", "src", "serve"]
