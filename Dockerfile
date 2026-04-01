# ── Build stage ────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY README.md pyproject.toml alembic.ini ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  libpq5 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app && chown -R app:app /app

COPY --from=builder /install /usr/local
COPY --chown=app:app alembic.ini .
COPY --chown=app:app src/ src/

USER app

EXPOSE 8000

# Single entry point.  Override CMD for different modes:
#   API:    python -m src serve    (default)
#   Worker: python -m src work
#   Train:  python -m src train
CMD ["python", "-m", "src", "serve"]
