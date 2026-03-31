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
  libpq5 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY src/ src/
COPY alembic.ini .

# Copy model artifacts into the container
COPY src/models/artifacts/ src/models/artifacts/

EXPOSE 8000

# Single entry point.  Override CMD for different modes:
#   API:    python -m src serve    (default)
#   Worker: python -m src work
#   Train:  python -m src train
CMD ["python", "-m", "src", "serve"]
