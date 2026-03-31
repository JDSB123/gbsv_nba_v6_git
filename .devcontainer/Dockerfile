FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libpq-dev \
  git \
  curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the project and dev tooling from pyproject.toml.
# The container is the Python environment; no local venv is required.
COPY README.md pyproject.toml alembic.ini ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

# Keep container running for dev
CMD ["sleep", "infinity"]
