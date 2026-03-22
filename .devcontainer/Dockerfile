FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libpq-dev \
  git \
  curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install ALL deps (prod + dev/test) inside the container.
# The container IS the virtual environment — no venv needed.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Keep container running for dev
CMD ["sleep", "infinity"]
