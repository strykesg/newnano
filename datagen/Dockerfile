FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata for dependency install
COPY pyproject.toml uv.lock /app/

# Install uv to resolve and install deps
RUN pip install --no-cache-dir uv==0.4.20

# Install project dependencies (only runtime)
RUN uv pip install --system .

# Copy source needed for generator and server
COPY scripts/synthetic_data_gen.py /app/scripts/synthetic_data_gen.py
COPY datagen/main.py /app/datagen/main.py

ENV PYTHONUNBUFFERED=1

# Default command starts the API server
CMD ["python", "-m", "datagen.main"]


