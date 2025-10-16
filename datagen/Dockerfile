FROM python:3.11-slim

WORKDIR /app

# Copy only minimal requirements to avoid building the whole project (and Rust extension)
COPY datagen/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the required Python sources for runtime
COPY scripts/synthetic_data_gen.py /app/scripts/synthetic_data_gen.py
COPY datagen/main.py /app/datagen/main.py

ENV PYTHONUNBUFFERED=1

# Default command starts the API server
CMD ["python", "-m", "datagen.main"]


