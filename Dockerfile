# ──────────────────────────────────────────────────────────────────
#  EmpaChat Dockerfile
#  Multi-stage build for API + Demo service
# ──────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs outputs/mlruns

# ──────────────────────────────────────────────────────────────────
#  API Service
# ──────────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ──────────────────────────────────────────────────────────────────
#  Demo Service
# ──────────────────────────────────────────────────────────────────
FROM base AS demo

EXPOSE 7860

CMD ["python", "demo/gradio_app.py"]
