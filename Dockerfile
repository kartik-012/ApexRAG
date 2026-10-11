# ==============================================================================
# ApexRAG Production Dockerfile — High-Throughput Enterprise RAG Microservice
# Multi-stage optimized build with non-root security execution & healthcheck
# ==============================================================================

# ----------------- Stage 1: Build Dependencies -----------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# ----------------- Stage 2: Production Runtime -----------------
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/home/apexuser/.local/bin:$PATH \
    PORT=8000

# Install runtime curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-privileged security user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash apexuser

# Copy installed wheels/packages from builder
COPY --from=builder --chown=apexuser:appgroup /root/.local /home/apexuser/.local

# Copy application code and data
COPY --chown=apexuser:appgroup src/ /app/src/
COPY --chown=apexuser:appgroup data/ /app/data/
COPY --chown=apexuser:appgroup pytest.ini /app/
COPY --chown=apexuser:appgroup .env.example /app/.env

# Create directories for persistent runtime logs, drift, feedback
RUN mkdir -p /app/data/feedback /app/data/logs /app/reports && \
    chown -R apexuser:appgroup /app

USER apexuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "65"]
