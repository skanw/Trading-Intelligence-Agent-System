# Multi-stage Production Dockerfile for Trading Intelligence Agent System

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib for technical analysis
RUN curl -L https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download -o ta-lib.tar.gz \
    && tar -xzf ta-lib.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib.tar.gz

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

CMD ["python", "-m", "src.realtime.orchestrator"]

# Production stage
FROM base as production

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading trading

# Install production dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy source code
COPY --chown=trading:trading . .

# Install package
RUN pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R trading:trading /app

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Default command
CMD ["python", "-m", "src.realtime.orchestrator"]

# Production optimized stage
FROM production as production-optimized

# Switch back to root for optimization
USER root

# Remove unnecessary packages and files
RUN apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache \
    && find /usr/local/lib/python3.11 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11 -name "__pycache__" -delete

# Switch back to non-root user
USER trading

# Expose common ports (will be overridden by docker-compose)
EXPOSE 8000 8501

# Use exec form for better signal handling
ENTRYPOINT ["python", "-m"]
CMD ["src.realtime.orchestrator"] 