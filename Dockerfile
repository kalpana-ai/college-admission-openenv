# Base Image
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
        "openenv-core[core]>=0.2.1" \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.24.0" \
        "gradio>=5.0.0,<7.0.0" \
        "openai>=1.0.0" \
        "groq>=0.9.0" \
        "python-dotenv>=1.0.0" \
        "pillow>=10.0.0" \
        "requests>=2.31.0"

# Copy full project
COPY . /app

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose BOTH ports (important)
# EXPOSE 8000
EXPOSE 7860

# Healthcheck (use API port — required for OpenEnv)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Default command → run FastAPI (OpenEnv compliant)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]