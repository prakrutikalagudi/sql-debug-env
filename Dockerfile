# SQLDebugEnv — HuggingFace Spaces Dockerfile
FROM python:3.11-slim

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/user/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=user . .

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
