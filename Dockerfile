FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/user/app

# Copy requirements first
COPY --chown=user requirements.txt .

# Install base dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY --chown=user . .

# 🔥 CRITICAL LINE (YOU WERE MISSING THIS)
RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]