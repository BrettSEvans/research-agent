# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps: poppler for pdf2image, build tools for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -v -r requirements.txt

# Copy application code
COPY . .

# ── Data directories on the persistent volume ────────────────────────────────
# Railway mounts a persistent disk at /data.  We symlink the runtime dirs there
# so uploads, contexts, and saved artefacts survive deploys.
# The entrypoint script handles the actual mkdir + symlink at runtime.
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "8000"]
