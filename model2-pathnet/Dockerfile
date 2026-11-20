FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Install system dependencies required for OpenMP / scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deterministically pinned Python dependencies (including scikit-learn==1.7.2 and httpx==0.28.1)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, test suite, and serialized model artifacts
COPY train_pathnet0.py pathnet0_inference.py calibration_and_fairness.py app.py test_pathnet0.py ./
COPY pathnet0_*.pt pathnet0_*.joblib pathnet0_*.parquet ./

# Run as non-root user for container security
RUN useradd -m -u 1000 pathnetuser && chown -R pathnetuser:pathnetuser /app
USER pathnetuser

ENV PORT=8001
EXPOSE 8001

# Single worker per container replica avoids duplicating in-memory PyTorch & Gradient Boosted Tree checkpoints.
# Scale horizontally via container orchestrator (Render / Fly.io / Railway / Kubernetes replicas).
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1"]
