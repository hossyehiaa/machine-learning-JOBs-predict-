FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY train_fitnet.py fitnet_inference.py calibration_and_fairness_fitnet.py app_fitnet.py test_fitnet.py ./
COPY fitnet_*.pt fitnet_*.joblib fitnet_*.parquet ./

RUN useradd -m -u 1000 fitnetuser && chown -R fitnetuser:fitnetuser /app
USER fitnetuser

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app_fitnet:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
