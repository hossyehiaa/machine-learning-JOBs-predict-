"""
FitNet (Model 1) Serving API — Resume-to-Job & Career Matching Service
Built with FastAPI, Pydantic v2, Prometheus 0.0.4 text exposition format (/metrics),
explicit CORS allowlisting, optional X-API-Key authentication, per-IP rate limiting,
and sanitized error responses.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import hmac
import uuid
import time
import logging
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Security, status, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fitnet_inference import load_fitnet_system

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger("fitnet-api")

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    if o.strip() and o.strip() != "*"
]
ALLOWED_ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", "").strip() or None
REQUIRED_API_KEY = os.environ.get("FITNET_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

engine_instance = None
start_time = None
request_counter = 0
error_counter = 0
total_latency_ms = 0.0
rate_limit_buckets: Dict[str, deque] = defaultdict(deque)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance, start_time
    logger.info(f"Initializing FitNet inference engine from DATA_DIR={DATA_DIR}...")
    start_time = time.time()
    engine_instance = load_fitnet_system(DATA_DIR)
    logger.info("FitNet successfully loaded into memory.")
    yield
    logger.info("Shutting down FitNet inference engine...")

app = FastAPI(
    title="FitNet Resume-to-Job & Career Matching API (Model 1)",
    description="Calibrated resume-to-job matching, tier classification (Low/Medium/High), and missing skill-gap analysis.",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

def verify_api_key_and_rate_limit(request: Request, x_api_key: Optional[str] = Security(api_key_header)):
    configured_key = os.environ.get("FITNET_API_KEY", REQUIRED_API_KEY).strip()
    if configured_key:
        if not x_api_key or not hmac.compare_digest(x_api_key, configured_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key header.")

    client_ip = request.client.host if request.client else "local"
    now = time.time()
    window = rate_limit_buckets[client_ip]
    while window and (now - window[0]) > 60.0:
        window.popleft()
    limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", str(RATE_LIMIT_PER_MINUTE)))
    if len(window) >= limit:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    window.append(now)

class JobMatchRequest(BaseModel):
    current_role: str = Field(..., min_length=2, max_length=150)
    current_skills: List[str] = Field(..., min_length=1)
    tenure_years: float = Field(..., ge=0.0, le=60.0)
    target_role: str = Field(..., min_length=2, max_length=150)
    required_skills: List[str] = Field(..., min_length=1)
    job_description: Optional[str] = Field("", max_length=5000)

    @field_validator('current_role', 'target_role')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Role cannot be empty or whitespace.")
        return clean

@app.get("/health")
def health_check():
    uptime = time.time() - start_time if start_time else 0.0
    return {
        "status": "healthy" if engine_instance is not None else "degraded",
        "uptime_seconds": round(uptime, 2),
        "model_loaded": engine_instance is not None,
        "version": "1.1.0"
    }

@app.post("/v1/predict-fit", dependencies=[Depends(verify_api_key_and_rate_limit)])
def predict_fit(payload: JobMatchRequest):
    global request_counter, error_counter, total_latency_ms
    if engine_instance is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    t0 = time.time()
    try:
        res = engine_instance.predict_job_fit(
            current_role=payload.current_role,
            current_skills=payload.current_skills,
            tenure_years=payload.tenure_years,
            target_role=payload.target_role,
            required_skills=payload.required_skills,
            job_description=payload.job_description or ""
        )
        latency = (time.time() - t0) * 1000.0
        res["latency_ms"] = round(latency, 2)
        request_counter += 1
        total_latency_ms += latency
        return res
    except Exception:
        error_counter += 1
        req_id = uuid.uuid4().hex[:12]
        logger.error(f"Fit prediction failed [request_id={req_id}]", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal inference error (reference_id={req_id}).")

@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics():
    uptime = time.time() - start_time if start_time else 0.0
    model_status = 1 if engine_instance is not None else 0
    avg_latency = (total_latency_ms / request_counter) if request_counter > 0 else 0.0
    lines = [
        "# HELP fitnet_requests_total Total number of successful job-fit requests.",
        "# TYPE fitnet_requests_total counter",
        f"fitnet_requests_total {request_counter}",
        "# HELP fitnet_errors_total Total number of failed job-fit requests.",
        "# TYPE fitnet_errors_total counter",
        f"fitnet_errors_total {error_counter}",
        "# HELP fitnet_uptime_seconds Service uptime in seconds.",
        "# TYPE fitnet_uptime_seconds gauge",
        f"fitnet_uptime_seconds {uptime:.2f}",
        "# HELP fitnet_model_loaded Whether FitNet model ensemble is loaded (1=loaded, 0=unloaded).",
        "# TYPE fitnet_model_loaded gauge",
        f"fitnet_model_loaded {model_status}",
        "# HELP fitnet_inference_latency_ms_avg Average inference latency in milliseconds.",
        "# TYPE fitnet_inference_latency_ms_avg gauge",
        f"fitnet_inference_latency_ms_avg {avg_latency:.2f}",
        ""
    ]
    return PlainTextResponse(content="\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")
