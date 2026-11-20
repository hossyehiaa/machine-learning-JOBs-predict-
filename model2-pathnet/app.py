"""
PathNet0 Serving API (Production-Shaped Prototype)
Built with FastAPI, Pydantic v2, Prometheus text exposition format (/metrics),
explicit CORS allowlisting, optional X-API-Key authentication, per-IP rate limiting,
and sanitized error handling.
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

# Ensure model directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pathnet0_inference import load_pathnet_system

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger("pathnet0-api")

# Security & Rate-Limiting Configuration via Environment Variables
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    if origin.strip() and origin.strip() != "*"
]
ALLOWED_ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", "").strip() or None
REQUIRED_API_KEY = os.environ.get("PATHNET_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global Inference Engine & Telemetry State
engine_instance = None
start_time = None
request_counter = 0
error_counter = 0
total_latency_ms = 0.0
rate_limit_buckets: Dict[str, deque] = defaultdict(deque)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance, start_time
    logger.info(f"Initializing PathNet0 inference engine from DATA_DIR={DATA_DIR}...")
    start_time = time.time()
    try:
        engine_instance = load_pathnet_system(DATA_DIR)
        logger.info("PathNet0 successfully loaded into memory and warm for inference.")
    except Exception as e:
        logger.error(f"Failed to load PathNet0 model artifacts: {e}", exc_info=True)
        raise e
    yield
    logger.info("Shutting down PathNet0 inference engine...")

app = FastAPI(
    title="PathNet0 Career Trajectory & Path Network API",
    description=(
        "Production-shaped ML prototype service for career progression feasibility modeling, "
        "transition category classification, calibrated probabilities, and TF-IDF skill-roadmap retrieval."
    ),
    version="1.1.0",
    lifespan=lifespan
)

# Hardened CORS policy: explicit origin allowlist, never combining wildcard origins with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# ---------------------------------------------------------------------------
# Security & Rate Limiting Dependencies
# ---------------------------------------------------------------------------
def verify_api_key_and_rate_limit(
    request: Request,
    x_api_key: Optional[str] = Security(api_key_header)
):
    """
    1. Enforces X-API-Key authentication when PATHNET_API_KEY is configured.
    2. Enforces per-IP sliding-window rate limiting (RATE_LIMIT_PER_MINUTE).
    """
    configured_key = os.environ.get("PATHNET_API_KEY", REQUIRED_API_KEY).strip()
    if configured_key:
        if not x_api_key or not hmac.compare_digest(x_api_key, configured_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header."
            )

    client_ip = request.client.host if request.client else "local"
    now = time.time()
    window = rate_limit_buckets[client_ip]
    while window and (now - window[0]) > 60.0:
        window.popleft()

    limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", str(RATE_LIMIT_PER_MINUTE)))
    if len(window) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry after 60 seconds."
        )
    window.append(now)

# ---------------------------------------------------------------------------
# Request & Response Pydantic Schemas
# ---------------------------------------------------------------------------
class TransitionRequest(BaseModel):
    current_role: str = Field(..., min_length=2, max_length=150, examples=["Junior Software Engineer"], description="Current job title")
    tenure_years: float = Field(..., ge=0.0, le=50.0, examples=[2.5], description="Years of experience in current position")
    target_role: str = Field(..., min_length=2, max_length=150, examples=["Senior Software Engineer"], description="Target future job title")
    step_index: Optional[int] = Field(2, ge=1, le=20, description="Career ladder transition step sequence")

    @field_validator('current_role', 'target_role')
    @classmethod
    def validate_role_name(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Role title cannot be empty or purely whitespace.")
        return clean

class TransitionResponse(BaseModel):
    current_role: str
    current_seniority: str
    tenure_years: float
    target_future_role: str
    target_seniority: str
    seniority_delta: str
    transition_probability: str
    transition_probability_raw: float
    predicted_transition_type: str
    type_distribution: Dict[str, str]
    estimated_skill_gap_count: int
    feasibility_assessment: str
    actionable_roadmap: Any
    latency_ms: float

class RecommendationRequest(BaseModel):
    current_role: str = Field(..., min_length=2, max_length=150, examples=["Data Analyst"])
    tenure_years: float = Field(..., ge=0.0, le=50.0, examples=[3.0])
    candidate_roles: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific roles to rank. If omitted, uses default diverse pool."
    )
    top_n: int = Field(3, ge=1, le=10, description="Number of top transitions to recommend")

class RecommendationResponse(BaseModel):
    current_role: str
    tenure_years: float
    recommendations: List[TransitionResponse]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    version: str

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Liveness & Readiness probe for container orchestrators."""
    uptime = time.time() - start_time if start_time else 0.0
    return {
        "status": "healthy" if engine_instance is not None else "degraded",
        "uptime_seconds": round(uptime, 2),
        "model_loaded": engine_instance is not None,
        "version": "1.1.0"
    }

@app.post(
    "/v1/predict-transition",
    response_model=TransitionResponse,
    dependencies=[Depends(verify_api_key_and_rate_limit)],
    tags=["Career Modeling"]
)
def predict_transition(payload: TransitionRequest):
    """
    Evaluates career transition feasibility between current_role and target_role.
    Returns isotonic-calibrated probability, transition category, and TF-IDF retrieved roadmap.
    """
    global request_counter, error_counter, total_latency_ms
    if engine_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized or still warming up."
        )
    
    t0 = time.time()
    try:
        res = engine_instance.predict_transition(
            current_role=payload.current_role,
            tenure_years=payload.tenure_years,
            target_role=payload.target_role,
            step_index=payload.step_index
        )
        latency = (time.time() - t0) * 1000.0
        res["latency_ms"] = round(latency, 2)
        request_counter += 1
        total_latency_ms += latency
        return res
    except Exception:
        error_counter += 1
        req_id = uuid.uuid4().hex[:12]
        logger.error(f"Prediction failed [request_id={req_id}] for input {payload}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference error (reference_id={req_id})."
        )

@app.post(
    "/v1/recommend-next-steps",
    response_model=RecommendationResponse,
    dependencies=[Depends(verify_api_key_and_rate_limit)],
    tags=["Career Modeling"]
)
def recommend_next_steps(payload: RecommendationRequest):
    """
    Recommends top feasible career trajectories from current position.
    """
    global request_counter, error_counter, total_latency_ms
    if engine_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized."
        )
        
    t0 = time.time()
    try:
        recs = engine_instance.recommend_next_steps(
            current_role=payload.current_role,
            tenure_years=payload.tenure_years,
            candidate_roles=payload.candidate_roles,
            top_n=payload.top_n
        )
        latency = (time.time() - t0) * 1000.0
        for r in recs:
            r["latency_ms"] = round(latency / max(1, len(recs)), 2)
            
        request_counter += 1
        total_latency_ms += latency
        return {
            "current_role": payload.current_role,
            "tenure_years": payload.tenure_years,
            "recommendations": recs,
            "latency_ms": round(latency, 2)
        }
    except Exception:
        error_counter += 1
        req_id = uuid.uuid4().hex[:12]
        logger.error(f"Recommendation failed [request_id={req_id}] for input {payload}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal recommendation error (reference_id={req_id})."
        )

class RagSearchRequest(BaseModel):
    query_role: str = Field(..., min_length=2, max_length=150, examples=["Senior Android Engineer"])
    query_description: Optional[str] = Field("", max_length=2000)
    top_k: int = Field(3, ge=1, le=10)

@app.post(
    "/v1/retrieve-roadmaps",
    dependencies=[Depends(verify_api_key_and_rate_limit)],
    tags=["RAG Knowledge Base"]
)
def retrieve_roadmaps(payload: RagSearchRequest):
    """
    Directly queries the 230,333-row TF-IDF RAG Knowledge Base for top-K matching career roadmaps.
    """
    global request_counter, error_counter, total_latency_ms
    if engine_instance is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")
    t0 = time.time()
    try:
        import json as _json
        df_res = engine_instance.rag_engine.retrieve(
            payload.query_role,
            payload.query_description or "",
            top_k=payload.top_k
        )
        matches = []
        for _, row in df_res.iterrows():
            raw_rd = row["skill_gap_roadmap"]
            try:
                parsed_rd = _json.loads(raw_rd) if isinstance(raw_rd, str) else raw_rd
            except Exception:
                parsed_rd = raw_rd
            matches.append({
                "target_future_role": str(row["target_future_role"]),
                "target_job_description": str(row["target_job_description"]),
                "skill_gap_count": int(row["skill_gap_count"]),
                "similarity": round(float(row["similarity"]), 4),
                "skill_gap_roadmap": parsed_rd
            })
        latency = (time.time() - t0) * 1000.0
        request_counter += 1
        total_latency_ms += latency
        return {
            "query_role": payload.query_role,
            "total_indexed_roadmaps": len(engine_instance.rag_engine.rag_kb),
            "matches": matches,
            "latency_ms": round(latency, 2)
        }
    except Exception:
        error_counter += 1
        req_id = uuid.uuid4().hex[:12]
        logger.error(f"RAG retrieval failed [request_id={req_id}]", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal RAG retrieval error (reference_id={req_id}).")

@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["System"],
    summary="Prometheus Text Exposition Format Metrics"
)
def get_metrics():
    """
    Exposes metrics in standard Prometheus text exposition format (0.0.4).
    """
    uptime = time.time() - start_time if start_time else 0.0
    model_status = 1 if engine_instance is not None else 0
    avg_latency = (total_latency_ms / request_counter) if request_counter > 0 else 0.0
    lines = [
        "# HELP pathnet0_requests_total Total number of successful inference requests.",
        "# TYPE pathnet0_requests_total counter",
        f"pathnet0_requests_total {request_counter}",
        "# HELP pathnet0_errors_total Total number of failed inference requests.",
        "# TYPE pathnet0_errors_total counter",
        f"pathnet0_errors_total {error_counter}",
        "# HELP pathnet0_uptime_seconds Service uptime in seconds.",
        "# TYPE pathnet0_uptime_seconds gauge",
        f"pathnet0_uptime_seconds {uptime:.2f}",
        "# HELP pathnet0_model_loaded Whether the PathNet0 model ensemble is loaded in memory (1=loaded, 0=unloaded).",
        "# TYPE pathnet0_model_loaded gauge",
        f"pathnet0_model_loaded {model_status}",
        "# HELP pathnet0_inference_latency_ms_avg Average inference latency in milliseconds.",
        "# TYPE pathnet0_inference_latency_ms_avg gauge",
        f"pathnet0_inference_latency_ms_avg {avg_latency:.2f}",
        ""
    ]
    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
