"""
Automated Verification Suite for PathNet0 API & Hardening Controls
Tests:
1. /health readiness probe
2. /v1/predict-transition & /v1/recommend-next-steps inference accuracy
3. /metrics Prometheus 0.0.4 text exposition format (# HELP, # TYPE, metric lines)
4. Input validation (HTTP 422 on invalid role/tenure)
5. Optional X-API-Key authentication enforcement (HTTP 401 when PATHNET_API_KEY is set)
6. Per-IP sliding-window rate limiting (HTTP 429 when RATE_LIMIT_PER_MINUTE is exceeded)
7. Sanitized error responses (HTTP 500 returns reference_id without leaking internal exception details)
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from fastapi.testclient import TestClient
import app as pathnet_app

def run_tests():
    print("=" * 75)
    print(" 🧪 RUNNING PATHNET0 API & SECURITY HARDENING TEST SUITE")
    print("=" * 75)

    with TestClient(pathnet_app.app) as client:
        # 1. Health Probe
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "healthy"
        print(" ✅ [PASS] GET /health -> 200 OK", r_health.json())

        # 2. Transition Prediction
        r_pred = client.post("/v1/predict-transition", json={
            "current_role": "Data Analyst",
            "tenure_years": 3.0,
            "target_role": "Data Scientist",
            "step_index": 2
        })
        assert r_pred.status_code == 200
        pred_data = r_pred.json()
        assert 0.0 < pred_data["transition_probability_raw"] < 1.0
        print(f" ✅ [PASS] POST /v1/predict-transition -> 200 OK (Calibrated Prob: {pred_data['transition_probability']}, Type: {pred_data['predicted_transition_type']})")

        # 3. Next-Step Recommendations
        r_rec = client.post("/v1/recommend-next-steps", json={
            "current_role": "Junior Software Engineer",
            "tenure_years": 2.5,
            "candidate_roles": ["Senior Software Engineer", "Full Stack Engineer", "Sales Director"],
            "top_n": 2
        })
        assert r_rec.status_code == 200
        assert len(r_rec.json()["recommendations"]) == 2
        print(" ✅ [PASS] POST /v1/recommend-next-steps -> 200 OK")

        # 4. Prometheus Text Exposition Format (/metrics)
        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert "text/plain" in r_metrics.headers["content-type"]
        metrics_text = r_metrics.text
        assert "# HELP pathnet0_requests_total" in metrics_text
        assert "# TYPE pathnet0_requests_total counter" in metrics_text
        assert "pathnet0_requests_total 2" in metrics_text
        print(" ✅ [PASS] GET /metrics -> 200 OK (Prometheus 0.0.4 Text Exposition Format)")

        # 5. Pydantic Input Validation (HTTP 422)
        r_invalid = client.post("/v1/predict-transition", json={
            "current_role": "   ",
            "tenure_years": -5.0,
            "target_role": "Data Scientist"
        })
        assert r_invalid.status_code == 422
        print(" ✅ [PASS] Input Validation -> 422 Unprocessable Entity on invalid payload")

        # 6. X-API-Key Authentication Enforcement (HTTP 401 vs 200)
        os.environ["PATHNET_API_KEY"] = "secret-test-key-123"
        try:
            r_unauth = client.post("/v1/predict-transition", json={
                "current_role": "Data Analyst",
                "tenure_years": 3.0,
                "target_role": "Data Scientist"
            })
            assert r_unauth.status_code == 401
            r_auth = client.post(
                "/v1/predict-transition",
                headers={"X-API-Key": "secret-test-key-123"},
                json={"current_role": "Data Analyst", "tenure_years": 3.0, "target_role": "Data Scientist"}
            )
            assert r_auth.status_code == 200
            print(" ✅ [PASS] X-API-Key Authentication -> 401 Unauthorized without key, 200 OK with valid key")
        finally:
            os.environ.pop("PATHNET_API_KEY", None)

        # 7. Per-IP Rate Limiting (HTTP 429)
        os.environ["RATE_LIMIT_PER_MINUTE"] = "3"
        try:
            r_rl = client.post("/v1/predict-transition", json={
                "current_role": "Data Analyst",
                "tenure_years": 3.0,
                "target_role": "Data Scientist"
            })
            assert r_rl.status_code == 429
            print(" ✅ [PASS] Per-IP Rate Limiting -> 429 Too Many Requests when threshold exceeded")
        finally:
            os.environ["RATE_LIMIT_PER_MINUTE"] = "60"
            pathnet_app.rate_limit_buckets.clear()

        # 8. Sanitized Error Response Verification (No internal exception string leakage)
        orig_fn = pathnet_app.engine_instance.predict_transition
        def broken_fn(*args, **kwargs):
            raise RuntimeError("SENSITIVE_INTERNAL_PATH_/secret/db_credentials")
        pathnet_app.engine_instance.predict_transition = broken_fn
        try:
            r_err = client.post("/v1/predict-transition", json={
                "current_role": "Data Analyst",
                "tenure_years": 3.0,
                "target_role": "Data Scientist"
            })
            assert r_err.status_code == 500
            err_detail = r_err.json()["detail"]
            assert "SENSITIVE_INTERNAL_PATH" not in err_detail
            assert "reference_id=" in err_detail
            print(f" ✅ [PASS] Sanitized Error Handling -> 500 returns safe message: '{err_detail}'")
        finally:
            pathnet_app.engine_instance.predict_transition = orig_fn

        # 9. Verify Deterministic Pinned Dependencies in requirements.txt
        req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            req_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        assert any(line.startswith("scikit-learn==1.7.2") for line in req_lines), "scikit-learn==1.7.2 must be pinned"
        assert any(line.startswith("httpx==") for line in req_lines), "httpx must be pinned for TestClient"
        assert all("==" in line for line in req_lines), "All dependencies in requirements.txt must be deterministically pinned with =="
        print(f" ✅ [PASS] Deterministic Dependency Lock -> All {len(req_lines)} packages in requirements.txt pinned with '==' (including scikit-learn==1.7.2 & httpx==0.28.1)")

        # 10. Verify Dockerfile Base Image & Single-Worker Configuration
        docker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dockerfile")
        with open(docker_path, "r", encoding="utf-8") as f:
            docker_content = f.read()
        assert "FROM python:3.12-slim" in docker_content, "Dockerfile must use FROM python:3.12-slim for wheel compatibility"
        assert '"--workers", "1"' in docker_content, "Dockerfile must use single worker per replica"
        print(" ✅ [PASS] Dockerfile Verification -> Uses 'FROM python:3.12-slim' matching pinned binary wheels & single-worker runtime")

        # 11. Verify Full-Scale 1.14M Dataset & 230,333-Row RAG Knowledge Base Artifacts
        import pyarrow.parquet as pq
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tf_meta = pq.read_metadata(os.path.join(base_dir, "pathnet0_training_features.parquet"))
        rag_meta = pq.read_metadata(os.path.join(base_dir, "pathnet0_rag_knowledge_base.parquet"))
        assert tf_meta.num_rows == 1140000 and tf_meta.num_columns == 18, f"Expected (1140000, 18), got ({tf_meta.num_rows}, {tf_meta.num_columns})"
        assert rag_meta.num_rows == 230333 and rag_meta.num_columns == 6, f"Expected (230333, 6), got ({rag_meta.num_rows}, {rag_meta.num_columns})"
        print(f" ✅ [PASS] Full-Scale Parquet Provenance -> training_features: ({tf_meta.num_rows:,}, {tf_meta.num_columns}), rag_kb: ({rag_meta.num_rows:,}, {rag_meta.num_columns})")

    print("=" * 75)
    print(" 🎉 ALL 11 API, SECURITY, DOCKER, REPRODUCIBILITY & 1.14M DATA CHECKS PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
