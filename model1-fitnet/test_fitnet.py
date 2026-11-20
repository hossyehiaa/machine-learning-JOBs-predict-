"""
Automated Verification Suite for FitNet (Model 1: Resume-to-Job Matching)
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pyarrow.parquet as pq
from fastapi.testclient import TestClient
import app_fitnet

def run_tests():
    print("=" * 75)
    print(" 🧪 RUNNING FITNET (MODEL 1) API, SECURITY & 221,903-ROW DATA TEST SUITE")
    print("=" * 75)

    with TestClient(app_fitnet.app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200 and r_health.json()["status"] == "healthy"
        print(" ✅ [PASS] GET /health -> 200 OK", r_health.json())

        r_fit = client.post("/v1/predict-fit", json={
            "current_role": "Python Developer",
            "current_skills": ["python", "django", "postgresql", "docker", "git", "rest apis"],
            "tenure_years": 3.5,
            "target_role": "Senior Backend Python Engineer",
            "required_skills": ["python", "django", "postgresql", "docker", "kubernetes", "redis"],
            "job_description": "Building scalable Python backend services with Django, PostgreSQL, and Docker."
        })
        assert r_fit.status_code == 200
        fit_data = r_fit.json()
        assert fit_data["predicted_match_tier"] in ("Low", "Medium", "High")
        print(f" ✅ [PASS] POST /v1/predict-fit -> 200 OK (Tier: {fit_data['predicted_match_tier']}, Fit Score: {fit_data['fit_score_percentage']})")

        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200 and "# HELP fitnet_requests_total" in r_metrics.text
        print(" ✅ [PASS] GET /metrics -> 200 OK (Prometheus 0.0.4 Text Exposition Format)")

        r_inv = client.post("/v1/predict-fit", json={
            "current_role": "   ",
            "current_skills": [],
            "tenure_years": -2.0,
            "target_role": "Engineer",
            "required_skills": ["python"]
        })
        assert r_inv.status_code == 422
        print(" ✅ [PASS] Input Validation -> 422 Unprocessable Entity on invalid payload")

        os.environ["FITNET_API_KEY"] = "fitnet-secret-key"
        try:
            assert client.post("/v1/predict-fit", json={
                "current_role": "Python Developer", "current_skills": ["python"], "tenure_years": 2.0,
                "target_role": "Backend Engineer", "required_skills": ["python"]
            }).status_code == 401
            assert client.post("/v1/predict-fit", headers={"X-API-Key": "fitnet-secret-key"}, json={
                "current_role": "Python Developer", "current_skills": ["python"], "tenure_years": 2.0,
                "target_role": "Backend Engineer", "required_skills": ["python"]
            }).status_code == 200
            print(" ✅ [PASS] X-API-Key Authentication -> 401 without key, 200 OK with valid key")
        finally:
            os.environ.pop("FITNET_API_KEY", None)

        os.environ["RATE_LIMIT_PER_MINUTE"] = "2"
        try:
            assert client.post("/v1/predict-fit", json={
                "current_role": "Python Developer", "current_skills": ["python"], "tenure_years": 2.0,
                "target_role": "Backend Engineer", "required_skills": ["python"]
            }).status_code == 429
            print(" ✅ [PASS] Per-IP Rate Limiting -> 429 Too Many Requests when threshold exceeded")
        finally:
            os.environ["RATE_LIMIT_PER_MINUTE"] = "60"
            app_fitnet.rate_limit_buckets.clear()

        orig_fn = app_fitnet.engine_instance.predict_job_fit
        def broken_fn(*args, **kwargs):
            raise RuntimeError("SENSITIVE_INTERNAL_SECRET_PATH")
        app_fitnet.engine_instance.predict_job_fit = broken_fn
        try:
            r_err = client.post("/v1/predict-fit", json={
                "current_role": "Python Developer", "current_skills": ["python"], "tenure_years": 2.0,
                "target_role": "Backend Engineer", "required_skills": ["python"]
            })
            assert r_err.status_code == 500
            assert "SENSITIVE_INTERNAL_SECRET_PATH" not in r_err.json()["detail"]
            print(f" ✅ [PASS] Sanitized Error Handling -> 500 returns safe message: '{r_err.json()['detail']}'")
        finally:
            app_fitnet.engine_instance.predict_job_fit = orig_fn

        meta = pq.read_metadata(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitnet_training_features.parquet"))
        assert meta.num_rows == 221903, f"Expected 221903 rows, got {meta.num_rows}"
        print(f" ✅ [PASS] Full-Scale FitNet Parquet Provenance -> ({meta.num_rows:,}, {meta.num_columns})")

    print("=" * 75)
    print(" 🎉 ALL FITNET API, SECURITY & 221,903-ROW DATA CHECKS PASSED!")
    print("=" * 75)

if __name__ == '__main__':
    run_tests()
