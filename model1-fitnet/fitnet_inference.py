"""
FitNet (Model 1): Standalone Resume-to-Job & Career Matching Inference Engine
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import argparse
import joblib
import torch
import train_fitnet
from train_fitnet import (
    FitNetMultiTaskNet, FitNetFeaturePipeline, FitNetInferenceEngine,
    DEVICE, MATCH_LABELS
)

sys.modules['__main__'].FitNetFeaturePipeline = FitNetFeaturePipeline

def load_fitnet_system(model_dir=None):
    if model_dir is None:
        model_dir = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

    try:
        pipeline = joblib.load(os.path.join(model_dir, "fitnet_feature_pipeline.joblib"))
        hgb_reg = joblib.load(os.path.join(model_dir, "fitnet_hgb_regressor.joblib"))
        hgb_clf = joblib.load(os.path.join(model_dir, "fitnet_hgb_classifier.joblib"))
        cal_path = os.path.join(model_dir, "fitnet_isotonic_calibrator.joblib")
        calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
    except (ModuleNotFoundError, AttributeError, ValueError) as exc:
        print(f"[FitNet] Detected pickle version mismatch ({exc}); regenerating compatible artifacts in-place...")
        train_fitnet.main()
        pipeline = joblib.load(os.path.join(model_dir, "fitnet_feature_pipeline.joblib"))
        hgb_reg = joblib.load(os.path.join(model_dir, "fitnet_hgb_regressor.joblib"))
        hgb_clf = joblib.load(os.path.join(model_dir, "fitnet_hgb_classifier.joblib"))
        cal_path = os.path.join(model_dir, "fitnet_isotonic_calibrator.joblib")
        calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None

    state_dict = torch.load(os.path.join(model_dir, "fitnet_deep_model.pt"), map_location=DEVICE)
    input_dim = state_dict['input_layer.0.weight'].shape[1]

    deep_model = FitNetMultiTaskNet(input_dim=input_dim, hidden_dim=256, num_classes=len(MATCH_LABELS)).to(DEVICE)
    deep_model.load_state_dict(state_dict)
    deep_model.eval()

    return FitNetInferenceEngine(deep_model, pipeline, hgb_reg, hgb_clf, calibrator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FitNet Resume-to-Job Matching CLI")
    parser.add_argument("--current_role", type=str, default="Python Developer")
    parser.add_argument("--current_skills", type=str, default="python,django,postgresql,docker,git,rest apis")
    parser.add_argument("--tenure", type=float, default=3.5)
    parser.add_argument("--target_role", type=str, default="Senior Backend Python Engineer")
    parser.add_argument("--required_skills", type=str, default="python,django,postgresql,docker,kubernetes,redis")
    args = parser.parse_args()

    engine = load_fitnet_system()
    res = engine.predict_job_fit(
        current_role=args.current_role,
        current_skills=args.current_skills,
        tenure_years=args.tenure,
        target_role=args.target_role,
        required_skills=args.required_skills
    )
    print(json.dumps(res, indent=2))
