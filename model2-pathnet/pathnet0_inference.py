"""
PathNet0: Standalone Inference & Career Navigation Engine
Supports CLI arguments, interactive mode, and Python API import.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import train_pathnet0
from train_pathnet0 import (
    PathNetMultiTaskNet, RAGRoadmapEngine, PathNetInferenceEngine, PathNetFeaturePipeline,
    DEVICE, SENIORITY_MAP, ID_TO_TRANSITION_TYPE, TRANSITION_TYPES
)

# Alias for unpickling if trained under __main__
sys.modules['__main__'].PathNetFeaturePipeline = PathNetFeaturePipeline

def load_pathnet_system(model_dir=None):
    if model_dir is None:
        model_dir = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
        
    try:
        pipeline = joblib.load(os.path.join(model_dir, "pathnet0_feature_pipeline.joblib"))
        hgb_reg = joblib.load(os.path.join(model_dir, "pathnet0_hgb_regressor.joblib"))
        hgb_clf = joblib.load(os.path.join(model_dir, "pathnet0_hgb_classifier.joblib"))
        iso_calibrator_path = os.path.join(model_dir, "pathnet0_isotonic_calibrator.joblib")
        iso_calibrator = joblib.load(iso_calibrator_path) if os.path.exists(iso_calibrator_path) else None
    except (ModuleNotFoundError, AttributeError, ValueError) as exc:
        print(f"[PathNet0] Detected scikit-learn pickle version mismatch ({exc}); regenerating compatible artifacts in-place...")
        train_pathnet0.main()
        pipeline = joblib.load(os.path.join(model_dir, "pathnet0_feature_pipeline.joblib"))
        hgb_reg = joblib.load(os.path.join(model_dir, "pathnet0_hgb_regressor.joblib"))
        hgb_clf = joblib.load(os.path.join(model_dir, "pathnet0_hgb_classifier.joblib"))
        iso_calibrator_path = os.path.join(model_dir, "pathnet0_isotonic_calibrator.joblib")
        iso_calibrator = joblib.load(iso_calibrator_path) if os.path.exists(iso_calibrator_path) else None
    
    rag_kb = pd.read_parquet(os.path.join(model_dir, "pathnet0_rag_knowledge_base.parquet"))
    rag_engine = RAGRoadmapEngine(rag_kb)
    
    # Dynamically infer input_dim from saved state_dict or pipeline
    deep_weights_path = os.path.join(model_dir, "pathnet0_deep_model.pt")
    state_dict = torch.load(deep_weights_path, map_location=DEVICE)
    input_dim = state_dict['input_layer.0.weight'].shape[1]
    
    deep_model = PathNetMultiTaskNet(input_dim=input_dim, hidden_dim=256, num_types=len(TRANSITION_TYPES))
    deep_model.load_state_dict(state_dict)
    deep_model.to(DEVICE)
    deep_model.eval()
    
    engine = PathNetInferenceEngine(deep_model, pipeline, rag_engine, hgb_reg, hgb_clf, iso_calibrator)
    return engine

def format_prediction_card(res):
    print("=" * 70)
    print(f" 🚀 CAREER TRAJECTORY: {res['current_role'].upper()} ➔ {res['target_future_role'].upper()}")
    print("=" * 70)
    print(f" • Transition Feasibility : {res['feasibility_assessment']}")
    print(f" • Transition Probability : {res['transition_probability']}")
    print(f" • Transition Category    : {res['predicted_transition_type']}")
    print(f" • Seniority Path         : {res['current_seniority']} ➔ {res['target_seniority']} ({res['seniority_delta']} level change)")
    print(f" • Estimated Skill Gap    : {res['estimated_skill_gap_count']} critical skills required")
    print(f" • Confidence Distribution: {json.dumps(res['type_distribution'])}")
    print("-" * 70)
    print(" 📚 ACTIONABLE SKILL ROADMAP:")
    if isinstance(res['actionable_roadmap'], dict):
        for phase, skills in res['actionable_roadmap'].items():
            print(f"   ▶ {phase}:")
            for s in skills:
                print(f"      - {s}")
    else:
        print(f"   {res['actionable_roadmap']}")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PathNet0 Career Trajectory Inference CLI")
    parser.add_argument("--current", type=str, default=None, help="Current job title (e.g. 'Junior Software Engineer')")
    parser.add_argument("--tenure", type=float, default=2.5, help="Years of experience in current role")
    parser.add_argument("--target", type=str, default=None, help="Target future job title (e.g. 'Senior Software Engineer')")
    args = parser.parse_args()

    print("Loading PathNet0 Inference Engine...")
    engine = load_pathnet_system()
    print("PathNet0 successfully initialized!\n")
    
    if args.current and args.target:
        res = engine.predict_transition(
            current_role=args.current,
            tenure_years=args.tenure,
            target_role=args.target
        )
        format_prediction_card(res)
    else:
        # Demo 1: Vertical Promotion
        res_promo = engine.predict_transition(
            current_role="Junior Software Engineer",
            tenure_years=2.5,
            target_role="Senior Software Engineer"
        )
        format_prediction_card(res_promo)
        
        # Demo 2: Adjacent Pivot
        res_pivot = engine.predict_transition(
            current_role="Junior Frontend Developer",
            tenure_years=2.0,
            target_role="Full Stack Engineer"
        )
        format_prediction_card(res_pivot)
        
        # Demo 3: Career Exploration & Next Step Recommendations
        print("💡 Discovering Top 4 Career Next Steps for 'Data Analyst' (3.0 years tenure):")
        recommendations = engine.recommend_next_steps(
            current_role="Data Analyst",
            tenure_years=3.0,
            candidate_roles=[
                "Senior Data Analyst", "Data Scientist", "Machine Learning Engineer",
                "Product Manager", "Android Engineer"
            ],
            top_n=4
        )
        for i, rec in enumerate(recommendations, 1):
            print(f"  Rank #{i}: {rec['target_future_role']:<28} | Probability: {rec['transition_probability']:<8} | Type: {rec['predicted_transition_type']}")
