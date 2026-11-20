"""
PathNet0: Probability Calibration & Algorithmic Fairness Audit Module
Evaluates Uncalibrated vs. Isotonic-Calibrated ECE and Brier Score on the strictly held-out
GroupShuffleSplit test set, alongside Subgroup Error Parity & Disparate Impact (EEOC 80% Rule)
across Occupational Sectors, Tenure Brackets, and Seniority Tiers.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    brier_score_loss, mean_absolute_error, accuracy_score, f1_score
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pathnet0_inference import load_pathnet_system
from train_pathnet0 import TRANSITION_TYPE_TO_ID, infer_primary_sector

def compute_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) across uniform confidence bins.
    ECE = sum_{b=1}^B (|B_b| / N) * |empirical_mean(B_b) - predicted_mean(B_b)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            avg_conf = np.mean(y_prob[mask])
            avg_true = np.mean(y_true[mask])
            abs_err = np.abs(avg_true - avg_conf)
            ece += (n_in_bin / len(y_true)) * abs_err
            bin_details.append({
                'bin': f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                'count': int(n_in_bin),
                'mean_pred': round(float(avg_conf), 4),
                'empirical_prob': round(float(avg_true), 4),
                'abs_error': round(float(abs_err), 4)
            })
            
    return float(ece), pd.DataFrame(bin_details)

def audit_subgroup_fairness_and_parity(df_test, y_prob_true, y_prob_pred, y_type_true, y_type_pred, threshold=0.35):
    """
    Audits Subgroup Error Parity (MAE, Classification Accuracy, Macro F1) and
    Disparate Impact Ratio (EEOC 80% rule) across:
      1. Occupational Sectors
      2. Experience Tenure Brackets
      3. Current Seniority Levels
    """
    df = df_test.copy().reset_index(drop=True)
    df['y_prob_true'] = y_prob_true
    df['y_prob_pred'] = y_prob_pred
    df['y_type_true'] = y_type_true
    df['y_type_pred'] = y_type_pred
    df['favorable_outcome'] = (df['y_prob_pred'] >= threshold).astype(int)
    
    unique_cur = df['current_role'].astype(str).unique()
    sector_map = {r: infer_primary_sector(r) for r in unique_cur}
    df['occupational_sector'] = df['current_role'].astype(str).map(sector_map)
    df['tenure_bracket'] = pd.cut(
        df['tenure_years'],
        bins=[-0.1, 1.5, 3.5, 6.0, 50.0],
        labels=['<1.5 yrs (Early)', '1.5-3.5 yrs (Mid)', '3.5-6.0 yrs (Experienced)', '6.0+ yrs (Veteran)']
    )
    
    print("\n" + "=" * 85)
    print(" ⚖️ MULTI-DIMENSIONAL SUBGROUP ERROR PARITY & DISPARATE IMPACT AUDIT (HELD-OUT TEST)")
    print("=" * 85)
    
    audit_dimensions = [
        ("Occupational Sector", "occupational_sector"),
        ("Experience Tenure Bracket", "tenure_bracket"),
        ("Current Seniority Tier", "current_seniority")
    ]
    
    results = {}
    for title, col in audit_dimensions:
        print(f"\n[Audit Dimension: {title}]")
        rows = []
        for group_val, grp in df.groupby(col, observed=False):
            if len(grp) < 5:
                continue
            mae = mean_absolute_error(grp['y_prob_true'], grp['y_prob_pred'])
            acc = accuracy_score(grp['y_type_true'], grp['y_type_pred'])
            f1 = f1_score(grp['y_type_true'], grp['y_type_pred'], average='macro', zero_division=0)
            fav_rate = grp['favorable_outcome'].mean()
            rows.append({
                'Subgroup': str(group_val),
                'Count': int(len(grp)),
                'Prob_MAE': round(float(mae), 4),
                'Type_Acc': round(float(acc), 4),
                'Macro_F1': round(float(f1), 4),
                'Favorable_Rate': round(float(fav_rate), 4)
            })
        sub_df = pd.DataFrame(rows)
        max_rate = sub_df['Favorable_Rate'].max() if len(sub_df) > 0 else 1.0
        sub_df['Disparate_Impact_Ratio'] = (sub_df['Favorable_Rate'] / (max_rate if max_rate > 0 else 1.0)).round(3)
        sub_df['EEOC_80_Rule'] = np.where(sub_df['Disparate_Impact_Ratio'] >= 0.80, 'PASS', 'REVIEW (Expected Structural Gap)')
        print(sub_df.to_string(index=False))
        results[col] = sub_df
        
    print("\n[Governance & Protected-Attribute Note]")
    print(" • Demographic attributes (gender, race/ethnicity, age, disability status) are intentionally excluded")
    print("   from model inputs to prevent direct disparate treatment.")
    print(" • Prior to any real-world HR deployment, disparate impact testing must be conducted on an external")
    print("   institutionally-governed validation cohort containing self-reported demographic labels.")
    return results

def main():
    print(f"Loading PathNet0 and evaluation dataset from DATA_DIR={DATA_DIR}...")
    features_path = os.path.join(DATA_DIR, "pathnet0_training_features.parquet")
    if not os.path.exists(features_path):
        print("Preprocessed features not found. Run train_pathnet0.py first.")
        return
        
    df = pd.read_parquet(features_path)
    df['transition_type_id'] = df['transition_type'].map(TRANSITION_TYPE_TO_ID).fillna(1).astype(int)
    
    # Reconstruct exact GroupShuffleSplit held-out test set (zero person_id overlap with train/val)
    y_type_all = df['transition_type_id'].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    idx_train, idx_temp = next(gss1.split(df, y_type_all, groups=df['person_id']))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    rel_val, rel_test = next(gss2.split(df.iloc[idx_temp], y_type_all[idx_temp], groups=df.iloc[idx_temp]['person_id']))
    idx_test = idx_temp[rel_test]
    df_test = df.iloc[idx_test].copy().reset_index(drop=True)
    print(f"Strictly Held-Out Test Set: {len(df_test):,} records (0 person_id overlap with train/val).")
    
    engine = load_pathnet_system(DATA_DIR)
    X_test = engine.pipeline.transform(df_test)
    
    device = engine.model.input_layer[0].weight.device
    deep_prob_list, deep_type_list = [], []
    with torch.no_grad():
        for start in range(0, len(X_test), 8192):
            x_tensor = torch.tensor(X_test[start:start+8192], dtype=torch.float32).to(device)
            p_b, l_b = engine.model(x_tensor)
            deep_prob_list.append(p_b.cpu().numpy().ravel())
            deep_type_list.append(torch.softmax(l_b, dim=1).cpu().numpy())
    deep_prob = np.concatenate(deep_prob_list)
    deep_type_probs = np.vstack(deep_type_list)
        
    hgb_prob = engine.hgb_reg.predict(X_test)
    hgb_type_probs = engine.hgb_clf.predict_proba(X_test)
    
    uncalibrated_prob = np.clip(0.55 * deep_prob + 0.45 * hgb_prob, 0.001, 0.999)
    if engine.calibrator is not None:
        calibrated_prob = np.clip(engine.calibrator.predict(uncalibrated_prob), 0.001, 0.999)
    else:
        calibrated_prob = uncalibrated_prob
        
    ens_type_probs = 0.50 * deep_type_probs + 0.50 * hgb_type_probs
    y_type_pred = np.argmax(ens_type_probs, axis=1)
    
    y_true_prob = df_test['transition_probability'].values
    y_true_type = df_test['transition_type_id'].values
    y_binary_outcome = (y_true_prob >= 0.35).astype(int)
    
    # 1. Compare Uncalibrated vs Isotonic-Calibrated Metrics on Held-Out Test Set
    ece_uncal, _ = compute_expected_calibration_error(y_true_prob, uncalibrated_prob, n_bins=10)
    ece_cal, ece_df = compute_expected_calibration_error(y_true_prob, calibrated_prob, n_bins=10)
    brier_uncal = brier_score_loss(y_binary_outcome, uncalibrated_prob)
    brier_cal = brier_score_loss(y_binary_outcome, calibrated_prob)
    
    print("\n" + "=" * 85)
    print(" 📈 POST-HOC ISOTONIC PROBABILITY CALIBRATION BENCHMARK (HELD-OUT TEST SET)")
    print("=" * 85)
    print(f" • Uncalibrated Ensemble ECE          : {ece_uncal:.4f} ({ece_uncal*100:.2f}%)")
    print(f" • Isotonic-Calibrated Ensemble ECE   : {ece_cal:.4f} ({ece_cal*100:.2f}%)")
    print(f" • Uncalibrated Brier Score (p>=0.35) : {brier_uncal:.4f}")
    print(f" • Calibrated Brier Score   (p>=0.35) : {brier_cal:.4f}")
    print("\nCalibrated Reliability Bins Breakdown (Test Set):")
    print(ece_df.to_string(index=False))
    
    # 2. Multi-Dimensional Subgroup Audit
    audit_subgroup_fairness_and_parity(
        df_test, y_true_prob, calibrated_prob, y_true_type, y_type_pred, threshold=0.35
    )
    print("=" * 85 + "\n")

if __name__ == '__main__':
    main()
