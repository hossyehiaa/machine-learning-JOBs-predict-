"""
FitNet (Model 1): Probability Calibration & Multi-Dimensional Subgroup Fairness Audit
Evaluates Uncalibrated vs. Isotonic-Calibrated ECE and Brier Score on the strictly held-out
GroupShuffleSplit test set (n=34,022 unseen candidate resume-job pairs), plus Subgroup Error
Parity & Disparate Impact across Occupational Sectors, Tenure Brackets, and Seniority Tiers.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, mean_absolute_error, accuracy_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fitnet_inference import load_fitnet_system
from train_fitnet import LABEL_TO_ID, build_continuous_fit_target, infer_primary_sector, predict_deep_batched

def compute_expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        mask = (bin_indices == i)
        n_in_bin = int(np.sum(mask))
        if n_in_bin > 0:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_true = float(np.mean(y_true[mask]))
            abs_err = abs(avg_true - avg_conf)
            ece += (n_in_bin / len(y_true)) * abs_err
            bin_details.append({
                'bin': f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                'count': n_in_bin,
                'mean_pred': round(avg_conf, 4),
                'empirical_score': round(avg_true, 4),
                'abs_error': round(abs_err, 4)
            })
    return float(ece), pd.DataFrame(bin_details)

def audit_subgroup_fairness_and_parity(df_test, y_score_true, y_score_pred, y_label_true, y_label_pred, threshold=0.60):
    df = df_test.copy().reset_index(drop=True)
    df['y_score_true'] = y_score_true
    df['y_score_pred'] = y_score_pred
    df['y_label_true'] = y_label_true
    df['y_label_pred'] = y_label_pred
    df['favorable_outcome'] = (df['y_score_pred'] >= threshold).astype(int)

    unique_cur = df['current_role'].astype(str).unique()
    sector_map = {r: infer_primary_sector(r) for r in unique_cur}
    df['occupational_sector'] = df['current_role'].astype(str).map(sector_map)
    df['tenure_bracket'] = pd.cut(
        df['tenure_years'],
        bins=[-0.1, 5.0, 12.0, 20.0, 100.0],
        labels=['<=5 yrs', '5-12 yrs', '12-20 yrs', '20+ yrs']
    )

    print("\n" + "=" * 85)
    print(" ⚖️ FITNET SUBGROUP ERROR PARITY & DISPARATE IMPACT AUDIT (HELD-OUT TEST, n=" + f"{len(df):,})")
    print("=" * 85)

    results = {}
    for title, col in [
        ("Occupational Sector", "occupational_sector"),
        ("Experience Tenure Bracket", "tenure_bracket"),
        ("Current Seniority Tier", "current_seniority")
    ]:
        print(f"\n[Audit Dimension: {title}]")
        rows = []
        for g_val, grp in df.groupby(col, observed=False):
            if len(grp) < 10:
                continue
            rows.append({
                'Subgroup': str(g_val),
                'Count': int(len(grp)),
                'Score_MAE': round(float(mean_absolute_error(grp['y_score_true'], grp['y_score_pred'])), 4),
                'Tier_Acc': round(float(accuracy_score(grp['y_label_true'], grp['y_label_pred'])), 4),
                'Macro_F1': round(float(f1_score(grp['y_label_true'], grp['y_label_pred'], average='macro', zero_division=0)), 4),
                'High_Fit_Rate': round(float(grp['favorable_outcome'].mean()), 4)
            })
        sub_df = pd.DataFrame(rows)
        max_rate = sub_df['High_Fit_Rate'].max() if len(sub_df) > 0 else 1.0
        sub_df['Disparate_Impact_Ratio'] = (sub_df['High_Fit_Rate'] / (max_rate if max_rate > 0 else 1.0)).round(3)
        sub_df['EEOC_80_Rule'] = np.where(sub_df['Disparate_Impact_Ratio'] >= 0.80, 'PASS', 'REVIEW (Expected Structural Gap)')
        print(sub_df.to_string(index=False))
        results[col] = sub_df
    return results

def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "fitnet_training_features.parquet"))
    y_label = df['initial_match_label'].astype(str).map(LABEL_TO_ID).values.astype(np.int64)
    y_score = build_continuous_fit_target(df)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    _, idx_temp = next(gss1.split(df, y_label, groups=df['candidate_id']))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    _, rel_test = next(gss2.split(df.iloc[idx_temp], y_label[idx_temp], groups=df.iloc[idx_temp]['candidate_id']))
    idx_test = idx_temp[rel_test]
    df_test = df.iloc[idx_test].copy().reset_index(drop=True)
    print(f"Strictly Held-Out FitNet Test Set: {len(df_test):,} records (0 candidate_id overlap).")

    engine = load_fitnet_system(DATA_DIR)
    X_test = engine.pipeline.transform(df_test)

    s_deep, p_deep = predict_deep_batched(engine.model, X_test)
    s_hgb = engine.hgb_reg.predict(X_test)
    p_hgb = engine.hgb_clf.predict_proba(X_test)

    uncal_score = np.clip(0.45 * s_deep + 0.55 * s_hgb, 0.01, 0.99)
    cal_score = np.clip(engine.calibrator.predict(uncal_score), 0.01, 0.99) if engine.calibrator else uncal_score
    y_label_pred = (0.45 * p_deep + 0.55 * p_hgb).argmax(axis=1)

    y_score_test = y_score[idx_test]
    y_label_test = y_label[idx_test]
    y_high_binary = (y_label_test == 2).astype(int)

    ece_uncal, _ = compute_expected_calibration_error(y_score_test, uncal_score)
    ece_cal, ece_df = compute_expected_calibration_error(y_score_test, cal_score)
    brier_uncal = brier_score_loss(y_high_binary, uncal_score)
    brier_cal = brier_score_loss(y_high_binary, cal_score)

    print("\n" + "=" * 85)
    print(" 📈 FITNET POST-HOC ISOTONIC CALIBRATION BENCHMARK (HELD-OUT TEST SET)")
    print("=" * 85)
    print(f" • Uncalibrated Ensemble ECE          : {ece_uncal:.4f} ({ece_uncal*100:.2f}%)")
    print(f" • Isotonic-Calibrated Ensemble ECE   : {ece_cal:.4f} ({ece_cal*100:.2f}%)")
    print(f" • Uncalibrated Brier Score (High Fit): {brier_uncal:.4f}")
    print(f" • Calibrated Brier Score   (High Fit): {brier_cal:.4f}")
    print("\nCalibrated Reliability Bins Breakdown:")
    print(ece_df.to_string(index=False))

    audit_subgroup_fairness_and_parity(df_test, y_score_test, cal_score, y_label_test, y_label_pred)

if __name__ == '__main__':
    main()
