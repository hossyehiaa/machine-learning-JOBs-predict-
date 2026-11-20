"""
FitNet (Model 1): End-to-End Resume-to-Job & Career Matching System
Trained and evaluated on the 221,903-record FitNet dataset (37,044 unique candidate resumes,
84,508 unique target job postings) using leak-resistant GroupShuffleSplit by candidate_id,
Multi-Task Deep Learning + Gradient Boosted Trees, and Post-Hoc Isotonic Calibration.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import re
import ast
import json
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, brier_score_loss
)
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", DEFAULT_DIR)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using compute device: {DEVICE} | DATA_DIR: {DATA_DIR}")

SENIORITY_MAP = {
    'Intern': 0, 'Junior': 1, 'Mid': 2,
    'Senior': 3, 'Lead/Manager': 4, 'Executive': 5
}

MATCH_LABELS = ['Low', 'Medium', 'High']
LABEL_TO_ID = {'Low': 0, 'Medium': 1, 'High': 2}
ID_TO_LABEL = {0: 'Low', 1: 'Medium', 2: 'High'}

DOMAIN_KEYWORDS = {
    'Software': {'software', 'backend', 'java', 'c++', 'python', 'systems', 'application', 'dotnet', '.net', 'golang', 'ruby', 'php', 'django', 'spring'},
    'Frontend': {'frontend', 'front-end', 'ui', 'ux', 'react', 'angular', 'vue', 'web', 'css', 'javascript', 'typescript'},
    'FullStack': {'fullstack', 'full-stack', 'full', 'stack', 'node', 'nodejs'},
    'Mobile': {'android', 'ios', 'mobile', 'swift', 'kotlin', 'flutter', 'native'},
    'Data & AI': {'data', 'analyst', 'analytics', 'scientist', 'machine', 'learning', 'ai', 'ml', 'nlp', 'deep', 'bi', 'database', 'sql', 'etl'},
    'QA & Testing': {'qa', 'aqa', 'test', 'testing', 'automation', 'quality', 'sdet'},
    'Product & Project': {'product', 'project', 'program', 'agile', 'scrum', 'owner', 'pm', 'delivery', 'business'},
    'Infrastructure & Security': {'devops', 'cloud', 'sre', 'infrastructure', 'administrator', 'network', 'security', 'cybersecurity', 'linux', 'aws', 'system'},
    'Finance & Consulting': {'finance', 'accountant', 'accounting', 'banking', 'consultant', 'audit', 'tax', 'financial'},
    'Sales & Marketing': {'sales', 'account', 'marketing', 'digital', 'media', 'growth', 'pr', 'content', 'seo', 'bpo', 'customer'}
}

ADJ_DOMAIN_MAP = {
    'Software': {'FullStack', 'Frontend', 'Mobile', 'Data & AI', 'QA & Testing', 'Infrastructure & Security'},
    'Frontend': {'FullStack', 'Software', 'Mobile', 'Product & Project'},
    'FullStack': {'Software', 'Frontend', 'Mobile', 'Infrastructure & Security'},
    'Mobile': {'Frontend', 'FullStack', 'Software', 'QA & Testing'},
    'Data & AI': {'Software', 'Infrastructure & Security', 'Product & Project'},
    'QA & Testing': {'Software', 'Frontend', 'Mobile', 'Infrastructure & Security'},
    'Product & Project': {'Software', 'Frontend', 'Data & AI', 'Finance & Consulting', 'Sales & Marketing'},
    'Infrastructure & Security': {'Software', 'FullStack', 'Data & AI', 'QA & Testing'},
    'Finance & Consulting': {'Product & Project', 'Data & AI', 'Sales & Marketing'},
    'Sales & Marketing': {'Product & Project', 'Finance & Consulting'}
}

def clean_text(val):
    if pd.isna(val):
        return ''
    val = str(val).strip().lower()
    val = re.sub(r'[^a-zA-Z0-9\s/\-\+#\.]', ' ', val)
    return ' '.join(val.split())

def parse_skill_list(val):
    if isinstance(val, list):
        return [str(x).strip().lower() for x in val if str(x).strip()]
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip().lower() for x in re.split(r'[,;|]+', s) if x.strip()]

def infer_seniority(role):
    role = str(role).lower()
    if re.search(r'\b(executive|chief|president|vice president|vp|cmo|cto|ceo|cpo|cfo)\b', role):
        return 'Executive'
    if re.search(r'\b(lead|manager|director|head|principal|supervisor|architect)\b', role):
        return 'Lead/Manager'
    if re.search(r'\b(senior|sr|staff)\b', role):
        return 'Senior'
    if re.search(r'\b(junior|jr|entry|associate|trainee|intern)\b', role):
        return 'Junior'
    return 'Mid'

def infer_domains(role_or_text):
    tokens = set(re.split(r'[\s/\-\.]+', clean_text(role_or_text)))
    matched = set()
    for dom, kws in DOMAIN_KEYWORDS.items():
        if tokens & kws:
            matched.add(dom)
    return matched

def infer_primary_sector(role):
    doms = infer_domains(role)
    if not doms:
        return 'General & Other'
    for d in ['Data & AI', 'QA & Testing', 'Infrastructure & Security', 'Mobile', 'Frontend',
              'FullStack', 'Software', 'Product & Project', 'Finance & Consulting', 'Sales & Marketing']:
        if d in doms:
            return d
    return sorted(doms)[0]

def compute_role_alignment(role_a, role_b):
    ta = set(t for t in re.split(r'[\s/\-\.]+', clean_text(role_a)) if len(t) > 1)
    tb = set(t for t in re.split(r'[\s/\-\.]+', clean_text(role_b)) if len(t) > 1)
    dom_a = infer_domains(role_a)
    dom_b = infer_domains(role_b)
    same_dom = bool(dom_a & dom_b)
    adj_dom = bool(any(db in ADJ_DOMAIN_MAP.get(da, set()) for da in dom_a for db in dom_b))
    lex_jac = len(ta & tb) / max(1, len(ta | tb))
    if same_dom:
        return float(max(lex_jac, 0.70))
    elif adj_dom:
        return float(max(lex_jac, 0.35))
    return float(lex_jac * 0.15)

def build_continuous_fit_target(df):
    """
    Constructs a bounded continuous fit score in [0.02, 0.98] anchored by the ground-truth
    match label (Low / Medium / High), empirical skill_match_ratio, and seniority alignment.
    """
    lbl_id = df['initial_match_label'].astype(str).map(LABEL_TO_ID).fillna(1).values.astype(np.float32)
    base_map = np.where(lbl_id == 2, 0.78, np.where(lbl_id == 1, 0.48, 0.16))
    skill_ratio = np.clip(df['skill_match_ratio'].values.astype(np.float32), 0.0, 1.0)
    sen_diff = np.abs(df['seniority_level_change'].values.astype(np.float32))
    score = base_map + 0.16 * skill_ratio - 0.02 * np.minimum(sen_diff, 3.0)
    return np.clip(score, 0.02, 0.98).astype(np.float32)

class FitNetFeaturePipeline:
    def __init__(self, text_tfidf_dim=4000, role_tfidf_dim=2000, svd_dim=24):
        self.text_tfidf_dim = text_tfidf_dim
        self.role_tfidf_dim = role_tfidf_dim
        self.svd_dim = svd_dim
        self.text_vectorizer = None
        self.role_vectorizer = None
        self.svd = None
        self.scaler = None
        self.num_cols = [
            'tenure_years', 'resume_length', 'job_description_length', 'has_structured_resume',
            'current_skill_count', 'target_skill_count', 'skill_gap_count', 'skill_match_ratio',
            'roadmap_phases_count', 'had_missing_tenure', 'is_zero_tenure',
            'current_seniority_level', 'target_seniority_level', 'seniority_level_change'
        ]

    def _build_texts(self, df):
        cur_txt = df['current_role'].astype(str) + ' ' + df['current_skills'].astype(str)
        tgt_txt = (
            df['target_future_role'].astype(str) + ' ' +
            df['required_future_skills'].astype(str) + ' ' +
            df['target_job_description'].fillna('').astype(str).str.slice(0, 180)
        )
        return cur_txt, tgt_txt

    def _extract_semantic(self, df):
        cur_roles = df['current_role'].astype(str).tolist()
        tgt_roles = df['target_future_role'].astype(str).tolist()

        u_roles = list(dict.fromkeys(cur_roles + tgt_roles))
        role_dom_cache = {r: (set(t for t in re.split(r'[\s/\-\.]+', clean_text(r)) if len(t) > 1), infer_domains(r)) for r in u_roles}
        pair_cache = {}
        align_scores = np.empty(len(cur_roles), dtype=np.float32)
        for i, (c, t) in enumerate(zip(cur_roles, tgt_roles)):
            k = (c, t)
            if k in pair_cache:
                align_scores[i] = pair_cache[k]
                continue
            ta, da = role_dom_cache[c]
            tb, db = role_dom_cache[t]
            same_d = bool(da & db)
            adj_d = bool(any(x in ADJ_DOMAIN_MAP.get(y, set()) for y in da for x in db))
            lex = len(ta & tb) / max(1, len(ta | tb))
            val = float(max(lex, 0.70)) if same_d else (float(max(lex, 0.35)) if adj_d else float(lex * 0.15))
            pair_cache[k] = val
            align_scores[i] = val
        align_scores = align_scores.reshape(-1, 1)

        role_to_i = {r: idx for idx, r in enumerate(u_roles)}
        c_idx = np.array([role_to_i[c] for c in cur_roles], dtype=np.int32)
        t_idx = np.array([role_to_i[t] for t in tgt_roles], dtype=np.int32)
        u_r_tfidf = self.role_vectorizer.transform(u_roles)
        role_cos = np.asarray(u_r_tfidf[c_idx].multiply(u_r_tfidf[t_idx]).sum(axis=1), dtype=np.float32).reshape(-1, 1)

        cur_txt, tgt_txt = self._build_texts(df)
        c_tfidf = self.text_vectorizer.transform(cur_txt)
        t_tfidf = self.text_vectorizer.transform(tgt_txt)
        text_cos = np.asarray(c_tfidf.multiply(t_tfidf).sum(axis=1), dtype=np.float32).reshape(-1, 1)

        c_svd = self.svd.transform(c_tfidf).astype(np.float32)
        t_svd = self.svd.transform(t_tfidf).astype(np.float32)
        norm_c = np.linalg.norm(c_svd, axis=1, keepdims=True) + 1e-8
        norm_t = np.linalg.norm(t_svd, axis=1, keepdims=True) + 1e-8
        svd_cos = np.sum((c_svd / norm_c) * (t_svd / norm_t), axis=1, keepdims=True).astype(np.float32)

        return align_scores, role_cos, text_cos, svd_cos, c_svd, t_svd

    def fit(self, df_train):
        print("Fitting leak-free FitNet role & skill-text TF-IDF vectorizers strictly on training split...")
        all_roles = pd.concat([df_train['current_role'].astype(str), df_train['target_future_role'].astype(str)], ignore_index=True)
        self.role_vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), min_df=2, max_features=self.role_tfidf_dim,
            sublinear_tf=True, dtype=np.float32
        ).fit(all_roles)

        cur_txt, tgt_txt = self._build_texts(df_train)
        self.text_vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), min_df=3, max_features=self.text_tfidf_dim,
            sublinear_tf=True, dtype=np.float32
        )
        c_tfidf = self.text_vectorizer.fit_transform(cur_txt)

        print(f"Fitting TruncatedSVD to {self.svd_dim} dense semantic components...")
        self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=SEED).fit(c_tfidf)

        align_scores, role_cos, text_cos, svd_cos, _, _ = self._extract_semantic(df_train)
        X_num_raw = np.hstack([df_train[self.num_cols].values.astype(np.float32), align_scores, role_cos, text_cos, svd_cos])
        print("Fitting numerical feature scaler...")
        self.scaler = StandardScaler().fit(X_num_raw)
        return self

    def transform(self, df):
        align_scores, role_cos, text_cos, svd_cos, c_svd, t_svd = self._extract_semantic(df)
        X_num_raw = np.hstack([df[self.num_cols].values.astype(np.float32), align_scores, role_cos, text_cos, svd_cos])
        X_num = self.scaler.transform(X_num_raw)
        return np.hstack([X_num, c_svd, t_svd, np.abs(c_svd - t_svd), c_svd * t_svd]).astype(np.float32)

class FitNetDataset(Dataset):
    def __init__(self, X, y_score, y_label):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_score = torch.tensor(y_score, dtype=torch.float32).unsqueeze(1)
        self.y_label = torch.tensor(y_label, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_score[idx], self.y_label[idx]

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class FitNetMultiTaskNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res1 = ResidualBlock(hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.res2(self.res1(self.input_layer(x)))
        return self.score_head(h), self.class_head(h)

def predict_deep_batched(model, X_arr, bs=4096):
    model.eval()
    scores_l, probs_l = [], []
    with torch.no_grad():
        for start in range(0, len(X_arr), bs):
            xb = torch.tensor(X_arr[start:start+bs], dtype=torch.float32).to(DEVICE)
            sb, lb = model(xb)
            scores_l.append(sb.cpu().numpy().ravel())
            probs_l.append(torch.softmax(lb, dim=1).cpu().numpy())
    return np.concatenate(scores_l), np.vstack(probs_l)

class FitNetInferenceEngine:
    def __init__(self, deep_model, pipeline, hgb_reg, hgb_clf, calibrator=None):
        self.model = deep_model
        self.pipeline = pipeline
        self.hgb_reg = hgb_reg
        self.hgb_clf = hgb_clf
        self.calibrator = calibrator
        if self.model:
            self.model.eval()

    def predict_job_fit(
        self,
        current_role: str,
        current_skills: list,
        tenure_years: float,
        target_role: str,
        required_skills: list,
        job_description: str = ""
    ):
        c_skills = parse_skill_list(current_skills)
        r_skills = parse_skill_list(required_skills)
        c_set = set(c_skills)
        r_set = set(r_skills)
        matched = sorted(c_set & r_set)
        missing = sorted(r_set - c_set)

        c_count = len(c_set)
        t_count = max(1, len(r_set))
        gap_count = len(missing)
        match_ratio = round(len(matched) / t_count, 4)

        c_sen = infer_seniority(current_role)
        t_sen = infer_seniority(target_role)
        c_lvl = SENIORITY_MAP.get(c_sen, 2)
        t_lvl = SENIORITY_MAP.get(t_sen, 2)
        sen_change = t_lvl - c_lvl

        p1 = max(1, len(missing) // 3) if missing else 0
        p2 = max(p1 + 1, (2 * len(missing)) // 3) if len(missing) > 1 else len(missing)
        roadmap = {
            "Phase 1: Immediate Job Prerequisites": missing[:p1],
            "Phase 2: Core Technical Alignment": missing[p1:p2],
            "Phase 3: Role Mastery": missing[p2:]
        } if missing else {"Ready for Role": ["All required skills matched"]}

        row_df = pd.DataFrame([{
            'current_role': clean_text(current_role),
            'target_future_role': clean_text(target_role),
            'current_skills': str(c_skills),
            'required_future_skills': str(r_skills),
            'target_job_description': clean_text(job_description)[:180],
            'tenure_years': float(tenure_years),
            'resume_length': max(1500, c_count * 180),
            'job_description_length': max(600, len(job_description) or (t_count * 150)),
            'has_structured_resume': 1,
            'current_skill_count': c_count,
            'target_skill_count': len(r_set),
            'skill_gap_count': gap_count,
            'skill_match_ratio': match_ratio,
            'roadmap_phases_count': min(3, max(1, (gap_count + 2) // 3)),
            'had_missing_tenure': 0,
            'is_zero_tenure': int(tenure_years == 0),
            'current_seniority_level': c_lvl,
            'target_seniority_level': t_lvl,
            'seniority_level_change': sen_change
        }])

        feat = self.pipeline.transform(row_df)
        s_deep, c_probs_deep = predict_deep_batched(self.model, feat, bs=1)
        s_hgb = float(self.hgb_reg.predict(feat)[0])
        c_probs_hgb = self.hgb_clf.predict_proba(feat)[0]

        raw_score = float(np.clip(0.45 * float(s_deep[0]) + 0.55 * s_hgb, 0.01, 0.99))
        class_probs = 0.45 * c_probs_deep[0] + 0.55 * c_probs_hgb

        if self.calibrator is not None:
            cal_score = float(np.clip(self.calibrator.predict([raw_score])[0], 0.01, 0.99))
        else:
            cal_score = raw_score

        # Incorporate explicit skill-overlap & role-domain alignment into live interactive scoring
        role_align = compute_role_alignment(current_role, target_role)
        explicit_fit = np.clip(0.55 * match_ratio + 0.30 * role_align + 0.15 * max(0.0, 1.0 - 0.2 * abs(sen_change)), 0.02, 0.98)
        final_score = float(np.clip(0.45 * cal_score + 0.55 * explicit_fit, 0.02, 0.98))

        if final_score >= 0.60:
            prior_cls = np.array([0.04, 0.20, 0.76], dtype=np.float32)
        elif final_score >= 0.32:
            prior_cls = np.array([0.14, 0.70, 0.16], dtype=np.float32)
        else:
            prior_cls = np.array([0.78, 0.18, 0.04], dtype=np.float32)
        final_cls_probs = 0.40 * class_probs + 0.60 * prior_cls
        final_cls_probs = final_cls_probs / np.sum(final_cls_probs)

        pred_id = int(np.argmax(final_cls_probs))
        match_tier = ID_TO_LABEL[pred_id]

        return {
            'current_role': current_role,
            'target_job_title': target_role,
            'tenure_years': tenure_years,
            'fit_score_percentage': f"{final_score * 100:.2f}%",
            'fit_score_raw': round(final_score, 4),
            'predicted_match_tier': match_tier,
            'tier_probabilities': {ID_TO_LABEL[i]: f"{final_cls_probs[i]*100:.1f}%" for i in range(3)},
            'skill_match_ratio': f"{match_ratio * 100:.1f}%",
            'matched_skills': matched,
            'missing_skill_gap': missing,
            'missing_skill_gap_count': gap_count,
            'upskilling_roadmap': roadmap
        }

def main():
    print("=" * 75)
    print(" FitNet (Model 1): Resume-to-Job Matching Training & Calibration Pipeline")
    print("=" * 75)

    pq_path = os.path.join(DATA_DIR, "fitnet_training_features.parquet")
    df = pd.read_parquet(pq_path)
    print(f"Loaded FitNet dataset: {len(df):,} rows x {df.shape[1]} columns ({df['candidate_id'].nunique():,} unique candidate resumes)")

    y_label = df['initial_match_label'].astype(str).map(LABEL_TO_ID).values.astype(np.int64)
    y_score = build_continuous_fit_target(df)

    # Leak-Resistant GroupShuffleSplit by candidate_id (70% Train, 15% Val, 15% Test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    idx_train, idx_temp = next(gss1.split(df, y_label, groups=df['candidate_id']))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    rel_val, rel_test = next(gss2.split(df.iloc[idx_temp], y_label[idx_temp], groups=df.iloc[idx_temp]['candidate_id']))
    idx_val = idx_temp[rel_val]
    idx_test = idx_temp[rel_test]

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    overlap = len(set(df_train['candidate_id']) & set(df_test['candidate_id']))
    print(f"Group Split by candidate_id -> Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,} (Candidate overlap: {overlap})")

    pipeline = FitNetFeaturePipeline(text_tfidf_dim=4000, role_tfidf_dim=2000, svd_dim=24)
    pipeline.fit(df_train)

    X_train = pipeline.transform(df_train)
    X_val = pipeline.transform(df_val)
    X_test = pipeline.transform(df_test)
    print(f"Engineered FitNet feature vector dimension: {X_train.shape[1]}")

    # Train PyTorch Multi-Task FitNet
    train_loader = DataLoader(FitNetDataset(X_train, y_score[idx_train], y_label[idx_train]), batch_size=1024, shuffle=True)
    val_loader = DataLoader(FitNetDataset(X_val, y_score[idx_val], y_label[idx_val]), batch_size=2048, shuffle=False)

    deep_model = FitNetMultiTaskNet(input_dim=X_train.shape[1], hidden_dim=256, num_classes=3).to(DEVICE)
    optimizer = optim.AdamW(deep_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    crit_score = nn.MSELoss()
    crit_cls = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    print("\n--- Training Deep Multi-Task FitNet (12 epochs) ---")
    for epoch in range(1, 13):
        deep_model.train()
        tr_loss = 0.0
        for xb, ys, yl in train_loader:
            xb, ys, yl = xb.to(DEVICE), ys.to(DEVICE), yl.to(DEVICE)
            optimizer.zero_grad()
            ps, pl = deep_model(xb)
            loss = crit_score(ps, ys) + 0.45 * crit_cls(pl, yl)
            loss.backward()
            nn.utils.clip_grad_norm_(deep_model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        s_val, p_val = predict_deep_batched(deep_model, X_val)
        v_mae = mean_absolute_error(y_score[idx_val], s_val)
        v_acc = accuracy_score(y_label[idx_val], p_val.argmax(axis=1))
        scheduler.step(v_mae)
        if v_mae < best_val_loss:
            best_val_loss = v_mae
            best_state = {k: v.detach().cpu().clone() for k, v in deep_model.state_dict().items()}
            flag = "*"
        else:
            flag = ""
        print(f"Epoch {epoch:02d}/12 | Train Loss: {tr_loss:.4f} | Val Score MAE: {v_mae:.4f} | Val Tier Acc: {v_acc*100:.2f}% {flag}")

    if best_state is not None:
        deep_model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    print("\n--- Training HistGradientBoosting Regressor & Classifier ---")
    hgb_reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=SEED).fit(X_train, y_score[idx_train])
    hgb_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED).fit(X_train, y_label[idx_train])

    # Fit Isotonic Calibrator on Validation Set
    s_val_deep, _ = predict_deep_batched(deep_model, X_val)
    s_val_hgb = hgb_reg.predict(X_val)
    s_val_ens = np.clip(0.45 * s_val_deep + 0.55 * s_val_hgb, 0.01, 0.99)
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip').fit(s_val_ens, y_score[idx_val])

    # Evaluate on Held-Out Test Set (n=33,286)
    s_te_deep, p_te_deep = predict_deep_batched(deep_model, X_test)
    s_te_hgb = hgb_reg.predict(X_test)
    p_te_hgb = hgb_clf.predict_proba(X_test)

    s_te_cal = calibrator.predict(np.clip(0.45 * s_te_deep + 0.55 * s_te_hgb, 0.01, 0.99))
    t_te_ens = (0.45 * p_te_deep + 0.55 * p_te_hgb).argmax(axis=1)

    print("\n" + "=" * 65)
    print(f" FitNet Held-Out Test Set Evaluation (n={len(df_test):,}, 0 Candidate Overlap)")
    print("=" * 65)
    for name, s_p, t_p in [
        ("Deep Multi-Task FitNet (PyTorch)", s_te_deep, p_te_deep.argmax(axis=1)),
        ("HistGradientBoosting (Trees)", s_te_hgb, p_te_hgb.argmax(axis=1)),
        ("Unified FitNet Ensemble (Calibrated)", s_te_cal, t_te_ens)
    ]:
        print(f"\n[{name}]")
        print(f"  * Fit Score MAE         : {mean_absolute_error(y_score[idx_test], s_p):.4f}")
        print(f"  * Fit Score RMSE        : {np.sqrt(mean_squared_error(y_score[idx_test], s_p)):.4f}")
        print(f"  * Fit Score R²          : {r2_score(y_score[idx_test], s_p):.4f}")
        print(f"  * Match Tier Accuracy   : {accuracy_score(y_label[idx_test], t_p)*100:.2f}%")
        print(f"  * Match Tier Macro F1   : {f1_score(y_label[idx_test], t_p, average='macro'):.4f}")

    torch.save(deep_model.state_dict(), os.path.join(DATA_DIR, "fitnet_deep_model.pt"))
    joblib.dump(pipeline, os.path.join(DATA_DIR, "fitnet_feature_pipeline.joblib"))
    joblib.dump(hgb_reg, os.path.join(DATA_DIR, "fitnet_hgb_regressor.joblib"))
    joblib.dump(hgb_clf, os.path.join(DATA_DIR, "fitnet_hgb_classifier.joblib"))
    joblib.dump(calibrator, os.path.join(DATA_DIR, "fitnet_isotonic_calibrator.joblib"))
    print(f"\nAll FitNet model artifacts saved to {DATA_DIR}")

    engine = FitNetInferenceEngine(deep_model, pipeline, hgb_reg, hgb_clf, calibrator)
    for c_role, c_sk, ten, t_role, r_sk in [
        ("Python Developer", ["python", "django", "postgresql", "docker", "git", "rest apis"], 3.5,
         "Senior Backend Python Engineer", ["python", "django", "postgresql", "docker", "kubernetes", "redis"]),
        ("Java Developer", ["java", "spring", "mysql", "git", "junit"], 3.0,
         "Frontend React Developer", ["javascript", "typescript", "react", "css", "html", "git"]),
        ("Project Manager", ["project management", "agile", "scrum", "jira", "stakeholder management"], 5.0,
         "Senior MLOps Engineer", ["python", "kubernetes", "terraform", "pytorch", "mlflow", "aws"])
    ]:
        res = engine.predict_job_fit(c_role, c_sk, ten, t_role, r_sk)
        print(f"\n🎯 [FitNet Match]: {c_role} -> {t_role}")
        print(f"   • Match Tier: {res['predicted_match_tier']} | Fit Score: {res['fit_score_percentage']} | Dist: {res['tier_probabilities']}")
        print(f"   • Skill Overlap: {res['skill_match_ratio']} | Matched: {res['matched_skills']} | Missing: {res['missing_skill_gap']}")

if __name__ == '__main__':
    main()
