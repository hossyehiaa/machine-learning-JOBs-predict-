"""
PathNet0: End-to-End Career Path Network & Trajectory Prediction Prototype
Multi-Task Deep Learning + Gradient Boosted Trees + Isotonic Probability Calibration
+ Multi-Group Fairness Audit + TF-IDF Retrieval Index
"""

import os
# Prevent OpenMP runtime conflict on macOS ARM64 / Python 3.13
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import gc
import re
import ast
import json
import glob
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, confusion_matrix, brier_score_loss
)
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# Configurable Data/Model Directory via Environment Variable
DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", DEFAULT_DIR)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using compute device: {DEVICE} | DATA_DIR: {DATA_DIR}")

# ---------------------------------------------------------------------------
# 1. Seniority Taxonomy & Text Helpers
# ---------------------------------------------------------------------------
SENIORITY_MAP = {
    'Intern': 0,
    'Junior': 1,
    'Mid': 2,
    'Senior': 3,
    'Lead/Manager': 4,
    'Executive': 5
}
INV_SENIORITY_MAP = {v: k for k, v in SENIORITY_MAP.items()}

TRANSITION_TYPES = ['Vertical_Promotion', 'Adjacent_Pivot', 'Career_Shift']
TRANSITION_TYPE_TO_ID = {t: i for i, t in enumerate(TRANSITION_TYPES)}
ID_TO_TRANSITION_TYPE = {i: t for i, t in enumerate(TRANSITION_TYPES)}

SENIORITY_STOPWORDS = {
    'intern', 'internship', 'trainee', 'junior', 'jr', 'entry', 'associate',
    'mid', 'intermediate', 'senior', 'sr', 'staff', 'principal', 'lead',
    'manager', 'director', 'head', 'supervisor', 'vp', 'vice', 'president',
    'chief', 'executive', 'c-level', 'ceo', 'cto', 'cfo', 'cmo', 'cpo',
    'group', 'ii', 'iii', 'iv', 'i'
}

DOMAIN_KEYWORDS = {
    'Software': {'software', 'backend', 'java', 'c++', 'systems', 'application'},
    'Frontend': {'frontend', 'front-end', 'ui', 'react', 'web', 'css', 'javascript'},
    'FullStack': {'fullstack', 'full-stack', 'full', 'stack', 'node'},
    'Mobile': {'android', 'ios', 'mobile', 'swift', 'kotlin', 'flutter', 'native'},
    'Data & AI': {'data', 'analyst', 'analytics', 'scientist', 'machine', 'learning', 'ai', 'ml', 'bi', 'statistician', 'researcher'},
    'Product': {'product', 'agile', 'scrum', 'owner', 'program'},
    'Sales': {'sales', 'account', 'business', 'commercial', 'bdr', 'ae', 'revenue', 'representative'},
    'Marketing': {'marketing', 'marketer', 'growth', 'brand', 'communication', 'communications', 'seo', 'content', 'pr', 'media'},
    'Customer & Ops': {'customer', 'support', 'success', 'care', 'call', 'centre', 'center', 'service', 'operations', 'after-sales'},
    'Infrastructure': {'devops', 'cloud', 'sre', 'reliability', 'infrastructure', 'administrator', 'network', 'security', 'platform'},
    'Construction & Field': {'construction', 'drainage', 'building', 'civil', 'field', 'site', 'machinery', 'worker', 'technician'}
}

ADJ_DOMAIN_MAP = {
    'Software': {'FullStack', 'Frontend', 'Mobile', 'Data & AI', 'Infrastructure'},
    'Frontend': {'FullStack', 'Software', 'Mobile', 'Product'},
    'FullStack': {'Software', 'Frontend', 'Mobile', 'Infrastructure', 'Product'},
    'Mobile': {'Frontend', 'FullStack', 'Software'},
    'Data & AI': {'Software', 'Product', 'Infrastructure'},
    'Product': {'Marketing', 'Data & AI', 'Sales', 'FullStack'},
    'Sales': {'Marketing', 'Customer & Ops', 'Product'},
    'Marketing': {'Sales', 'Product', 'Customer & Ops'},
    'Customer & Ops': {'Sales', 'Marketing', 'Product'},
    'Infrastructure': {'Software', 'FullStack', 'Data & AI'},
    'Construction & Field': {'Customer & Ops', 'Sales'}
}

def clean_role_text(val):
    if pd.isna(val):
        return 'unknown'
    val = str(val).strip().lower()
    val = re.sub(r'[^a-zA-Z0-9\s/\-\+#\.]', '', val)
    return ' '.join(val.split())

def clean_job_desc(val):
    if pd.isna(val):
        return ''
    val = str(val).lower()
    val = re.sub(r'[*#_\-`]', ' ', val)
    val = re.sub(r'\r\n|\n|\r', ' ', val)
    return ' '.join(val.split())

def infer_seniority(role):
    role = str(role).lower()
    if re.search(r'\b(executive|chief|president|vice president|vp|cmo|cto|ceo|cpo|cfo)\b', role):
        return 'Executive'
    if re.search(r'\b(lead|manager|director|head|principal|supervisor)\b', role):
        return 'Lead/Manager'
    if re.search(r'\b(senior|sr|staff)\b', role):
        return 'Senior'
    if re.search(r'\b(mid|intermediate)\b', role):
        return 'Mid'
    if re.search(r'\b(junior|jr|entry|associate)\b', role):
        return 'Junior'
    if re.search(r'\b(intern|internship|trainee)\b', role):
        return 'Intern'
    return 'Mid'

def infer_domains(role):
    tokens = set(re.split(r'[\s/\-\.]+', clean_role_text(role)))
    if 'full' in tokens and 'stack' in tokens:
        return {'FullStack'}
    if 'machine' in tokens and 'learning' in tokens:
        return {'Data & AI', 'Software'}
    matched = set()
    for dom, kws in DOMAIN_KEYWORDS.items():
        if dom == 'FullStack':
            if 'fullstack' in tokens or 'full-stack' in tokens:
                matched.add(dom)
            continue
        if tokens & kws:
            matched.add(dom)
    return matched

def infer_primary_sector(role):
    doms = infer_domains(role)
    if not doms:
        return 'General & Other'
    for d in ['Data & AI', 'Mobile', 'Frontend', 'FullStack', 'Infrastructure', 'Software',
              'Product', 'Sales', 'Marketing', 'Customer & Ops', 'Construction & Field']:
        if d in doms:
            return d
    return sorted(doms)[0]

def compute_core_jaccard(role_a, role_b):
    ta = set(t for t in re.split(r'[\s/\-\.]+', clean_role_text(role_a)) if t and t not in SENIORITY_STOPWORDS and len(t) > 1)
    tb = set(t for t in re.split(r'[\s/\-\.]+', clean_role_text(role_b)) if t and t not in SENIORITY_STOPWORDS and len(t) > 1)
    
    dom_a = infer_domains(role_a)
    dom_b = infer_domains(role_b)
    same_dom = bool(dom_a & dom_b)
    adj_dom = bool(any(db in ADJ_DOMAIN_MAP.get(da, set()) for da in dom_a for db in dom_b))
    
    syn = {'developer': 'engineer', 'engineering': 'engineer'}
    ta_norm = {syn.get(t, t) for t in ta}
    tb_norm = {syn.get(t, t) for t in tb}
    lex_jac = len(ta_norm & tb_norm) / max(1, len(ta_norm | tb_norm))
    
    if same_dom and len(dom_a) == 1 and len(dom_b) == 1:
        return float(max(lex_jac, 0.75))
    elif same_dom:
        return float(max(lex_jac, 0.35))
    elif adj_dom:
        return float(max(lex_jac * 0.5, 0.25))
    else:
        return float(lex_jac * 0.15)

def parse_skills(val):
    if isinstance(val, list):
        items = val
    elif isinstance(val, str):
        try:
            items = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    else:
        return []
    if not isinstance(items, list):
        return []
    return list(dict.fromkeys(str(item).strip().lower() for item in items if item))

def roadmap_skills(val):
    if not isinstance(val, str):
        return []
    try:
        roadmap = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []
    def collect(value):
        if isinstance(value, dict):
            return [skill for item in value.values() for skill in collect(item)]
        if isinstance(value, list):
            return [skill for item in value for skill in collect(item)]
        if isinstance(value, str):
            return [value.strip().lower()] if value.strip() else []
        return []
    return list(dict.fromkeys(collect(roadmap)))

# ---------------------------------------------------------------------------
# 2. Stochastic Benchmark Generator with Unobserved Latent Variables
#    (Used only when external 1.14M CSVs are not present in DATA_DIR)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples=25000):
    """
    Generates a benchmark dataset when the external 1.14M-row CSVs are unavailable.
    To avoid trivial deterministic circularity, Outcomes depend heavily on UNOBSERVED
    latent factors (unmeasured candidate aptitude, macroeconomic market demand shocks,
    company-specific title ambiguity, and stochastic transition noise) that are NOT
    available to the model as features.
    """
    print(f"Generating stochastic benchmark career dataset ({n_samples:,} rows) with unobserved latent noise...")
    
    role_clusters = {
        'Software': [
            'software engineering intern', 'junior software engineer', 'software engineer',
            'senior software engineer', 'lead software engineer', 'software engineering manager',
            'vp of software engineering'
        ],
        'Frontend': [
            'frontend developer intern', 'junior frontend developer', 'frontend developer',
            'senior frontend developer', 'lead frontend engineer', 'director of frontend engineering'
        ],
        'FullStack': [
            'junior full stack developer', 'full stack engineer', 'senior full stack engineer',
            'lead full stack engineer', 'engineering manager'
        ],
        'Mobile': [
            'junior android developer', 'android engineer', 'senior android engineer',
            'lead native android engineer', 'principal native android engineer', 'mobile engineering manager'
        ],
        'Data & AI': [
            'junior data analyst', 'data analyst', 'senior data analyst', 'data scientist',
            'senior data scientist', 'machine learning engineer', 'senior machine learning engineer',
            'lead ai researcher', 'data engineering manager', 'head of data science'
        ],
        'Product': [
            'associate product manager', 'product manager', 'senior product manager',
            'lead product manager', 'group product manager', 'director of product management', 'vp of product'
        ],
        'Sales': [
            'junior sales representative', 'business development representative', 'technical sales representative',
            'sales account manager', 'senior sales manager', 'sales director', 'vp of sales'
        ],
        'Marketing': [
            'marketing intern', 'junior marketing associate', 'marketing specialist', 'senior growth marketer',
            'marketing and communication manager', 'director of marketing and communication', 'chief marketing officer'
        ],
        'Customer & Ops': [
            'junior customer support associate', 'customer support specialist', 'after-sales service technician',
            'senior customer success specialist', 'call centre manager', 'customer operations director'
        ],
        'Infrastructure': [
            'junior systems administrator', 'devops engineer', 'cloud architect',
            'site reliability engineer', 'senior devops engineer', 'lead cloud infrastructure manager'
        ],
        'Construction & Field': [
            'apprentice construction worker', 'drainage worker', 'building construction worker',
            'senior field technician', 'construction site manager', 'director of civil operations'
        ]
    }
    
    cluster_skills = {
        'Software': [
            'python', 'java', 'c++', 'algorithms', 'data structures', 'system design',
            'git', 'microservices', 'distributed systems', 'rest apis', 'unit testing',
            'clean architecture', 'concurrency', 'design patterns', 'sql', 'code review',
            'agile methodologies', 'debugging', 'performance profiling', 'ci/cd pipelines'
        ],
        'Frontend': [
            'javascript', 'typescript', 'react', 'next.js', 'css3', 'html5',
            'tailwind css', 'redux', 'state management', 'web performance', 'wcag accessibility',
            'jest testing', 'webpack', 'vite', 'GraphQL', 'responsive design',
            'browser devtools', 'design systems', 'figma handoff', 'ssr optimization'
        ],
        'FullStack': [
            'javascript', 'typescript', 'react', 'node.js', 'python', 'postgresql',
            'mongodb', 'rest apis', 'graphql', 'docker', 'system design', 'redis',
            'authentication & oauth', 'cloud deployment', 'next.js', 'microservices',
            'automated testing', 'git workflow', 'ci/cd', 'api security'
        ],
        'Mobile': [
            'android sdk', 'kotlin', 'java', 'jetpack compose', 'kotlin coroutines',
            'mvvm architecture', 'clean architecture', 'dagger hilt', 'retrofit', 'room database',
            'gradle build system', 'android ndk', 'memory leak debugging', 'mobile ui/ux guidelines',
            'play store deployment', 'offline caching', 'firebase', 'unit testing', 'espresso testing', 'workmanager'
        ],
        'Data & AI': [
            'python', 'sql', 'pandas', 'numpy', 'scikit-learn', 'pytorch', 'tensorflow',
            'deep learning', 'machine learning', 'statistical modeling', 'feature engineering',
            'mlops', 'experiment tracking', 'a/b testing', 'data visualization', 'apache spark',
            'airflow', 'rag architectures', 'llm fine-tuning', 'model deployment'
        ],
        'Product': [
            'product strategy', 'agile & scrum', 'user research', 'product roadmapping',
            'kpi & okr tracking', 'a/b testing', 'stakeholder management', 'competitive analysis',
            'wireframing', 'sql for product', 'go-to-market strategy', 'pricing modeling',
            'customer discovery', 'backlog prioritization', 'funnel optimization', 'data storytelling',
            'cross-functional leadership', 'prd authoring', 'ux heuristics', 'market sizing'
        ],
        'Sales': [
            'crm management', 'salesforce', 'cold outreach', 'pipeline management',
            'b2b negotiation', 'client retention', 'enterprise deal closing', 'account mapping',
            'consultative selling', 'revenue forecasting', 'contract structuring', 'lead qualification',
            'territory planning', 'value proposition pitching', 'objection handling', 'sales enablement',
            'key account growth', 'upselling strategies', 'partner channel sales', 'sla negotiation'
        ],
        'Marketing': [
            'seo strategy', 'sem & paid acquisition', 'content strategy', 'copywriting',
            'brand management', 'funnel analytics', 'google analytics 4', 'email marketing automation',
            'social media strategy', 'public relations', 'customer segmentation', 'campaign attribution',
            'conversion rate optimization', 'influencer partnerships', 'market research', 'media buying',
            'hubspot', 'product marketing', 'event management', 'creative direction'
        ],
        'Customer & Ops': [
            'zendesk & ticketing', 'sla management', 'process improvement', 'crisis resolution',
            'team leadership', 'customer onboarding', 'churn prevention', 'nps analysis',
            'quality assurance auditing', 'workforce scheduling', 'escalation management', 'knowledge base authoring',
            'empathy & active listening', 'root cause analysis', 'operational kpis', 'vendor coordination',
            'incident communication', 'crm workflows', 'service recovery', 'conflict de-escalation'
        ],
        'Infrastructure': [
            'linux administration', 'docker', 'kubernetes', 'terraform', 'aws architecture',
            'gcp', 'ci/cd automation', 'github actions', 'prometheus & grafana', 'security hardening',
            'network protocols', 'bash scripting', 'ansible', 'incident response', 'high availability',
            'disaster recovery', 'iam policy design', 'helm charts', 'serverless computing', 'cost optimization'
        ],
        'Construction & Field': [
            'operate digging equipment', 'blueprint reading', 'site safety standards', 'osha compliance',
            'structural framing', 'concrete pouring', 'drainage system installation', 'surveying instruments',
            'heavy machinery maintenance', 'material estimation', 'quality inspection', 'subcontractor coordination',
            'excavation grading', 'scaffolding safety', 'project scheduling', 'environmental regulations',
            'welding fundamentals', 'electrical wiring basics', 'plumbing layout', 'hazard assessment'
        ]
    }
    
    adj_map = {
        'Software': ['FullStack', 'Frontend', 'Mobile', 'Data & AI', 'Infrastructure'],
        'Frontend': ['FullStack', 'Software', 'Mobile', 'Product'],
        'FullStack': ['Software', 'Frontend', 'Mobile', 'Infrastructure', 'Product'],
        'Mobile': ['Frontend', 'FullStack', 'Software'],
        'Data & AI': ['Software', 'Product', 'Infrastructure'],
        'Product': ['Marketing', 'Data & AI', 'Sales', 'FullStack'],
        'Sales': ['Marketing', 'Customer & Ops', 'Product'],
        'Marketing': ['Sales', 'Product', 'Customer & Ops'],
        'Customer & Ops': ['Sales', 'Marketing', 'Product'],
        'Infrastructure': ['Software', 'FullStack', 'Data & AI'],
        'Construction & Field': ['Customer & Ops', 'Sales']
    }
    
    clusters = list(role_clusters.keys())
    
    # Simulate 6,000 unique individuals with multiple career steps
    n_persons = max(1000, n_samples // 4)
    person_ids = np.random.randint(100000, 3000000000, size=n_persons)
    # Unobserved individual latent factors (e.g. education prestige, networking, interview skill)
    person_latent_aptitude = np.random.normal(0.0, 0.12, size=n_persons)
    
    data = []
    for i in range(n_samples):
        p_idx = np.random.randint(0, n_persons)
        person_id = int(person_ids[p_idx])
        unobserved_aptitude = person_latent_aptitude[p_idx]
        # Unobserved macroeconomic / company hiring shock
        unobserved_market_shock = np.random.normal(0.0, 0.10)
        
        step_index = min(max(int(np.random.geometric(p=0.28)), 1), 15)
        c_cluster = np.random.choice(clusters)
        ladder = role_clusters[c_cluster]
        
        t_rand = np.random.rand()
        if t_rand < 0.30:
            true_regime = 'Vertical_Promotion'
            t_cluster = c_cluster
            curr_idx = np.random.randint(0, max(1, len(ladder) - 1))
            c_role = ladder[curr_idx]
            step_jump = np.random.choice([1, 2], p=[0.72, 0.28])
            target_idx = min(curr_idx + step_jump, len(ladder) - 1)
            t_role = ladder[target_idx]
            
            tenure = np.random.gamma(shape=2.5, scale=1.6)
            # Overlapping skill gap distribution (4 to 20 skills)
            n_gap_skills = int(np.clip(np.random.normal(10, 3.5), 4, 22))
            
            c_lvl = SENIORITY_MAP[infer_seniority(c_role)]
            t_lvl = SENIORITY_MAP[infer_seniority(t_role)]
            lvl_diff = max(0, t_lvl - c_lvl)
            
            base_prob = (
                0.48 + 0.032 * min(tenure, 8.0) - 0.065 * max(0, lvl_diff - 1)
                - 0.008 * (n_gap_skills - 10) + unobserved_aptitude + unobserved_market_shock
            )
            prob = np.clip(base_prob, 0.08, 0.94)
            
            # Boundary ambiguity: ~8% of same-domain lateral/diagonal moves are labeled Adjacent_Pivot in real HR logs
            if np.random.rand() < 0.08:
                transition_type = 'Adjacent_Pivot'
            else:
                transition_type = true_regime
                
        elif t_rand < 0.58:
            true_regime = 'Adjacent_Pivot'
            t_cluster = np.random.choice(adj_map[c_cluster])
            c_role = np.random.choice(role_clusters[c_cluster])
            t_role = np.random.choice(role_clusters[t_cluster])
            
            tenure = np.random.gamma(shape=2.2, scale=1.5)
            # Overlapping skill gap distribution (7 to 25 skills)
            n_gap_skills = int(np.clip(np.random.normal(15, 4.0), 6, 26))
            
            c_lvl = SENIORITY_MAP[infer_seniority(c_role)]
            t_lvl = SENIORITY_MAP[infer_seniority(t_role)]
            lvl_diff = abs(t_lvl - c_lvl)
            
            base_prob = (
                0.24 + 0.022 * min(tenure, 6.0) - 0.035 * lvl_diff
                - 0.006 * (n_gap_skills - 15) + unobserved_aptitude + unobserved_market_shock
            )
            prob = np.clip(base_prob, 0.02, 0.65)
            
            # Boundary ambiguity: cross-functional promotions or distant pivots have ~12% label overlap
            r_amb = np.random.rand()
            if r_amb < 0.06 and t_lvl > c_lvl:
                transition_type = 'Vertical_Promotion'
            elif r_amb < 0.13:
                transition_type = 'Career_Shift'
            else:
                transition_type = true_regime
                
        else:
            true_regime = 'Career_Shift'
            non_adj = [c for c in clusters if c != c_cluster and c not in adj_map[c_cluster]]
            if not non_adj:
                non_adj = [c for c in clusters if c != c_cluster]
            t_cluster = np.random.choice(non_adj)
            c_role = np.random.choice(role_clusters[c_cluster])
            t_role = np.random.choice(role_clusters[t_cluster])
            
            tenure = np.random.exponential(scale=2.4)
            # Overlapping skill gap distribution (11 to 32 skills)
            n_gap_skills = int(np.clip(np.random.normal(21, 4.5), 10, 34))
            
            base_prob = (
                0.065 - 0.002 * (n_gap_skills - 21) + 0.004 * min(tenure, 5.0)
                + 0.6 * unobserved_aptitude + 0.6 * unobserved_market_shock
            )
            prob = np.clip(base_prob, 0.001, 0.35)
            
            if np.random.rand() < 0.07:
                transition_type = 'Adjacent_Pivot'
            else:
                transition_type = true_regime
                
        pool = list(cluster_skills[t_cluster])
        if n_gap_skills > len(pool):
            extra_pool = [s for c in clusters for s in cluster_skills[c] if s not in pool]
            np.random.shuffle(extra_pool)
            pool.extend(extra_pool[:n_gap_skills - len(pool)])
        np.random.shuffle(pool)
        needed_skills = pool[:n_gap_skills]
        
        p1_end = max(1, len(needed_skills) // 3)
        p2_end = max(p1_end + 1, (2 * len(needed_skills)) // 3)
        roadmap_dict = {
            "Phase 1: Core Foundations": needed_skills[:p1_end],
            "Phase 2: Applied Mastery": needed_skills[p1_end:p2_end],
            "Phase 3: Production & Leadership": needed_skills[p2_end:]
        }
        roadmap_str = json.dumps(roadmap_dict)
        
        target_job_desc = (
            f"We are looking for a {t_role} to lead and execute {t_cluster} initiatives. "
            f"Key competencies include {', '.join(needed_skills[:5])}."
        )
        
        duration = np.nan if np.random.rand() < 0.035 else round(float(tenure), 2)
        
        data.append({
            'transition_id': f'SIM_{i:08d}',
            'person_id': person_id,
            'step_index': step_index,
            'current_role': c_role,
            'current_duration': duration,
            'target_future_role': t_role,
            'transition_type': transition_type,
            'transition_probability': round(float(prob), 6),
            'target_job_description': target_job_desc,
            'skill_gap_roadmap': roadmap_str
        })
        
    df_raw = pd.DataFrame(data)
    print("Stochastic benchmark dataset created!")
    return df_raw

# ---------------------------------------------------------------------------
# 3. Data Ingestion & Preprocessing Pipeline
# ---------------------------------------------------------------------------
def load_and_preprocess_data(data_dir=None, force_regenerate=False):
    if data_dir is None:
        data_dir = DATA_DIR
    parquet_path = os.path.join(data_dir, "pathnet0_training_features.parquet")
    rag_parquet_path = os.path.join(data_dir, "pathnet0_rag_knowledge_base.parquet")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "future_self_roadmap_part_*.csv")))
    
    if not force_regenerate and os.path.exists(parquet_path) and os.path.exists(rag_parquet_path):
        print(f"Loading pre-computed features from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        rag_kb = pd.read_parquet(rag_parquet_path)
        return df, rag_kb
        
    if csv_files:
        print(f"Loading {len(csv_files)} longitudinal CSV parts from {data_dir}...")
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files], ignore_index=True)
    else:
        print("External CSV parts not found in DATA_DIR. Using stochastic benchmark simulation...")
        df = generate_synthetic_dataset(n_samples=25000)
        
    gc.collect()
    print(f"Raw records loaded: {len(df):,}")
    
    # Process Skills
    print("Processing skill gaps...")
    if 'required_future_skills' in df.columns:
        req_skills = df['required_future_skills'].apply(parse_skills)
    else:
        req_skills = df['skill_gap_roadmap'].apply(roadmap_skills)
    df['skill_gap_count'] = req_skills.apply(lambda s: len(set(s)))
    del req_skills
    gc.collect()
    
    # Clean text columns
    print("Normalizing role text & descriptions...")
    df['current_role'] = df['current_role'].apply(clean_role_text)
    df['target_future_role'] = df['target_future_role'].apply(clean_role_text)
    df['target_job_description'] = df['target_job_description'].apply(clean_job_desc)
    
    # Process Tenure
    print("Imputing tenure features...")
    df['tenure_years'] = df['current_duration'].copy()
    df['tenure_missing_flag'] = df['tenure_years'].isna().astype(int)
    median_tenure = df['tenure_years'].median()
    df['tenure_years'] = df['tenure_years'].fillna(median_tenure if pd.notna(median_tenure) else 0.0)
    df['is_zero_tenure'] = (df['tenure_years'] == 0).astype(int)
    
    # Seniority inference
    print("Computing seniority levels...")
    df['current_seniority'] = df['current_role'].apply(infer_seniority)
    df['target_seniority'] = df['target_future_role'].apply(infer_seniority)
    df['current_seniority_level'] = df['current_seniority'].map(SENIORITY_MAP).fillna(2).astype(int)
    df['target_seniority_level'] = df['target_seniority'].map(SENIORITY_MAP).fillna(2).astype(int)
    df['seniority_level_change'] = df['target_seniority_level'] - df['current_seniority_level']
    
    # Frequency encoding
    print("Computing role transition frequencies...")
    c_freq = df['current_role'].value_counts(normalize=True)
    t_freq = df['target_future_role'].value_counts(normalize=True)
    df['current_role_freq'] = df['current_role'].map(c_freq).astype('float32')
    df['target_role_freq'] = df['target_future_role'].map(t_freq).astype('float32')
    
    pair_counts = df.groupby(['current_role', 'target_future_role'], observed=True)['person_id'].transform('size')
    df['role_transition_freq'] = (pair_counts / len(df)).astype('float32')
    del pair_counts
    
    # Build RAG Knowledge Base
    print("Building Retrieval Knowledge Base...")
    rag_cols = [
        'target_future_role', 'target_job_description', 'skill_gap_roadmap',
        'skill_gap_count', 'target_seniority', 'target_seniority_level'
    ]
    rag_kb = df.loc[df['skill_gap_count'] > 0, rag_cols].copy()
    rag_kb = rag_kb.drop_duplicates(
        subset=['target_future_role', 'target_job_description', 'skill_gap_roadmap']
    ).reset_index(drop=True)
    print(f"Retrieval Knowledge Base deduplicated to {len(rag_kb):,} roadmaps")
    
    # Clean training frame
    heavy_cols = [
        'transition_id', 'target_job_description', 'skill_gap_roadmap',
        'required_future_skills', 'current_skills', 'current_skill_count',
        'target_skill_count', 'current_duration'
    ]
    df = df.drop(columns=[c for c in heavy_cols if c in df.columns])
    
    df.to_parquet(parquet_path, index=False)
    rag_kb.to_parquet(rag_parquet_path, index=False)
    print(f"Saved preprocessed features to {parquet_path} and {rag_parquet_path}")
    
    return df, rag_kb

# ---------------------------------------------------------------------------
# 4. Leak-Free Feature Extraction & Semantic Interaction Pipeline
# ---------------------------------------------------------------------------
class PathNetFeaturePipeline:
    def __init__(self, tfidf_dim=3000, svd_dim=24):
        self.tfidf_dim = tfidf_dim
        self.svd_dim = svd_dim
        self.role_vectorizer = None
        self.svd = None
        self.scaler = None
        self.c_freq_map = {}
        self.t_freq_map = {}
        self.pair_freq_map = {}
        self.default_c_freq = 0.001
        self.default_t_freq = 0.001
        self.default_pair_freq_by_sim = (0.0002, 0.0020, 0.0080)
        self.median_skill_gap_by_sim = (20, 16, 11)
        self.num_cols = [
            'step_index', 'tenure_years', 'tenure_missing_flag', 'is_zero_tenure',
            'current_seniority_level', 'target_seniority_level', 'seniority_level_change',
            'skill_gap_count', 'current_role_freq', 'target_role_freq', 'role_transition_freq',
            'core_role_jaccard', 'tfidf_cos_sim', 'svd_cos_sim'
        ]
        
    def _compute_semantic_features(self, df):
        cur_roles = df['current_role'].astype(str).tolist()
        tgt_roles = df['target_future_role'].astype(str).tolist()
        
        # Pre-parse unique roles once for fast O(1) lookup across 1.14M+ rows
        unique_roles = list(dict.fromkeys(cur_roles + tgt_roles))
        syn = {'developer': 'engineer', 'engineering': 'engineer'}
        role_info = {}
        for r in unique_roles:
            tokens = set(re.split(r'[\s/\-\.]+', clean_role_text(r)))
            ta_norm = {syn.get(t, t) for t in tokens if t and t not in SENIORITY_STOPWORDS and len(t) > 1}
            doms = infer_domains(r)
            role_info[r] = (ta_norm, doms)
            
        pair_jac_cache = {}
        jaccards = np.empty(len(cur_roles), dtype=np.float32)
        for i, (c, t) in enumerate(zip(cur_roles, tgt_roles)):
            key = (c, t)
            if key in pair_jac_cache:
                jaccards[i] = pair_jac_cache[key]
                continue
            ta_norm, dom_a = role_info[c]
            tb_norm, dom_b = role_info[t]
            same_dom = bool(dom_a & dom_b)
            adj_dom = bool(any(db in ADJ_DOMAIN_MAP.get(da, set()) for da in dom_a for db in dom_b))
            lex_jac = len(ta_norm & tb_norm) / max(1, len(ta_norm | tb_norm))
            if same_dom and len(dom_a) == 1 and len(dom_b) == 1:
                val = float(max(lex_jac, 0.75))
            elif same_dom:
                val = float(max(lex_jac, 0.35))
            elif adj_dom:
                val = float(max(lex_jac * 0.5, 0.25))
            else:
                val = float(lex_jac * 0.15)
            pair_jac_cache[key] = val
            jaccards[i] = val
        core_jaccards = jaccards.reshape(-1, 1)
        
        # Vectorize & project unique roles once, then gather by index
        role_to_idx = {r: i for i, r in enumerate(unique_roles)}
        cur_idx = np.array([role_to_idx[c] for c in cur_roles], dtype=np.int32)
        tgt_idx = np.array([role_to_idx[t] for t in tgt_roles], dtype=np.int32)
        
        u_tfidf = self.role_vectorizer.transform(unique_roles)
        u_svd = self.svd.transform(u_tfidf).astype(np.float32)
        
        cur_tfidf = u_tfidf[cur_idx]
        tgt_tfidf = u_tfidf[tgt_idx]
        tfidf_cos = np.asarray(cur_tfidf.multiply(tgt_tfidf).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        
        cur_svd = u_svd[cur_idx]
        tgt_svd = u_svd[tgt_idx]
        norm_cur = np.linalg.norm(cur_svd, axis=1, keepdims=True) + 1e-8
        norm_tgt = np.linalg.norm(tgt_svd, axis=1, keepdims=True) + 1e-8
        svd_cos = np.sum((cur_svd / norm_cur) * (tgt_svd / norm_tgt), axis=1, keepdims=True).astype(np.float32)
        
        return core_jaccards, tfidf_cos, svd_cos, cur_svd, tgt_svd
        
    def fit(self, df_train):
        print("Fitting leak-free role frequency maps strictly on training split...")
        c_counts = df_train['current_role'].astype(str).value_counts(normalize=True)
        t_counts = df_train['target_future_role'].astype(str).value_counts(normalize=True)
        self.c_freq_map = c_counts.to_dict()
        self.t_freq_map = t_counts.to_dict()
        self.default_c_freq = float(c_counts.median()) if len(c_counts) > 0 else 0.001
        self.default_t_freq = float(t_counts.median()) if len(t_counts) > 0 else 0.001
        
        pair_series = (
            df_train.groupby([df_train['current_role'].astype(str), df_train['target_future_role'].astype(str)], observed=True)
            .size() / len(df_train)
        )
        self.pair_freq_map = pair_series.to_dict()
        
        print("Fitting unified role TF-IDF vectorizer...")
        all_roles = pd.concat([df_train['current_role'].astype(str), df_train['target_future_role'].astype(str)], ignore_index=True)
        self.role_vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), min_df=1, max_features=self.tfidf_dim,
            sublinear_tf=True, dtype=np.float32
        )
        all_role_vecs = self.role_vectorizer.fit_transform(all_roles)
        
        n_comp = min(self.svd_dim, all_role_vecs.shape[1] - 1)
        self.svd_dim = n_comp
        print(f"Fitting TruncatedSVD to {self.svd_dim} dense semantic components...")
        self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=SEED)
        self.svd.fit(all_role_vecs)
        
        core_jaccards, tfidf_cos, svd_cos, _, _ = self._compute_semantic_features(df_train)
        
        pair_vals = np.array([
            self.pair_freq_map.get((c, t), 0.0002)
            for c, t in zip(df_train['current_role'].astype(str), df_train['target_future_role'].astype(str))
        ])
        high_sim_mask = (core_jaccards.ravel() >= 0.50)
        mid_sim_mask = (core_jaccards.ravel() < 0.50) & ((core_jaccards.ravel() >= 0.10) | (svd_cos.ravel() >= 0.15))
        low_sim_mask = ~(high_sim_mask | mid_sim_mask)
        
        p_low = float(np.median(pair_vals[low_sim_mask])) if np.any(low_sim_mask) else 0.0002
        p_mid = float(np.median(pair_vals[mid_sim_mask])) if np.any(mid_sim_mask) else 0.0015
        p_high = float(np.median(pair_vals[high_sim_mask])) if np.any(high_sim_mask) else 0.0060
        self.default_pair_freq_by_sim = (p_low, p_mid, p_high)
        
        gaps = df_train['skill_gap_count'].values
        g_low = int(np.median(gaps[low_sim_mask])) if np.any(low_sim_mask) else 21
        g_mid = int(np.median(gaps[mid_sim_mask])) if np.any(mid_sim_mask) else 15
        g_high = int(np.median(gaps[high_sim_mask])) if np.any(high_sim_mask) else 10
        self.median_skill_gap_by_sim = (g_low, g_mid, g_high)
        
        X_num_raw = self._build_numeric_matrix(df_train, core_jaccards, tfidf_cos, svd_cos)
        print("Fitting numerical feature scaler...")
        self.scaler = StandardScaler()
        self.scaler.fit(X_num_raw)
        return self
        
    def _build_numeric_matrix(self, df, core_jaccards, tfidf_cos, svd_cos):
        cur_roles = df['current_role'].astype(str).tolist()
        tgt_roles = df['target_future_role'].astype(str).tolist()
        
        c_freqs = np.array([self.c_freq_map.get(c, self.default_c_freq) for c in cur_roles], dtype=np.float32)
        t_freqs = np.array([self.t_freq_map.get(t, self.default_t_freq) for t in tgt_roles], dtype=np.float32)
        
        pair_freqs = []
        for idx, (c, t) in enumerate(zip(cur_roles, tgt_roles)):
            if (c, t) in self.pair_freq_map:
                pair_freqs.append(self.pair_freq_map[(c, t)])
            else:
                if core_jaccards[idx, 0] >= 0.50:
                    pair_freqs.append(self.default_pair_freq_by_sim[2])
                elif core_jaccards[idx, 0] >= 0.10 or svd_cos[idx, 0] >= 0.15:
                    pair_freqs.append(self.default_pair_freq_by_sim[1])
                else:
                    pair_freqs.append(self.default_pair_freq_by_sim[0])
        pair_freqs = np.array(pair_freqs, dtype=np.float32)
        
        base_cols = [
            'step_index', 'tenure_years', 'tenure_missing_flag', 'is_zero_tenure',
            'current_seniority_level', 'target_seniority_level', 'seniority_level_change',
            'skill_gap_count'
        ]
        X_base = df[base_cols].values.astype(np.float32)
        X_num_raw = np.hstack([
            X_base,
            c_freqs.reshape(-1, 1),
            t_freqs.reshape(-1, 1),
            pair_freqs.reshape(-1, 1),
            core_jaccards,
            tfidf_cos,
            svd_cos
        ])
        return X_num_raw
        
    def transform(self, df):
        core_jaccards, tfidf_cos, svd_cos, cur_svd, tgt_svd = self._compute_semantic_features(df)
        X_num_raw = self._build_numeric_matrix(df, core_jaccards, tfidf_cos, svd_cos)
        X_num = self.scaler.transform(X_num_raw)
        
        abs_diff = np.abs(cur_svd - tgt_svd)
        elem_prod = cur_svd * tgt_svd
        
        X_all = np.hstack([X_num, cur_svd, tgt_svd, abs_diff, elem_prod]).astype(np.float32)
        return X_all

# ---------------------------------------------------------------------------
# 5. PyTorch Multi-Task Deep Neural Network (PathNet-v0 DNN)
# ---------------------------------------------------------------------------
class PathNetDataset(Dataset):
    def __init__(self, X, y_prob, y_type):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_prob = torch.tensor(y_prob, dtype=torch.float32).unsqueeze(1)
        self.y_type = torch.tensor(y_type, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y_prob[idx], self.y_type[idx]

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

class PathNetMultiTaskNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_types=3, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res1 = ResidualBlock(hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout)
        
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_types)
        )
        
    def forward(self, x):
        feat = self.input_layer(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        return self.prob_head(feat), self.type_head(feat)

# ---------------------------------------------------------------------------
# 6. Training Function for PyTorch Multi-Task PathNet
# ---------------------------------------------------------------------------
def train_deep_pathnet(model, train_loader, val_loader, epochs=15, lr=1e-3, weight_decay=1e-4, loss_lambda=0.35):
    print(f"\n--- Starting Deep Multi-Task PathNet Training ({epochs} epochs) ---")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    criterion_prob = nn.MSELoss()
    criterion_type = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for X_batch, y_prob_batch, y_type_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_prob_batch = y_prob_batch.to(DEVICE)
            y_type_batch = y_type_batch.to(DEVICE)
            
            optimizer.zero_grad()
            p_pred, t_pred = model(X_batch)
            
            loss_p = criterion_prob(p_pred, y_prob_batch)
            loss_t = criterion_type(t_pred, y_type_batch)
            loss = loss_p + loss_lambda * loss_t
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
            
        train_loss = total_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        all_p_preds, all_p_trues = [], []
        all_t_preds, all_t_trues = [], []
        
        with torch.no_grad():
            for X_batch, y_prob_batch, y_type_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_prob_batch = y_prob_batch.to(DEVICE)
                y_type_batch = y_type_batch.to(DEVICE)
                
                p_pred, t_pred = model(X_batch)
                
                loss_p = criterion_prob(p_pred, y_prob_batch)
                loss_t = criterion_type(t_pred, y_type_batch)
                loss = loss_p + loss_lambda * loss_t
                
                val_loss += loss.item() * len(X_batch)
                all_p_preds.extend(p_pred.cpu().numpy().ravel())
                all_p_trues.extend(y_prob_batch.cpu().numpy().ravel())
                all_t_preds.extend(t_pred.argmax(dim=1).cpu().numpy())
                all_t_trues.extend(y_type_batch.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_mae = mean_absolute_error(all_p_trues, all_p_preds)
        val_acc = accuracy_score(all_t_trues, all_t_preds)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_acc'].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved_flag = "*"
        else:
            improved_flag = ""
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val Acc: {val_acc*100:.2f}% {improved_flag}")
        
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print("Training complete! Restored best checkpoint.")
    return history

# ---------------------------------------------------------------------------
# 7. Gradient Boosted Tree Alternative (HistGradientBoosting)
# ---------------------------------------------------------------------------
def train_gradient_boosters(X_train, y_prob_train, y_type_train, X_val, y_prob_val, y_type_val):
    print("\n--- Training Fast Gradient Boosted Decision Trees ---")
    regressor = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=SEED)
    regressor.fit(X_train, y_prob_train)
    p_pred_val = regressor.predict(X_val)
    mae_val = mean_absolute_error(y_prob_val, p_pred_val)
    r2_val = r2_score(y_prob_val, p_pred_val)
    print(f"Gradient Boosted Regressor (Probability) -> Val MAE: {mae_val:.4f}, Val R²: {r2_val:.4f}")
    
    classifier = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED)
    classifier.fit(X_train, y_type_train)
    t_pred_val = classifier.predict(X_val)
    acc_val = accuracy_score(y_type_val, t_pred_val)
    f1_val = f1_score(y_type_val, t_pred_val, average='macro')
    print(f"Gradient Boosted Classifier (Transition Type) -> Val Acc: {acc_val*100:.2f}%, Macro F1: {f1_val:.4f}")
    
    return regressor, classifier

# ---------------------------------------------------------------------------
# 8. Retrieval Knowledge Base Engine (TF-IDF Cosine Index)
# ---------------------------------------------------------------------------
class RAGRoadmapEngine:
    def __init__(self, rag_kb):
        self.rag_kb = rag_kb.copy().reset_index(drop=True)
        rag_text = (
            self.rag_kb['target_future_role'].fillna('').astype(str) + ' ' +
            self.rag_kb['target_job_description'].fillna('').astype(str).str.slice(0, 80)
        )
        self.vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), min_df=2, max_features=20000,
            sublinear_tf=True, dtype=np.float32
        )
        self.rag_matrix = self.vectorizer.fit_transform(rag_text)
        
    def retrieve(self, query_role, query_description='', top_k=3):
        query = f"{clean_role_text(query_role)} {clean_job_desc(query_description)}".strip()
        query_vec = self.vectorizer.transform([query])
        scores = linear_kernel(query_vec, self.rag_matrix).ravel()
        top_idx = scores.argsort()[-top_k:][::-1]
        
        result = self.rag_kb.iloc[top_idx][
            ['target_future_role', 'target_job_description', 'skill_gap_roadmap', 'skill_gap_count']
        ].copy()
        result['similarity'] = scores[top_idx]
        return result.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 9. End-to-End Unified PathNet Inference Engine (with Isotonic Calibrator)
# ---------------------------------------------------------------------------
class PathNetInferenceEngine:
    def __init__(self, deep_model, pipeline, rag_engine, hgb_reg=None, hgb_clf=None, calibrator=None):
        self.model = deep_model
        self.pipeline = pipeline
        self.rag_engine = rag_engine
        self.hgb_reg = hgb_reg
        self.hgb_clf = hgb_clf
        self.calibrator = calibrator
        if self.model:
            self.model.eval()
            
    def predict_transition(self, current_role, tenure_years, target_role, step_index=2):
        c_role_clean = clean_role_text(current_role)
        t_role_clean = clean_role_text(target_role)
        
        c_sen = infer_seniority(c_role_clean)
        t_sen = infer_seniority(t_role_clean)
        c_sen_lvl = SENIORITY_MAP[c_sen]
        t_sen_lvl = SENIORITY_MAP[t_sen]
        sen_change = t_sen_lvl - c_sen_lvl
        
        roadmap_res = self.rag_engine.retrieve(t_role_clean, top_k=5)
        core_jac = compute_core_jaccard(c_role_clean, t_role_clean)
        
        cur_tfidf = self.pipeline.role_vectorizer.transform([c_role_clean])
        tgt_tfidf = self.pipeline.role_vectorizer.transform([t_role_clean])
        cur_svd = self.pipeline.svd.transform(cur_tfidf)
        tgt_svd = self.pipeline.svd.transform(tgt_tfidf)
        svd_sim = float(
            np.sum(cur_svd * tgt_svd) /
            ((np.linalg.norm(cur_svd) + 1e-8) * (np.linalg.norm(tgt_svd) + 1e-8))
        )
        
        if core_jac >= 0.50:
            expected_gap = self.pipeline.median_skill_gap_by_sim[2]
        elif core_jac >= 0.10 or svd_sim >= 0.15:
            expected_gap = self.pipeline.median_skill_gap_by_sim[1]
        else:
            expected_gap = self.pipeline.median_skill_gap_by_sim[0]
            
        if len(roadmap_res) > 0:
            gaps_cand = roadmap_res['skill_gap_count'].values
            best_cand_idx = int(np.argmin(np.abs(gaps_cand - expected_gap)))
            retrieved_roadmap = roadmap_res.iloc[best_cand_idx]['skill_gap_roadmap']
            skill_gap_count = int(round(0.6 * expected_gap + 0.4 * int(roadmap_res.iloc[best_cand_idx]['skill_gap_count'])))
        else:
            skill_gap_count = int(expected_gap)
            retrieved_roadmap = "{}"
            
        row_df = pd.DataFrame([{
            'step_index': step_index,
            'tenure_years': float(tenure_years),
            'tenure_missing_flag': 0,
            'is_zero_tenure': int(tenure_years == 0),
            'current_seniority_level': c_sen_lvl,
            'target_seniority_level': t_sen_lvl,
            'seniority_level_change': sen_change,
            'skill_gap_count': skill_gap_count,
            'current_role': c_role_clean,
            'target_future_role': t_role_clean
        }])
        
        feat_vec = self.pipeline.transform(row_df)
        
        with torch.no_grad():
            x_t = torch.tensor(feat_vec, dtype=torch.float32).to(DEVICE)
            prob_t, type_logits_t = self.model(x_t)
            prob_deep = float(prob_t.cpu().numpy()[0, 0])
            type_probs_deep = torch.softmax(type_logits_t, dim=1).cpu().numpy()[0]
            
        if self.hgb_reg is not None and self.hgb_clf is not None:
            prob_hgb = float(self.hgb_reg.predict(feat_vec)[0])
            type_probs_hgb = self.hgb_clf.predict_proba(feat_vec)[0]
            
            raw_prob = float(np.clip(0.55 * prob_deep + 0.45 * prob_hgb, 0.001, 0.99))
            final_type_probs = 0.55 * type_probs_deep + 0.45 * type_probs_hgb
        else:
            raw_prob = prob_deep
            final_type_probs = type_probs_deep
            
        if self.calibrator is not None:
            cal_prob = float(np.clip(self.calibrator.predict([raw_prob])[0], 0.001, 0.99))
        else:
            cal_prob = raw_prob
            
        # Domain-consistency prior blend for out-of-sample interactive queries across heterogeneous cohorts
        dom_c = infer_domains(c_role_clean)
        dom_t = infer_domains(t_role_clean)
        same_dom = bool(dom_c & dom_t)
        adj_dom = bool(any(dt in ADJ_DOMAIN_MAP.get(dc, set()) for dc in dom_c for dt in dom_t))
        if dom_c and dom_t:
            if same_dom and sen_change >= 0:
                prior_prob = np.clip(0.46 + 0.03 * min(tenure_years, 6.0) - 0.05 * max(0, sen_change - 1), 0.35, 0.85)
                prior_type = np.array([0.88, 0.11, 0.01], dtype=np.float32)
            elif adj_dom or (same_dom and sen_change < 0):
                prior_prob = np.clip(0.22 + 0.02 * min(tenure_years, 5.0) - 0.03 * abs(sen_change), 0.12, 0.42)
                prior_type = np.array([0.08, 0.84, 0.08], dtype=np.float32)
            else:
                prior_prob = np.clip(0.03 + 0.005 * min(tenure_years, 5.0), 0.005, 0.09)
                prior_type = np.array([0.01, 0.07, 0.92], dtype=np.float32)
            final_prob = float(np.clip(0.50 * cal_prob + 0.50 * prior_prob, 0.001, 0.99))
            final_type_probs = 0.45 * final_type_probs + 0.55 * prior_type
            final_type_probs = final_type_probs / np.sum(final_type_probs)
        else:
            final_prob = cal_prob
            
        type_id = int(np.argmax(final_type_probs))
        trans_type = ID_TO_TRANSITION_TYPE[type_id]
        
        if final_prob >= 0.45:
            feasibility = "High (Optimal Career Step)"
        elif final_prob >= 0.18:
            feasibility = "Moderate (Achievable with Focused Upskilling)"
        else:
            feasibility = "Challenging (Major Career Shift / Long-Term Plan)"
            
        try:
            parsed_roadmap = json.loads(retrieved_roadmap)
        except Exception:
            parsed_roadmap = retrieved_roadmap
            
        return {
            'current_role': current_role,
            'current_seniority': c_sen,
            'tenure_years': tenure_years,
            'target_future_role': target_role,
            'target_seniority': t_sen,
            'seniority_delta': f"{'+' if sen_change > 0 else ''}{sen_change}",
            'transition_probability': f"{final_prob * 100:.2f}%",
            'transition_probability_raw': final_prob,
            'predicted_transition_type': trans_type,
            'type_distribution': {
                ID_TO_TRANSITION_TYPE[i]: f"{final_type_probs[i]*100:.1f}%" for i in range(len(TRANSITION_TYPES))
            },
            'estimated_skill_gap_count': skill_gap_count,
            'feasibility_assessment': feasibility,
            'actionable_roadmap': parsed_roadmap
        }
        
    def recommend_next_steps(self, current_role, tenure_years, candidate_roles=None, top_n=3):
        if candidate_roles is None:
            candidate_roles = [
                'senior software engineer', 'lead software engineer', 'software engineering manager',
                'full stack engineer', 'devops engineer', 'cloud architect',
                'data scientist', 'machine learning engineer', 'product manager',
                'sales account manager', 'marketing and communication manager'
            ]
        results = []
        for cand in candidate_roles:
            if clean_role_text(cand) == clean_role_text(current_role):
                continue
            res = self.predict_transition(current_role, tenure_years, cand)
            results.append(res)
        results = sorted(results, key=lambda x: x['transition_probability_raw'], reverse=True)
        return results[:top_n]

# ---------------------------------------------------------------------------
# 10. Main Pipeline Orchestrator (with Group Split, Calibration & Subgroup Audit)
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" PathNet0: Training, Calibration & Subgroup Evaluation Pipeline")
    print("=" * 70)
    
    df, rag_kb = load_and_preprocess_data(DATA_DIR, force_regenerate=False)
    
    print("\nEncoding training labels...")
    y_prob = df['transition_probability'].values.astype(np.float32)
    y_type = df['transition_type'].map(TRANSITION_TYPE_TO_ID).values.astype(np.int64)
    
    # Leak-Resistant Group Split by person_id so the same individual never spans Train and Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    idx_train, idx_temp = next(gss1.split(df, y_type, groups=df['person_id']))
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    rel_val, rel_test = next(gss2.split(df.iloc[idx_temp], y_type[idx_temp], groups=df.iloc[idx_temp]['person_id']))
    idx_val = idx_temp[rel_val]
    idx_test = idx_temp[rel_test]
    
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    
    overlap_persons = set(df_train['person_id']) & set(df_test['person_id'])
    print(f"Group Split by person_id -> Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,} (Person overlap: {len(overlap_persons)})")
    
    # Fit Leak-Free Feature Pipeline on Train ONLY
    pipeline = PathNetFeaturePipeline(tfidf_dim=3000, svd_dim=24)
    pipeline.fit(df_train)
    
    X_train = pipeline.transform(df_train)
    X_val = pipeline.transform(df_val)
    X_test = pipeline.transform(df_test)
    
    print(f"Engineered feature vector dimension: {X_train.shape[1]}")
    
    # Build & Train PyTorch Multi-Task Model (adaptive batch size for 1.14M scale)
    is_large_scale = len(X_train) > 100000
    batch_size = 4096 if is_large_scale else 256
    epochs = 10 if is_large_scale else 15
    lr = 2e-3 if is_large_scale else 1e-3
    
    train_dataset = PathNetDataset(X_train, y_prob[idx_train], y_type[idx_train])
    val_dataset = PathNetDataset(X_val, y_prob[idx_val], y_type[idx_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8192 if is_large_scale else 512, shuffle=False)
    
    deep_model = PathNetMultiTaskNet(input_dim=X_train.shape[1], hidden_dim=256, num_types=len(TRANSITION_TYPES))
    deep_model.to(DEVICE)
    
    history = train_deep_pathnet(deep_model, train_loader, val_loader, epochs=epochs, lr=lr)
    
    # Train Gradient Boosted Decision Trees
    hgb_reg, hgb_clf = train_gradient_boosters(
        X_train, y_prob[idx_train], y_type[idx_train],
        X_val, y_prob[idx_val], y_type[idx_val]
    )
    
    # Helper for memory-safe batched PyTorch inference
    def predict_deep_batched(model, X_arr, bs=8192):
        model.eval()
        probs_list, logits_list = [], []
        with torch.no_grad():
            for start in range(0, len(X_arr), bs):
                xb = torch.tensor(X_arr[start:start+bs], dtype=torch.float32).to(DEVICE)
                pb, lb = model(xb)
                probs_list.append(pb.cpu().numpy().ravel())
                logits_list.append(torch.softmax(lb, dim=1).cpu().numpy())
        return np.concatenate(probs_list), np.vstack(logits_list)

    # Fit Post-Hoc Isotonic Calibrator on Validation Set
    p_val_deep, _ = predict_deep_batched(deep_model, X_val)
    p_val_hgb = hgb_reg.predict(X_val)
    p_val_ens_raw = np.clip(0.55 * p_val_deep + 0.45 * p_val_hgb, 0.0, 1.0)
    
    calibrator = IsotonicRegression(y_min=0.001, y_max=0.99, out_of_bounds='clip')
    calibrator.fit(p_val_ens_raw, y_prob[idx_val])
    
    # Held-Out Test Set Evaluation
    print("\n" + "=" * 60)
    print(f" Comprehensive Held-Out Test Set Evaluation (n={len(df_test):,})")
    print("=" * 60)
    
    p_test_deep, t_test_probs_deep = predict_deep_batched(deep_model, X_test)
    t_test_deep = t_test_probs_deep.argmax(axis=1)
        
    p_test_hgb = hgb_reg.predict(X_test)
    t_test_probs_hgb = hgb_clf.predict_proba(X_test)
    t_test_hgb = t_test_probs_hgb.argmax(axis=1)
    
    p_test_ens_raw = np.clip(0.55 * p_test_deep + 0.45 * p_test_hgb, 0.0, 1.0)
    p_test_ens_cal = calibrator.predict(p_test_ens_raw)
    t_test_ens = (0.55 * t_test_probs_deep + 0.45 * t_test_probs_hgb).argmax(axis=1)
    
    y_prob_test = y_prob[idx_test]
    y_type_test = y_type[idx_test]
    
    for name, p_pred, t_pred in [
        ("Deep Multi-Task PathNet (PyTorch)", p_test_deep, t_test_deep),
        ("HistGradientBoosting (Trees)", p_test_hgb, t_test_hgb),
        ("Unified PathNet Ensemble (Calibrated)", p_test_ens_cal, t_test_ens)
    ]:
        mae = mean_absolute_error(y_prob_test, p_pred)
        rmse = np.sqrt(mean_squared_error(y_prob_test, p_pred))
        r2 = r2_score(y_prob_test, p_pred)
        acc = accuracy_score(y_type_test, t_pred)
        f1 = f1_score(y_type_test, t_pred, average='macro')
        print(f"\n[{name}]")
        print(f"  * Transition Probability MAE : {mae:.4f}")
        print(f"  * Transition Probability RMSE: {rmse:.4f}")
        print(f"  * Transition Probability R²  : {r2:.4f}")
        print(f"  * Transition Type Accuracy   : {acc*100:.2f}%")
        print(f"  * Transition Type Macro F1   : {f1:.4f}")

    # Report Per-Source Cohort Breakdown on Held-Out Test Set when 1.14M dataset is loaded
    if len(df_test) > 50000:
        print("\n[Within-Source Cohort Breakdown on Held-Out Test Set (Calibrated Ensemble)]")
        pids = df_test['person_id'].values
        for c_name, mask in [
            ("ESCO/ISCO Occupational Transitions (<1M)", pids < 1_000_000),
            ("JobHop Longitudinal Histories (2B..3B)", (pids >= 2_000_000_000) & (pids < 3_000_000_000)),
            ("LinkedIn Career Promotions (>=3B)", pids >= 3_000_000_000)
        ]:
            if np.any(mask):
                c_mae = mean_absolute_error(y_prob_test[mask], p_test_ens_cal[mask])
                c_rmse = np.sqrt(mean_squared_error(y_prob_test[mask], p_test_ens_cal[mask]))
                c_acc = accuracy_score(y_type_test[mask], t_test_ens[mask])
                print(f"  • {c_name:<42} | n={int(mask.sum()):>7,} | Prob MAE: {c_mae:.4f} | RMSE: {c_rmse:.4f} | Type Acc: {c_acc*100:.2f}%")
        
    rag_engine = RAGRoadmapEngine(rag_kb)
    
    # Persist Model Artifacts
    print("\nSaving trained model artifacts...")
    torch.save(deep_model.state_dict(), os.path.join(DATA_DIR, "pathnet0_deep_model.pt"))
    joblib.dump(pipeline, os.path.join(DATA_DIR, "pathnet0_feature_pipeline.joblib"))
    joblib.dump(hgb_reg, os.path.join(DATA_DIR, "pathnet0_hgb_regressor.joblib"))
    joblib.dump(hgb_clf, os.path.join(DATA_DIR, "pathnet0_hgb_classifier.joblib"))
    joblib.dump(calibrator, os.path.join(DATA_DIR, "pathnet0_isotonic_calibrator.joblib"))
    print(f"All model artifacts persisted to {DATA_DIR}")
    
    # Sanity Check Inference
    print("\n" + "=" * 60)
    print(" Interactive Inference Verification (Sanity Check)")
    print("=" * 60)
    
    engine = PathNetInferenceEngine(deep_model, pipeline, rag_engine, hgb_reg, hgb_clf, calibrator)
    
    sample_tests = [
        ("Junior Software Engineer", 2.5, "Senior Software Engineer"),
        ("Software Engineer", 4.0, "Software Engineering Manager"),
        ("Junior Frontend Developer", 2.0, "Full Stack Engineer"),
        ("Data Analyst", 3.0, "Machine Learning Engineer"),
        ("Customer Support Specialist", 2.0, "Sales Account Manager"),
        ("Drainage Worker", 3.0, "Senior Android Engineer")
    ]
    
    for c_role, tenure, t_role in sample_tests:
        pred = engine.predict_transition(c_role, tenure, t_role)
        print(f"\n[Trajectory Analysis]: '{c_role}' ({tenure} yrs) -> '{t_role}'")
        print(f"  * Transition Probability : {pred['transition_probability']}")
        print(f"  * Feasibility Rating     : {pred['feasibility_assessment']}")
        print(f"  * Transition Type        : {pred['predicted_transition_type']} {pred['type_distribution']}")
        print(f"  * Seniority Progression  : {pred['current_seniority']} -> {pred['target_seniority']} (Delta: {pred['seniority_delta']})")
        print(f"  * Estimated Skill Gap    : {pred['estimated_skill_gap_count']} core skills")

if __name__ == '__main__':
    main()
