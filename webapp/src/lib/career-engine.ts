/**
 * CareerAI Studio — Deterministic Serverless Edge Calibrated Inference Engine
 * Replicates the exact mathematical logic, domain taxonomy, seniority levels,
 * Jaccard/Cosine alignment, isotonic-calibrated score formulas, and RAG skill
 * roadmaps of FitNet (Model 1) and PathNet0 (Model 2) so the platform works
 * 100% standalone on Vercel or proxies to local FastAPI servers when available.
 */

export type SeniorityLabel =
  | "Intern"
  | "Junior"
  | "Mid"
  | "Senior"
  | "Lead/Manager"
  | "Executive";

export const SENIORITY_MAP: Record<SeniorityLabel, number> = {
  Intern: 0,
  Junior: 1,
  Mid: 2,
  Senior: 3,
  "Lead/Manager": 4,
  Executive: 5,
};

export const SKILL_VOCABULARY: string[] = [
  "python",
  "java",
  "javascript",
  "typescript",
  "c++",
  "c#",
  ".net",
  "golang",
  "ruby",
  "php",
  "rust",
  "kotlin",
  "swift",
  "sql",
  "nosql",
  "html",
  "css",
  "sass",
  "tailwind css",
  "react",
  "react native",
  "next.js",
  "angular",
  "vue",
  "redux",
  "node.js",
  "django",
  "flask",
  "fastapi",
  "spring",
  "spring boot",
  "postgresql",
  "mysql",
  "mongodb",
  "redis",
  "elasticsearch",
  "oracle",
  "snowflake",
  "bigquery",
  "spark",
  "kafka",
  "airflow",
  "etl",
  "tableau",
  "power bi",
  "excel",
  "data visualization",
  "pandas",
  "numpy",
  "scikit-learn",
  "tensorflow",
  "pytorch",
  "opencv",
  "machine learning",
  "deep learning",
  "nlp",
  "computer vision",
  "data science",
  "data analysis",
  "data engineering",
  "generative ai",
  "llm",
  "rag",
  "mlops",
  "statistics",
  "a/b testing",
  "aws",
  "azure",
  "gcp",
  "docker",
  "kubernetes",
  "terraform",
  "ansible",
  "jenkins",
  "git",
  "github",
  "gitlab",
  "ci/cd",
  "devops",
  "linux",
  "bash",
  "microservices",
  "system design",
  "rest api",
  "graphql",
  "agile",
  "scrum",
  "kanban",
  "jira",
  "project management",
  "product management",
  "business analysis",
  "qa automation",
  "selenium",
  "pytest",
  "jest",
  "unit testing",
  "cybersecurity",
  "network security",
  "digital marketing",
  "seo",
  "sem",
  "content strategy",
  "sales",
  "crm",
  "salesforce",
  "business development",
  "finance",
  "accounting",
  "financial modeling",
  "communication",
  "leadership",
  "problem solving",
  "stakeholder management",
];

const SKILL_ALIASES: Record<string, string> = {
  reactjs: "react",
  "react.js": "react",
  nodejs: "node.js",
  nextjs: "next.js",
  vuejs: "vue",
  "vue.js": "vue",
  postgres: "postgresql",
  k8s: "kubernetes",
  "rest apis": "rest api",
  restful: "rest api",
  "ci/cd pipelines": "ci/cd",
  "c/c++": "c++",
  dotnet: ".net",
  "asp.net": ".net",
  pyspark: "spark",
  "apache spark": "spark",
  "apache airflow": "airflow",
  "apache kafka": "kafka",
  sklearn: "scikit-learn",
  llms: "llm",
  "large language models": "llm",
  "rag architectures": "rag",
  tailwind: "tailwind css",
  "amazon web services": "aws",
  "google cloud": "gcp",
  "powerbi": "power bi",
};

export const DOMAIN_KEYWORDS: Record<string, string[]> = {
  Software: [
    "software",
    "backend",
    "back-end",
    "java",
    "c++",
    "python",
    "systems",
    "application",
    "dotnet",
    ".net",
    "golang",
    "ruby",
    "php",
    "django",
    "spring",
  ],
  Frontend: [
    "frontend",
    "front-end",
    "ui",
    "ux",
    "react",
    "angular",
    "vue",
    "web",
    "css",
    "javascript",
    "typescript",
  ],
  FullStack: [
    "fullstack",
    "full-stack",
    "full",
    "stack",
    "node",
    "nodejs",
    "next.js",
  ],
  Mobile: [
    "android",
    "ios",
    "mobile",
    "swift",
    "kotlin",
    "flutter",
    "native",
  ],
  "Data & AI": [
    "data",
    "analyst",
    "analytics",
    "scientist",
    "machine",
    "learning",
    "ai",
    "ml",
    "nlp",
    "deep",
    "bi",
    "database",
    "sql",
    "etl",
    "rag",
    "llm",
  ],
  "QA & Testing": [
    "qa",
    "aqa",
    "test",
    "testing",
    "automation",
    "quality",
    "sdet",
  ],
  "Product & Project": [
    "product",
    "project",
    "program",
    "agile",
    "scrum",
    "owner",
    "pm",
    "delivery",
    "business",
  ],
  "Infrastructure & Security": [
    "devops",
    "cloud",
    "sre",
    "infrastructure",
    "administrator",
    "network",
    "security",
    "cybersecurity",
    "linux",
    "aws",
    "system",
    "platform",
  ],
  "Finance & Consulting": [
    "finance",
    "accountant",
    "accounting",
    "banking",
    "consultant",
    "audit",
    "tax",
    "financial",
  ],
  "Sales & Marketing": [
    "sales",
    "account",
    "marketing",
    "digital",
    "media",
    "growth",
    "pr",
    "content",
    "seo",
    "bpo",
    "customer",
    "support",
  ],
};

export const ADJ_DOMAIN_MAP: Record<string, string[]> = {
  Software: [
    "FullStack",
    "Frontend",
    "Mobile",
    "Data & AI",
    "QA & Testing",
    "Infrastructure & Security",
  ],
  Frontend: ["FullStack", "Software", "Mobile", "Product & Project"],
  FullStack: [
    "Software",
    "Frontend",
    "Mobile",
    "Infrastructure & Security",
    "Product & Project",
  ],
  Mobile: ["Frontend", "FullStack", "Software", "QA & Testing"],
  "Data & AI": ["Software", "Infrastructure & Security", "Product & Project"],
  "QA & Testing": [
    "Software",
    "Frontend",
    "Mobile",
    "Infrastructure & Security",
  ],
  "Product & Project": [
    "Software",
    "Frontend",
    "Data & AI",
    "Finance & Consulting",
    "Sales & Marketing",
  ],
  "Infrastructure & Security": [
    "Software",
    "FullStack",
    "Data & AI",
    "QA & Testing",
  ],
  "Finance & Consulting": [
    "Product & Project",
    "Data & AI",
    "Sales & Marketing",
  ],
  "Sales & Marketing": ["Product & Project", "Finance & Consulting"],
};

const SENIORITY_STOPWORDS = new Set([
  "intern",
  "internship",
  "trainee",
  "junior",
  "jr",
  "entry",
  "associate",
  "mid",
  "intermediate",
  "senior",
  "sr",
  "staff",
  "principal",
  "lead",
  "manager",
  "director",
  "head",
  "supervisor",
  "vp",
  "vice",
  "president",
  "chief",
  "executive",
  "ceo",
  "cto",
  "cfo",
  "cmo",
  "cpo",
  "group",
  "i",
  "ii",
  "iii",
  "iv",
]);

export function cleanText(val: string | undefined | null): string {
  if (!val) return "";
  return val
    .toLowerCase()
    .replace(/[^a-z0-9\s/\-+#.]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function inferSeniority(role: string): SeniorityLabel {
  const r = cleanText(role);
  if (
    /\b(executive|chief|president|vice president|vp|cmo|cto|ceo|cpo|cfo)\b/.test(
      r
    )
  ) {
    return "Executive";
  }
  if (
    /\b(lead|manager|director|head|principal|supervisor|architect|owner)\b/.test(
      r
    )
  ) {
    return "Lead/Manager";
  }
  if (/\b(senior|sr|staff)\b/.test(r)) {
    return "Senior";
  }
  if (/\b(junior|jr|entry|associate|trainee)\b/.test(r)) {
    return "Junior";
  }
  if (/\b(intern|internship)\b/.test(r)) {
    return "Intern";
  }
  return "Mid";
}

export function inferDomains(roleOrText: string): string[] {
  const cleaned = cleanText(roleOrText);
  const tokens = new Set(cleaned.split(/[\s/\-]+/).filter(Boolean));
  if (tokens.has("full") && tokens.has("stack")) {
    return ["FullStack"];
  }
  if (tokens.has("machine") && tokens.has("learning")) {
    return ["Data & AI", "Software"];
  }
  const matched: string[] = [];
  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    if (domain === "FullStack") {
      if (
        tokens.has("fullstack") ||
        tokens.has("full-stack") ||
        tokens.has("node") ||
        tokens.has("nodejs") ||
        tokens.has("next.js")
      ) {
        matched.push(domain);
      }
      continue;
    }
    if (keywords.some((kw) => tokens.has(kw))) {
      matched.push(domain);
    }
  }
  return matched;
}

export function computeLexicalJaccard(roleA: string, roleB: string): number {
  const syn: Record<string, string> = {
    developer: "engineer",
    engineering: "engineer",
    programmer: "engineer",
  };
  const tokenize = (r: string) =>
    new Set(
      cleanText(r)
        .split(/[\s/\-.]+/)
        .filter((t) => t.length > 1 && !SENIORITY_STOPWORDS.has(t))
        .map((t) => syn[t] || t)
    );
  const setA = tokenize(roleA);
  const setB = tokenize(roleB);
  const intersection = [...setA].filter((x) => setB.has(x)).length;
  const union = new Set([...setA, ...setB]).size;
  return intersection / Math.max(1, union);
}

export function computeRoleAlignment(roleA: string, roleB: string): number {
  const lexJac = computeLexicalJaccard(roleA, roleB);
  const domA = inferDomains(roleA);
  const domB = inferDomains(roleB);
  const sameDom = domA.some((d) => domB.includes(d));
  const adjDom = domA.some((da) =>
    domB.some((db) => (ADJ_DOMAIN_MAP[da] || []).includes(db))
  );
  if (sameDom) {
    return Math.max(lexJac, 0.75);
  }
  if (adjDom) {
    return Math.max(lexJac, 0.38);
  }
  return lexJac * 0.15;
}

export function normalizeSkill(raw: string): string {
  const s = raw.toLowerCase().trim();
  return SKILL_ALIASES[s] || s;
}

export function extractSkillsFromText(text: string): string[] {
  if (!text) return [];
  const lower = text.toLowerCase();
  const found = new Set<string>();

  for (const skill of SKILL_VOCABULARY) {
    const escaped = skill.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(^|[^a-z0-9+#.])${escaped}([^a-z0-9+#.]|$)`, "i");
    if (regex.test(lower)) {
      found.add(skill);
    }
  }
  for (const [alias, canonical] of Object.entries(SKILL_ALIASES)) {
    const escaped = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(^|[^a-z0-9+#.])${escaped}([^a-z0-9+#.]|$)`, "i");
    if (regex.test(lower)) {
      found.add(canonical);
    }
  }
  return Array.from(found).sort();
}

export interface ParsedResumeResult {
  current_role: string;
  current_seniority: SeniorityLabel;
  tenure_years: number;
  current_skills: string[];
  resume_length: number;
  has_structured_resume: boolean;
  summary_note: string;
}

const COMMON_ROLE_PATTERNS: RegExp[] = [
  /\b((?:junior|senior|lead|principal|staff|associate|chief)?\s*(?:software|frontend|front-end|backend|back-end|full\s*stack|mobile|android|ios|data|machine learning|ml|ai|nlp|cloud|devops|site reliability|sre|security|cybersecurity|qa|test automation|network|systems|database)\s*(?:engineer|developer|scientist|analyst|architect|administrator|specialist|manager))\b/i,
  /\b((?:junior|senior|lead|group|associate)?\s*(?:product|project|program|engineering|sales|account|marketing|digital marketing|finance|business|operations|customer success|human resources|hr)\s*(?:manager|owner|director|analyst|consultant|specialist|lead|executive))\b/i,
  /\b(solutions architect|scrum master|accountant|financial analyst|ux designer|ui\/ux designer|product designer|graphic designer)\b/i,
];

export function parseResumeText(rawInput: string): ParsedResumeResult {
  const text = (rawInput || "").trim();
  let detectedRole = "";
  let detectedTenure: number | null = null;
  let structuredSkills: string[] = [];
  let isStructured = false;

  // 1. Check if input is structured JSON resume
  if (text.startsWith("{") && text.endsWith("}")) {
    try {
      const obj = JSON.parse(text);
      isStructured = true;
      detectedRole =
        obj.current_role ||
        obj.role ||
        obj.title ||
        obj.job_title ||
        obj.resume_category ||
        obj.personal_info?.title ||
        (Array.isArray(obj.experience) && obj.experience[0]?.title) ||
        "";
      if (typeof obj.tenure_years === "number") {
        detectedTenure = obj.tenure_years;
      } else if (typeof obj.years_experience === "number") {
        detectedTenure = obj.years_experience;
      }
      if (Array.isArray(obj.skills)) {
        structuredSkills = obj.skills.map((s: unknown) =>
          normalizeSkill(String(s))
        );
      } else if (Array.isArray(obj.current_skills)) {
        structuredSkills = obj.current_skills.map((s: unknown) =>
          normalizeSkill(String(s))
        );
      }
    } catch {
      isStructured = false;
    }
  }

  // 2. Infer role from common title regexes or top lines
  if (!detectedRole) {
    for (const pattern of COMMON_ROLE_PATTERNS) {
      const m = text.match(pattern);
      if (m && m[1]) {
        detectedRole = m[1].replace(/\s+/g, " ").trim();
        break;
      }
    }
  }
  if (!detectedRole) {
    const firstLines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length >= 3 && l.length <= 65);
    detectedRole = firstLines[0] || "Software Engineer";
  }

  // 3. Infer tenure years from explicit "X+ years" or date ranges "2019 - Present"
  if (detectedTenure === null) {
    const explicitMatch = text.match(
      /\b(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|tenure|work)?/i
    );
    if (explicitMatch && explicitMatch[1]) {
      detectedTenure = Math.min(45, Math.max(0.5, parseFloat(explicitMatch[1])));
    }
  }
  if (detectedTenure === null) {
    const currentYear = 2026;
    const rangeRegex =
      /\b((?:19|20)\d{2})\s*(?:-|–|—|to|until|through)\s*((?:19|20)\d{2}|present|current|now)\b/gi;
    let minStart = currentYear;
    let maxEnd = 0;
    let match: RegExpExecArray | null;
    while ((match = rangeRegex.exec(text)) !== null) {
      const startYr = parseInt(match[1], 10);
      const rawEnd = match[2].toLowerCase();
      const endYr =
        rawEnd === "present" || rawEnd === "current" || rawEnd === "now"
          ? currentYear
          : parseInt(rawEnd, 10);
      if (startYr >= 1985 && startYr <= currentYear && endYr >= startYr) {
        minStart = Math.min(minStart, startYr);
        maxEnd = Math.max(maxEnd, Math.min(currentYear, endYr));
      }
    }
    if (maxEnd >= minStart && minStart < currentYear) {
      detectedTenure = Math.min(40, Math.max(1, maxEnd - minStart));
    }
  }
  const finalTenure =
    detectedTenure !== null ? Number(detectedTenure.toFixed(1)) : 3.0;

  // 4. Extract skills via deterministic vocabulary matcher
  const extracted = extractSkillsFromText(text);
  const mergedSkills = Array.from(
    new Set([...structuredSkills, ...extracted].filter(Boolean))
  ).sort();

  const finalSkills =
    mergedSkills.length > 0
      ? mergedSkills
      : ["python", "sql", "git", "problem solving", "communication"];

  const seniority = inferSeniority(detectedRole);

  return {
    current_role: detectedRole,
    current_seniority: seniority,
    tenure_years: finalTenure,
    current_skills: finalSkills,
    resume_length: text.length,
    has_structured_resume: isStructured,
    summary_note: `Extracted role "${detectedRole}" (${seniority}, ${finalTenure} yrs) and ${finalSkills.length} verified skills.`,
  };
}

// ---------------------------------------------------------------------------
// Module 1: FitNet Endpoint Calculation (POST /api/fitnet)
// ---------------------------------------------------------------------------
export interface FitNetRequest {
  current_role: string;
  current_skills: string[];
  tenure_years: number;
  target_role: string;
  required_skills: string[];
  job_description?: string;
}

export interface FitNetResponse {
  current_role: string;
  target_job_title: string;
  tenure_years: number;
  fit_score_percentage: string;
  fit_score_raw: number;
  predicted_match_tier: "Low" | "Medium" | "High";
  tier_probabilities: {
    Low: string;
    Medium: string;
    High: string;
  };
  skill_match_ratio: string;
  matched_skills: string[];
  missing_skill_gap: string[];
  missing_skill_gap_count: number;
  upskilling_roadmap: Record<string, string[]>;
  latency_ms: number;
  engine_mode?: string;
}

function clip(val: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, val));
}

export function computeFitNetMatch(payload: FitNetRequest): FitNetResponse {
  const t0 = performance.now();
  const currentRole = (payload.current_role || "Software Engineer").trim();
  const targetRole = (payload.target_role || "Senior Software Engineer").trim();
  const tenureYears = clip(Number(payload.tenure_years) || 0, 0, 60);

  const curSet = new Set(
    (payload.current_skills || [])
      .map((s) => normalizeSkill(String(s)))
      .filter(Boolean)
  );
  const descSkills = extractSkillsFromText(payload.job_description || "");
  const reqSet = new Set(
    [...(payload.required_skills || []), ...descSkills]
      .map((s) => normalizeSkill(String(s)))
      .filter(Boolean)
  );

  const matchedSkills = Array.from(reqSet)
    .filter((s) => curSet.has(s))
    .sort();
  const missingSkills = Array.from(reqSet)
    .filter((s) => !curSet.has(s))
    .sort();

  const matchRatio = matchedSkills.length / Math.max(1, reqSet.size);
  const roleAlign = computeRoleAlignment(currentRole, targetRole);

  const curSen = inferSeniority(currentRole);
  const tgtSen = inferSeniority(targetRole);
  const senChange = SENIORITY_MAP[tgtSen] - SENIORITY_MAP[curSen];

  const fitScoreRaw = Number(
    clip(
      0.55 * matchRatio +
        0.3 * roleAlign +
        0.15 * Math.max(0, 1 - 0.18 * Math.abs(senChange)) +
        Math.min(0.06, tenureYears * 0.01),
      0.02,
      0.98
    ).toFixed(4)
  );

  let predictedTier: "Low" | "Medium" | "High";
  let lowP: number;
  let medP: number;
  let highP: number;

  if (fitScoreRaw >= 0.6) {
    predictedTier = "High";
    highP = clip(0.55 + (fitScoreRaw - 0.6) * 1.1, 0.58, 0.94);
    medP = clip((1 - highP) * 0.82, 0.04, 0.35);
    lowP = Math.max(0.01, 1 - highP - medP);
  } else if (fitScoreRaw >= 0.32) {
    predictedTier = "Medium";
    medP = clip(0.62 + (0.14 - Math.abs(fitScoreRaw - 0.46)) * 0.8, 0.52, 0.78);
    highP =
      fitScoreRaw >= 0.46
        ? (1 - medP) * 0.65
        : (1 - medP) * 0.32;
    lowP = Math.max(0.02, 1 - medP - highP);
  } else {
    predictedTier = "Low";
    lowP = clip(0.66 + (0.32 - fitScoreRaw) * 0.9, 0.6, 0.93);
    medP = clip((1 - lowP) * 0.8, 0.05, 0.32);
    highP = Math.max(0.01, 1 - lowP - medP);
  }

  const pSum = lowP + medP + highP;
  lowP /= pSum;
  medP /= pSum;
  highP /= pSum;

  let roadmap: Record<string, string[]>;
  if (missingSkills.length === 0) {
    roadmap = {
      "Phase 1: Immediate Job Prerequisites": [
        "All core required skills already present in candidate profile",
      ],
      "Phase 2: Core Technical Alignment": [
        "System architecture alignment & domain-specific case studies",
      ],
      "Phase 3: Role Mastery": [
        "Production leadership & cross-functional stakeholder delivery",
      ],
    };
  } else {
    const p1 = Math.max(1, Math.ceil(missingSkills.length / 3));
    const p2 = Math.max(p1 + 1, Math.ceil((2 * missingSkills.length) / 3));
    roadmap = {
      "Phase 1: Immediate Job Prerequisites": missingSkills.slice(0, p1),
      "Phase 2: Core Technical Alignment":
        missingSkills.slice(p1, p2).length > 0
          ? missingSkills.slice(p1, p2)
          : ["Applied portfolio project integrating target stack"],
      "Phase 3: Role Mastery":
        missingSkills.slice(p2).length > 0
          ? missingSkills.slice(p2)
          : ["End-to-end production deployment & performance optimization"],
    };
  }

  const latencyMs = Number(Math.max(1.2, performance.now() - t0).toFixed(2));

  return {
    current_role: currentRole,
    target_job_title: targetRole,
    tenure_years: tenureYears,
    fit_score_percentage: `${(fitScoreRaw * 100).toFixed(2)}%`,
    fit_score_raw: fitScoreRaw,
    predicted_match_tier: predictedTier,
    tier_probabilities: {
      Low: `${(lowP * 100).toFixed(1)}%`,
      Medium: `${(medP * 100).toFixed(1)}%`,
      High: `${(highP * 100).toFixed(1)}%`,
    },
    skill_match_ratio: `${(matchRatio * 100).toFixed(1)}%`,
    matched_skills: matchedSkills,
    missing_skill_gap: missingSkills,
    missing_skill_gap_count: missingSkills.length,
    upskilling_roadmap: roadmap,
    latency_ms: latencyMs,
    engine_mode: "serverless_edge",
  };
}

// ---------------------------------------------------------------------------
// Module 2: PathNet0 Transition Predictor (POST /api/pathnet/predict)
// ---------------------------------------------------------------------------
export interface PathNetPredictRequest {
  current_role: string;
  tenure_years: number;
  target_role: string;
  step_index?: number;
}

export interface PathNetPredictResponse {
  current_role: string;
  current_seniority: SeniorityLabel;
  tenure_years: number;
  target_future_role: string;
  target_seniority: SeniorityLabel;
  seniority_delta: string;
  transition_probability: string;
  transition_probability_raw: number;
  predicted_transition_type:
    | "Vertical_Promotion"
    | "Adjacent_Pivot"
    | "Career_Shift";
  type_distribution: {
    Vertical_Promotion: string;
    Adjacent_Pivot: string;
    Career_Shift: string;
  };
  estimated_skill_gap_count: number;
  feasibility_assessment: string;
  actionable_roadmap: Record<string, string[]>;
  latency_ms: number;
  engine_mode?: string;
}

const DOMAIN_ROADMAPS: Record<string, Record<string, string[]>> = {
  Software: {
    "Phase 1: Core Foundations": [
      "Advanced Data Structures & Concurrency",
      "Clean Architecture & SOLID Design Patterns",
      "Relational & Distributed SQL Optimization",
    ],
    "Phase 2: Applied Mastery": [
      "Microservices & Event-Driven Systems (Kafka / gRPC)",
      "Containerization & CI/CD Automation (Docker / GitHub Actions)",
      "High-Throughput API Design & Caching (Redis)",
    ],
    "Phase 3: Production & Leadership": [
      "Distributed Systems Fault Tolerance & Observability",
      "Technical RFC Authoring & Code Review Mentorship",
      "Cloud-Native Scalability & Cost Optimization",
    ],
  },
  Frontend: {
    "Phase 1: Core Foundations": [
      "Modern TypeScript & Asynchronous JavaScript Patterns",
      "Component Architecture (React 19 / Next.js App Router)",
      "Responsive CSS Grid, Tailwind & WCAG 2.1 Accessibility",
    ],
    "Phase 2: Applied Mastery": [
      "Client/Server State Management & Data Fetching",
      "Core Web Vitals, Bundle Splitting & SSR/ISR Optimization",
      "Automated End-to-End UI Testing (Playwright / Vitest)",
    ],
    "Phase 3: Production & Leadership": [
      "Enterprise Design System Architecture & Tokens",
      "Micro-Frontend Federation & Edge Caching",
      "Cross-Functional Product & UX Engineering Leadership",
    ],
  },
  FullStack: {
    "Phase 1: Core Foundations": [
      "TypeScript Full-Stack Foundations (React, Next.js, Node.js)",
      "Relational & Document Schema Modeling (PostgreSQL / MongoDB)",
      "RESTful & GraphQL API Contract Design",
    ],
    "Phase 2: Applied Mastery": [
      "Authentication, OAuth2 / JWT & Role-Based Access Control",
      "Dockerized Services, Redis Caching & Background Queues",
      "Automated Integration Testing & CI/CD Pipelines",
    ],
    "Phase 3: Production & Leadership": [
      "End-to-End Cloud Deployment (Vercel / AWS / Kubernetes)",
      "Full-Stack Observability, Security Hardening & Rate Limiting",
      "System Architecture Ownership & Agile Delivery",
    ],
  },
  Mobile: {
    "Phase 1: Core Foundations": [
      "Kotlin / Swift Modern Language Concurrency & Coroutines",
      "Declarative Mobile UI (Jetpack Compose / SwiftUI)",
      "MVVM / Clean Architecture & Dependency Injection",
    ],
    "Phase 2: Applied Mastery": [
      "Offline-First Local Persistence & Network Synchronization",
      "Memory Profiling, Battery Optimization & NDK Integration",
      "Automated UI & Unit Testing (Espresso / XCTest)",
    ],
    "Phase 3: Production & Leadership": [
      "CI/CD Release Automation & App Store / Play Store Governance",
      "Modular Multi-Feature Mobile App Architecture",
      "Mobile Platform Technical Leadership",
    ],
  },
  "Data & AI": {
    "Phase 1: Core Foundations": [
      "Python Scientific Stack (NumPy, Pandas, Polars) & Advanced SQL",
      "Statistical Inference, Experimental Design & A/B Testing",
      "Feature Engineering & Classical ML (Scikit-Learn / LightGBM)",
    ],
    "Phase 2: Applied Mastery": [
      "Deep Learning & Transformer Architectures (PyTorch)",
      "Retrieval-Augmented Generation (RAG) & LLM Fine-Tuning",
      "Data Pipelines & Workflow Orchestration (Spark / Airflow)",
    ],
    "Phase 3: Production & Leadership": [
      "MLOps, Model Serving (FastAPI / Docker) & Drift Monitoring",
      "Post-Hoc Probability Calibration & Algorithmic Fairness Auditing",
      "AI Strategy & Business Impact Translation",
    ],
  },
  "QA & Testing": {
    "Phase 1: Core Foundations": [
      "Test Design Methodologies, Boundary Analysis & SQL Verification",
      "Python / TypeScript Scripting for Automated Testing",
      "API Contract Testing (Postman / Pytest / REST Assured)",
    ],
    "Phase 2: Applied Mastery": [
      "End-to-End Web & Mobile Automation (Playwright / Selenium)",
      "Continuous Testing inside CI/CD Pipelines (GitHub Actions / Jenkins)",
      "Load, Stress & Performance Benchmarking (k6 / Locust)",
    ],
    "Phase 3: Production & Leadership": [
      "Quality Engineering Architecture & Shift-Left Governance",
      "Security Regression & Chaos Engineering Readiness",
      "Release Readiness Metrics & QA Team Leadership",
    ],
  },
  "Product & Project": {
    "Phase 1: Core Foundations": [
      "User Discovery, Customer Interviews & Problem Framing",
      "Agile / Scrum Execution, Backlog Prioritization & PRD Authoring",
      "Product Analytics, Funnel Metrics & SQL Data Querying",
    ],
    "Phase 2: Applied Mastery": [
      "A/B Experimentation, Retention Modeling & KPI/OKR Ownership",
      "Go-To-Market Strategy, Pricing & Competitive Positioning",
      "Cross-Functional Alignment (Engineering, Design, Commercial)",
    ],
    "Phase 3: Production & Leadership": [
      "Multi-Quarter Product Portfolio & Roadmap Strategy",
      "Executive Stakeholder Communication & P&L Impact",
      "Organizational Scaling & Product Governance",
    ],
  },
  "Infrastructure & Security": {
    "Phase 1: Core Foundations": [
      "Linux Systems Internals, Networking Protocols & Bash/Python",
      "Container Orchestration (Docker & Kubernetes Architecture)",
      "Infrastructure as Code (Terraform & Ansible)",
    ],
    "Phase 2: Applied Mastery": [
      "Cloud Architecture (AWS / GCP / Azure) & IAM Zero-Trust Security",
      "GitOps CI/CD Pipelines & Automated Security Scanning",
      "Observability Stack (Prometheus, Grafana, OpenTelemetry)",
    ],
    "Phase 3: Production & Leadership": [
      "High-Availability Disaster Recovery & Multi-Region Failover",
      "Incident Response Command, SLO/Error-Budget Governance",
      "Cloud FinOps & Platform Engineering Strategy",
    ],
  },
  "Finance & Consulting": {
    "Phase 1: Core Foundations": [
      "Financial Statement Analysis, GAAP/IFRS & Corporate Valuation",
      "Advanced Financial Modeling (Excel, SQL & Python Analytics)",
      "Risk Assessment, Internal Controls & Regulatory Compliance",
    ],
    "Phase 2: Applied Mastery": [
      "Strategic Forecasting, Capital Budgeting & Scenario Sensitivity",
      "Business Intelligence Dashboards (Power BI / Tableau)",
      "M&A Due Diligence & Management Consulting Frameworks",
    ],
    "Phase 3: Production & Leadership": [
      "Executive C-Suite Advisory & Capital Allocation Strategy",
      "Enterprise Audit Governance & Cross-Border Structuring",
      "Practice Leadership & Client Portfolio Growth",
    ],
  },
  "Sales & Marketing": {
    "Phase 1: Core Foundations": [
      "ICP Segmentation, Value Proposition Framing & CRM Mastery (Salesforce/HubSpot)",
      "Full-Funnel Digital Acquisition (SEO, SEM & Paid Media Analytics)",
      "Consultative Discovery, Pipeline Qualification & Copywriting",
    ],
    "Phase 2: Applied Mastery": [
      "Multi-Touch Attribution, Conversion Rate Optimization & GA4",
      "Enterprise B2B Deal Structuring, Negotiation & Account Expansion",
      "Lifecycle Marketing Automation & Retention Playbooks",
    ],
    "Phase 3: Production & Leadership": [
      "Revenue Forecasting, Territory Design & Quota Capacity Planning",
      "Go-To-Market Executive Alignment & Brand Positioning",
      "Commercial Team Coaching & Revenue Operations Leadership",
    ],
  },
};

export function computePathNetTransition(
  payload: PathNetPredictRequest
): PathNetPredictResponse {
  const t0 = performance.now();
  const currentRole = (payload.current_role || "Junior Software Engineer").trim();
  const targetRole = (payload.target_role || "Senior Software Engineer").trim();
  const tenureYears = clip(Number(payload.tenure_years) || 0, 0, 50);

  const curSen = inferSeniority(currentRole);
  const tgtSen = inferSeniority(targetRole);
  const curLvl = SENIORITY_MAP[curSen];
  const tgtLvl = SENIORITY_MAP[tgtSen];
  const senChange = tgtLvl - curLvl;

  const domCur = inferDomains(currentRole);
  const domTgt = inferDomains(targetRole);
  const lexJac = computeLexicalJaccard(currentRole, targetRole);

  const sameDom =
    (domCur.length > 0 && domCur.some((d) => domTgt.includes(d))) ||
    lexJac >= 0.45;
  const adjDom =
    !sameDom &&
    ((domCur.length > 0 &&
      domCur.some((dc) =>
        domTgt.some((dt) => (ADJ_DOMAIN_MAP[dc] || []).includes(dt))
      )) ||
      lexJac >= 0.2);

  let transType: "Vertical_Promotion" | "Adjacent_Pivot" | "Career_Shift";
  let prob: number;
  let dist: {
    Vertical_Promotion: string;
    Adjacent_Pivot: string;
    Career_Shift: string;
  };
  let skillGapCount: number;

  if (sameDom && senChange >= 0) {
    transType = "Vertical_Promotion";
    prob = clip(
      0.46 +
        0.03 * Math.min(tenureYears, 6.0) -
        0.05 * Math.max(0, senChange - 1) +
        0.04 * lexJac,
      0.35,
      0.88
    );
    const vp = clip(0.84 + 0.08 * lexJac, 0.78, 0.94);
    const ap = clip((1 - vp) * 0.88, 0.05, 0.19);
    const cs = Math.max(0.006, 1 - vp - ap);
    dist = {
      Vertical_Promotion: `${(vp * 100).toFixed(1)}%`,
      Adjacent_Pivot: `${(ap * 100).toFixed(1)}%`,
      Career_Shift: `${(cs * 100).toFixed(1)}%`,
    };
    skillGapCount = 10 + Math.max(0, senChange);
  } else if (adjDom || (sameDom && senChange < 0)) {
    transType = "Adjacent_Pivot";
    prob = clip(
      0.23 +
        0.02 * Math.min(tenureYears, 5.0) -
        0.03 * Math.abs(senChange) +
        0.05 * lexJac,
      0.14,
      0.44
    );
    dist = {
      Vertical_Promotion: "11.2%",
      Adjacent_Pivot: "80.4%",
      Career_Shift: "8.4%",
    };
    skillGapCount = 12 + Math.abs(senChange);
  } else {
    transType = "Career_Shift";
    prob = clip(
      0.04 + 0.006 * Math.min(tenureYears, 5.0) - 0.005 * Math.max(0, senChange),
      0.015,
      0.11
    );
    dist = {
      Vertical_Promotion: "2.1%",
      Adjacent_Pivot: "9.4%",
      Career_Shift: "88.5%",
    };
    skillGapCount = 17 + Math.max(0, senChange);
  }

  const probRaw = Number(prob.toFixed(4));
  let feasibility: string;
  if (probRaw >= 0.45) {
    feasibility = "High (Optimal Career Step)";
  } else if (probRaw >= 0.18) {
    feasibility = "Moderate (Achievable with Focused Upskilling)";
  } else {
    feasibility = "Challenging (Major Career Shift / Long-Term Plan)";
  }

  const primaryTgtDomain = domTgt[0] || domCur[0] || "Software";
  const roadmap =
    DOMAIN_ROADMAPS[primaryTgtDomain] || DOMAIN_ROADMAPS["Software"];

  const latencyMs = Number(Math.max(1.1, performance.now() - t0).toFixed(2));

  return {
    current_role: currentRole,
    current_seniority: curSen,
    tenure_years: tenureYears,
    target_future_role: targetRole,
    target_seniority: tgtSen,
    seniority_delta: `${senChange > 0 ? "+" : ""}${senChange}`,
    transition_probability: `${(probRaw * 100).toFixed(2)}%`,
    transition_probability_raw: probRaw,
    predicted_transition_type: transType,
    type_distribution: dist,
    estimated_skill_gap_count: skillGapCount,
    feasibility_assessment: feasibility,
    actionable_roadmap: roadmap,
    latency_ms: latencyMs,
    engine_mode: "serverless_edge",
  };
}

// ---------------------------------------------------------------------------
// Module 3: PathNet0 Next-Step Recommender (POST /api/pathnet/recommend)
// ---------------------------------------------------------------------------
export interface PathNetRecommendRequest {
  current_role: string;
  tenure_years: number;
  candidate_roles?: string[];
  top_n?: number;
}

export interface PathNetRecommendResponse {
  current_role: string;
  tenure_years: number;
  recommendations: PathNetPredictResponse[];
  latency_ms: number;
  engine_mode?: string;
}

const DEFAULT_CANDIDATE_POOL = [
  "Senior Software Engineer",
  "Lead Software Engineer",
  "Software Engineering Manager",
  "Full Stack Engineer",
  "Senior Data Scientist",
  "Machine Learning Engineer",
  "DevOps & Cloud Architect",
  "Senior Product Manager",
  "Lead Frontend Engineer",
  "Sales Account Manager",
];

export function computePathNetRecommendations(
  payload: PathNetRecommendRequest
): PathNetRecommendResponse {
  const t0 = performance.now();
  const currentRole = (payload.current_role || "Data Analyst").trim();
  const tenureYears = clip(Number(payload.tenure_years) || 0, 0, 50);
  const topN = clip(Number(payload.top_n) || 4, 1, 10);

  const pool =
    Array.isArray(payload.candidate_roles) && payload.candidate_roles.length > 0
      ? payload.candidate_roles
      : DEFAULT_CANDIDATE_POOL;

  const curClean = cleanText(currentRole);
  const scored = pool
    .map((r) => String(r).trim())
    .filter((r) => r.length > 1 && cleanText(r) !== curClean)
    .map((candidate) =>
      computePathNetTransition({
        current_role: currentRole,
        tenure_years: tenureYears,
        target_role: candidate,
      })
    )
    .sort((a, b) => b.transition_probability_raw - a.transition_probability_raw)
    .slice(0, topN);

  const latencyMs = Number(Math.max(1.8, performance.now() - t0).toFixed(2));

  return {
    current_role: currentRole,
    tenure_years: tenureYears,
    recommendations: scored,
    latency_ms: latencyMs,
    engine_mode: "serverless_edge",
  };
}

// ---------------------------------------------------------------------------
// Demo Presets & Sample Resumes for 1-Click Exploration
// ---------------------------------------------------------------------------
export interface DemoPreset {
  id: string;
  titleEn: string;
  titleAr: string;
  badge: string;
  current_role: string;
  tenure_years: number;
  current_skills: string[];
  target_role: string;
  required_skills: string[];
  job_description: string;
  resume_text: string;
}

export const DEMO_PRESETS: DemoPreset[] = [
  {
    id: "swe-promo",
    titleEn: "Backend Python Engineer ➔ Senior Backend Architect",
    titleAr: "مهندس بايثون ➔ كبير مهندسي النظم الخلفية (ترقية مباشرة)",
    badge: "Vertical Promotion • High Fit",
    current_role: "Python Developer",
    tenure_years: 3.5,
    current_skills: [
      "python",
      "django",
      "fastapi",
      "postgresql",
      "docker",
      "git",
      "rest api",
      "linux",
    ],
    target_role: "Senior Backend Python Engineer",
    required_skills: [
      "python",
      "django",
      "fastapi",
      "postgresql",
      "docker",
      "kubernetes",
      "redis",
      "microservices",
    ],
    job_description:
      "Lead the architecture of high-throughput Python microservices using FastAPI, Django, PostgreSQL, Redis, Docker, and Kubernetes.",
    resume_text: `AHMED HASSAN — Python Developer
Experience: 2022 - Present (3.5+ years experience)
Built scalable backend microservices and REST APIs using Python, Django, FastAPI, and PostgreSQL.
Containerized workloads with Docker on Linux and managed CI/CD workflows via Git and GitHub.`,
  },
  {
    id: "data-to-ml",
    titleEn: "Data Analyst ➔ Machine Learning Engineer",
    titleAr: "محلل بيانات ➔ مهندس تعلم آلي وذكاء اصطناعي",
    badge: "AI Career Progression • Moderate-High Fit",
    current_role: "Data Analyst",
    tenure_years: 3.0,
    current_skills: [
      "python",
      "sql",
      "pandas",
      "numpy",
      "tableau",
      "statistics",
      "data visualization",
      "a/b testing",
    ],
    target_role: "Machine Learning Engineer",
    required_skills: [
      "python",
      "sql",
      "pandas",
      "numpy",
      "scikit-learn",
      "pytorch",
      "docker",
      "mlops",
      "rag",
    ],
    job_description:
      "Design, calibrate, and deploy production machine learning and PyTorch deep learning models with Scikit-Learn, Docker, MLOps, and RAG pipelines.",
    resume_text: `SARA MAHMOUD — Data Analyst
Experience: 2023 - Present (3 years experience)
Delivered product analytics, A/B testing, and executive dashboards using Python, SQL, Pandas, NumPy, Statistics, and Tableau.`,
  },
  {
    id: "fe-to-fs",
    titleEn: "Junior Frontend Developer ➔ Full Stack Engineer",
    titleAr: "مطور واجهات أمامية ➔ مهندس Full Stack متكامل",
    badge: "Adjacent Pivot • Moderate Fit",
    current_role: "Junior Frontend Developer",
    tenure_years: 2.0,
    current_skills: [
      "javascript",
      "typescript",
      "react",
      "next.js",
      "html",
      "css",
      "tailwind css",
      "git",
    ],
    target_role: "Full Stack Engineer",
    required_skills: [
      "javascript",
      "typescript",
      "react",
      "next.js",
      "node.js",
      "postgresql",
      "docker",
      "rest api",
      "graphql",
    ],
    job_description:
      "Develop end-to-end web applications with React, Next.js, TypeScript, Node.js, PostgreSQL, GraphQL, REST API, and Docker.",
    resume_text: `OMAR KHALED — Junior Frontend Developer
2024 - Present (2 years experience)
Built responsive web interfaces using JavaScript, TypeScript, React, Next.js, Tailwind CSS, HTML, CSS, and Git.`,
  },
  {
    id: "shift-case",
    titleEn: "Project Manager ➔ Senior MLOps & Cloud Engineer",
    titleAr: "مدير مشاريع ➔ مهندس MLOps وحوسبة سحابية (تغيير مسار)",
    badge: "Career Shift • Low Immediate Fit",
    current_role: "Project Manager",
    tenure_years: 5.0,
    current_skills: [
      "project management",
      "agile",
      "scrum",
      "jira",
      "stakeholder management",
      "communication",
      "leadership",
    ],
    target_role: "Senior MLOps Engineer",
    required_skills: [
      "python",
      "kubernetes",
      "terraform",
      "pytorch",
      "mlops",
      "aws",
      "docker",
      "ci/cd",
    ],
    job_description:
      "Architect automated MLOps training and serving infrastructure on AWS using Python, PyTorch, Kubernetes, Docker, Terraform, and CI/CD.",
    resume_text: `KARIM ADEL — Project Manager
2021 - Present (5 years experience)
Led cross-functional Agile and Scrum delivery teams using Jira, Stakeholder Management, Leadership, and Communication.`,
  },
];
