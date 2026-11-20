"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles,
  Briefcase,
  Compass,
  TrendingUp,
  FileText,
  Upload,
  Plus,
  X,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Clock,
  Layers,
  Globe,
  ChevronRight,
} from "lucide-react";
import {
  DEMO_PRESETS,
  DemoPreset,
  FitNetResponse,
  PathNetPredictResponse,
  PathNetRecommendResponse,
  SKILL_VOCABULARY,
} from "@/lib/career-engine";
import DefenseReportTab from "@/components/DefenseReportTab";

type TabId = "overview" | "match" | "trajectory" | "recommend" | "report";

function MinimalCircularGauge({
  percentage,
  tone = "emerald",
  label,
  sublabel,
}: {
  percentage: number;
  tone?: "emerald" | "slate" | "amber";
  label: string;
  sublabel?: string;
}) {
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(100, Math.max(0, percentage));
  const offset = circumference - (clamped / 100) * circumference;

  const strokeColor =
    tone === "emerald"
      ? "#34d399"
      : tone === "amber"
      ? "#fbbf24"
      : "#e4e4e7";

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative h-36 w-36">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 132 132">
          <circle
            cx="66"
            cy="66"
            r={radius}
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="8"
            fill="transparent"
          />
          <motion.circle
            cx="66"
            cy="66"
            r={radius}
            stroke={strokeColor}
            strokeWidth="8"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            strokeLinecap="round"
            fill="transparent"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="font-mono text-2xl font-semibold tracking-tight text-zinc-100">
            {clamped.toFixed(1)}%
          </span>
          <span className="mt-0.5 text-[10px] font-medium uppercase tracking-widest text-zinc-500">
            {label}
          </span>
        </div>
      </div>
      {sublabel && (
        <span className="mt-2 text-center text-xs text-zinc-400">
          {sublabel}
        </span>
      )}
    </div>
  );
}

export default function CareerStudioPage() {
  const [lang, setLang] = useState<"en" | "ar">("en");
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [engineMode, setEngineMode] = useState<string>("serverless_edge");
  const isAr = lang === "ar";

  // Resume & Profile State
  const [resumeRawText, setResumeRawText] = useState<string>(
    DEMO_PRESETS[0].resume_text
  );
  const [parseStatusNote, setParseStatusNote] = useState<string>("");
  const [isParsing, setIsParsing] = useState<boolean>(false);

  const [currentRole, setCurrentRole] = useState<string>(
    DEMO_PRESETS[0].current_role
  );
  const [tenureYears, setTenureYears] = useState<number>(
    DEMO_PRESETS[0].tenure_years
  );
  const [currentSkills, setCurrentSkills] = useState<string[]>(
    DEMO_PRESETS[0].current_skills
  );
  const [newCurrentSkillInput, setNewCurrentSkillInput] = useState<string>("");

  const [targetRole, setTargetRole] = useState<string>(
    DEMO_PRESETS[0].target_role
  );
  const [requiredSkills, setRequiredSkills] = useState<string[]>(
    DEMO_PRESETS[0].required_skills
  );
  const [newReqSkillInput, setNewReqSkillInput] = useState<string>("");
  const [jobDescription, setJobDescription] = useState<string>(
    DEMO_PRESETS[0].job_description
  );
  const [stepIndex, setStepIndex] = useState<number>(2);

  // Next-Step Explorer State
  const [candidateRoles, setCandidateRoles] = useState<string[]>([
    "Senior Backend Python Engineer",
    "Lead Software Engineer",
    "Software Engineering Manager",
    "Full Stack Engineer",
    "Machine Learning Engineer",
    "DevOps & Cloud Architect",
    "Senior Data Scientist",
    "Product Manager",
  ]);
  const [newCandidateRoleInput, setNewCandidateRoleInput] =
    useState<string>("");
  const [topN, setTopN] = useState<number>(4);
  const [expandedRecIdx, setExpandedRecIdx] = useState<number | null>(0);

  // Evaluation Results
  const [matchResult, setMatchResult] = useState<FitNetResponse | null>(null);
  const [matchLoading, setMatchLoading] = useState<boolean>(false);

  const [trajectoryResult, setTrajectoryResult] =
    useState<PathNetPredictResponse | null>(null);
  const [trajectoryLoading, setTrajectoryLoading] = useState<boolean>(false);
  const [ragMatches, setRagMatches] = useState<
    Array<{
      target_future_role: string;
      target_job_description: string;
      skill_gap_count: number;
      similarity: number;
      skill_gap_roadmap: Record<string, string[]>;
    }>
  >([]);

  const [recResult, setRecResult] = useState<PathNetRecommendResponse | null>(
    null
  );
  const [recLoading, setRecLoading] = useState<boolean>(false);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        if (d.engine_mode) setEngineMode(d.engine_mode);
      })
      .catch(() => setEngineMode("serverless_edge"));

    runJobMatchEvaluation(DEMO_PRESETS[0]);
    runTrajectoryEvaluation(
      DEMO_PRESETS[0].current_role,
      DEMO_PRESETS[0].tenure_years,
      DEMO_PRESETS[0].target_role,
      2
    );
    runRecommendationEvaluation(
      DEMO_PRESETS[0].current_role,
      DEMO_PRESETS[0].tenure_years,
      candidateRoles,
      4
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runJobMatchEvaluation(override?: Partial<DemoPreset>) {
    setMatchLoading(true);
    try {
      const payload = {
        current_role: override?.current_role ?? currentRole,
        current_skills: override?.current_skills ?? currentSkills,
        tenure_years: override?.tenure_years ?? tenureYears,
        target_role: override?.target_role ?? targetRole,
        required_skills: override?.required_skills ?? requiredSkills,
        job_description: override?.job_description ?? jobDescription,
      };
      const res = await fetch("/api/fitnet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setMatchResult(data);
      if (data.engine_mode) setEngineMode(data.engine_mode);
    } finally {
      setMatchLoading(false);
    }
  }

  async function runTrajectoryEvaluation(
    cRole = currentRole,
    ten = tenureYears,
    tRole = targetRole,
    sIdx = stepIndex
  ) {
    setTrajectoryLoading(true);
    try {
      const [res, ragRes] = await Promise.all([
        fetch("/api/pathnet/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            current_role: cRole,
            tenure_years: ten,
            target_role: tRole,
            step_index: sIdx,
          }),
        }),
        fetch("/api/pathnet/rag", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query_role: tRole,
            top_k: 3,
          }),
        }),
      ]);
      const data = await res.json();
      setTrajectoryResult(data);
      if (data.engine_mode) setEngineMode(data.engine_mode);
      if (ragRes.ok) {
        const ragData = await ragRes.json();
        if (Array.isArray(ragData.matches)) {
          setRagMatches(ragData.matches);
        }
      }
    } finally {
      setTrajectoryLoading(false);
    }
  }

  async function runRecommendationEvaluation(
    cRole = currentRole,
    ten = tenureYears,
    cands = candidateRoles,
    n = topN
  ) {
    setRecLoading(true);
    try {
      const res = await fetch("/api/pathnet/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_role: cRole,
          tenure_years: ten,
          candidate_roles: cands,
          top_n: n,
        }),
      });
      const data = await res.json();
      setRecResult(data);
      if (data.engine_mode) setEngineMode(data.engine_mode);
    } finally {
      setRecLoading(false);
    }
  }

  async function handleParseResumeText(textToParse = resumeRawText) {
    setIsParsing(true);
    try {
      const res = await fetch("/api/parse-resume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToParse }),
      });
      const data = await res.json();
      if (data.current_role) setCurrentRole(data.current_role);
      if (typeof data.tenure_years === "number")
        setTenureYears(data.tenure_years);
      if (Array.isArray(data.current_skills))
        setCurrentSkills(data.current_skills);
      setParseStatusNote(data.summary_note || "Profile extracted.");
      runJobMatchEvaluation({
        current_role: data.current_role || currentRole,
        tenure_years:
          typeof data.tenure_years === "number"
            ? data.tenure_years
            : tenureYears,
        current_skills: Array.isArray(data.current_skills)
          ? data.current_skills
          : currentSkills,
      });
    } finally {
      setIsParsing(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsParsing(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/parse-resume", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.extracted_text_preview) {
        setResumeRawText(data.extracted_text_preview);
      }
      if (data.current_role) setCurrentRole(data.current_role);
      if (typeof data.tenure_years === "number")
        setTenureYears(data.tenure_years);
      if (Array.isArray(data.current_skills))
        setCurrentSkills(data.current_skills);
      setParseStatusNote(
        `Extracted from "${file.name}": ${data.summary_note || ""}`
      );
      runJobMatchEvaluation({
        current_role: data.current_role || currentRole,
        tenure_years:
          typeof data.tenure_years === "number"
            ? data.tenure_years
            : tenureYears,
        current_skills: Array.isArray(data.current_skills)
          ? data.current_skills
          : currentSkills,
      });
    } finally {
      setIsParsing(false);
    }
  }

  function loadPreset(preset: DemoPreset, targetTab: TabId) {
    setCurrentRole(preset.current_role);
    setTenureYears(preset.tenure_years);
    setCurrentSkills(preset.current_skills);
    setTargetRole(preset.target_role);
    setRequiredSkills(preset.required_skills);
    setJobDescription(preset.job_description);
    setResumeRawText(preset.resume_text);
    setParseStatusNote(
      `Loaded example: ${preset.current_role} ➔ ${preset.target_role}`
    );
    runJobMatchEvaluation(preset);
    runTrajectoryEvaluation(
      preset.current_role,
      preset.tenure_years,
      preset.target_role,
      2
    );
    runRecommendationEvaluation(
      preset.current_role,
      preset.tenure_years,
      candidateRoles,
      topN
    );
    setActiveTab(targetTab);
  }

  function addCurrentSkill(skill: string) {
    const clean = skill.toLowerCase().trim();
    if (!clean) return;
    if (!currentSkills.includes(clean)) {
      setCurrentSkills([...currentSkills, clean]);
    }
    setNewCurrentSkillInput("");
  }

  function addRequiredSkill(skill: string) {
    const clean = skill.toLowerCase().trim();
    if (!clean) return;
    if (!requiredSkills.includes(clean)) {
      setRequiredSkills([...requiredSkills, clean]);
    }
    setNewReqSkillInput("");
  }

  const navTabs: {
    id: TabId;
    labelEn: string;
    labelAr: string;
    icon: React.ReactNode;
  }[] = [
    {
      id: "overview",
      labelEn: "Overview",
      labelAr: "الرئيسية",
      icon: <Sparkles className="h-3.5 w-3.5" />,
    },
    {
      id: "match",
      labelEn: "Job Match",
      labelAr: "مطابقة الوظيفة",
      icon: <Briefcase className="h-3.5 w-3.5" />,
    },
    {
      id: "trajectory",
      labelEn: "Career Path",
      labelAr: "المسار الوظيفي",
      icon: <Compass className="h-3.5 w-3.5" />,
    },
    {
      id: "recommend",
      labelEn: "Next Steps",
      labelAr: "الخطوات التالية",
      icon: <TrendingUp className="h-3.5 w-3.5" />,
    },
    {
      id: "report",
      labelEn: "Technical Report",
      labelAr: "التقرير التقني",
      icon: <FileText className="h-3.5 w-3.5" />,
    },
  ];

  return (
    <div
      dir={isAr ? "rtl" : "ltr"}
      className="min-h-screen bg-zinc-950 text-zinc-100"
    >
      {/* Minimal Header */}
      <header className="no-print sticky top-0 z-40 border-b border-zinc-800/80 bg-zinc-950/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 text-zinc-100">
              <Compass className="h-4 w-4" />
            </div>
            <div>
              <span className="text-sm font-semibold tracking-tight text-zinc-100">
                {isAr ? "منصة الذكاء المهني" : "Career Intelligence"}
              </span>
              <span className="mx-2 text-zinc-700">•</span>
              <span className="hidden text-xs text-zinc-400 sm:inline">
                {isAr
                  ? "مطابقة السير الذاتية وتخطيط المسار الوظيفي"
                  : "Resume Matching & Career Trajectory Studio"}
              </span>
            </div>
          </div>

          {/* Navigation Pills */}
          <nav className="flex items-center gap-1 overflow-x-auto rounded-xl border border-zinc-800/80 bg-zinc-900/60 p-1">
            {navTabs.map((t) => {
              const active = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`relative inline-flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                    active
                      ? "text-zinc-950"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  {active && (
                    <motion.div
                      layoutId="activeNavTab"
                      className="absolute inset-0 rounded-lg bg-zinc-100"
                      transition={{
                        type: "spring",
                        stiffness: 420,
                        damping: 34,
                      }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-1.5">
                    {t.icon}
                    <span>{isAr ? t.labelAr : t.labelEn}</span>
                  </span>
                </button>
              );
            })}
          </nav>

          {/* Status & Language Toggle */}
          <div className="flex items-center gap-2.5">
            <div className="hidden items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900/80 px-2.5 py-1 text-[11px] text-zinc-400 md:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              {engineMode === "fastapi_production"
                ? isAr
                  ? "محرك الإنتاج متصل"
                  : "Production API"
                : isAr
                ? "المحرك المعاير جاهز"
                : "Calibrated Engine"}
            </div>

            <button
              onClick={() => setLang(isAr ? "en" : "ar")}
              className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900/80 px-2.5 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-zinc-700 hover:text-white"
            >
              <Globe className="h-3.5 w-3.5 text-zinc-400" />
              {isAr ? "EN" : "عربي"}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <AnimatePresence mode="wait">
          {/* ================================================================
              TAB 1: OVERVIEW
             ================================================================ */}
          {activeTab === "overview" && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="space-y-10"
            >
              {/* Minimal Hero */}
              <div className="rounded-3xl border border-zinc-800/80 bg-gradient-to-b from-zinc-900/70 to-zinc-950 p-8 sm:p-10">
                <div className="max-w-2xl space-y-4">
                  <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900 px-3 py-1 text-xs font-medium text-zinc-300">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                    {isAr
                      ? "نظام ذكاء اصطناعي متكامل للتوجيه المهني"
                      : "AI-Powered Career Guidance & Skill Intelligence"}
                  </div>
                  <h1 className="text-3xl font-semibold tracking-tight text-zinc-100 sm:text-4xl">
                    {isAr
                      ? "قيّم توافق سيرتك الذاتية مع الوظائف، وخطط لخطوتك المهنية القادمة بوضوح."
                      : "Evaluate your resume against any job and map your next career move with clarity."}
                  </h1>
                  <p className="text-sm leading-relaxed text-zinc-400">
                    {isAr
                      ? "صُممت هذه المنصة لتكون سهلة الاستخدام لأي شخص: ارفع سيرتك الذاتية لمعرفة نسبة التوافق والمهارات الناقصة لأي وظيفة، أو استكشف احتمالية الترقية والانتقال الوظيفي مع خارطة طريق تعليمية مفصلة."
                      : "Built to be effortless for anyone: upload your resume to check your fit and missing skills for a specific job posting, or explore your future career path with a step-by-step learning roadmap."}
                  </p>
                  <div className="flex flex-wrap gap-3 pt-2">
                    <button
                      onClick={() => setActiveTab("match")}
                      className="inline-flex items-center gap-2 rounded-xl bg-zinc-100 px-5 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-white"
                    >
                      <Briefcase className="h-3.5 w-3.5" />
                      {isAr ? "ابدأ مطابقة الوظيفة" : "Check Job Match"}
                    </button>
                    <button
                      onClick={() => setActiveTab("trajectory")}
                      className="inline-flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/80 px-5 py-2.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-700 hover:bg-zinc-900"
                    >
                      <Compass className="h-3.5 w-3.5" />
                      {isAr ? "استكشف المسار الوظيفي" : "Explore Career Path"}
                    </button>
                    <button
                      onClick={() => setActiveTab("recommend")}
                      className="inline-flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-2.5 text-xs font-medium text-zinc-300 transition hover:border-zinc-700 hover:text-white"
                    >
                      <TrendingUp className="h-3.5 w-3.5" />
                      {isAr ? "ترشيح الوظائف التالية" : "Discover Next Roles"}
                    </button>
                  </div>
                </div>

                {/* Clean Stats Row */}
                <div className="mt-8 grid grid-cols-2 gap-4 border-t border-zinc-800/80 pt-6 sm:grid-cols-4">
                  <div>
                    <div className="font-mono text-xl font-semibold text-zinc-100">
                      1,140,000
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-500">
                      {isAr
                        ? "مسار وظيفي حقيقي تم تحليلها"
                        : "Career transitions modeled"}
                    </div>
                  </div>
                  <div>
                    <div className="font-mono text-xl font-semibold text-zinc-100">
                      221,903
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-500">
                      {isAr
                        ? "مطابقة سيرة ذاتية ووظيفة"
                        : "Resume-to-job match records"}
                    </div>
                  </div>
                  <div>
                    <div className="font-mono text-xl font-semibold text-zinc-100">
                      230,333
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-500">
                      {isAr
                        ? "خارطة طريق مهارية موثقة"
                        : "Verified skill roadmaps"}
                    </div>
                  </div>
                  <div>
                    <div className="font-mono text-xl font-semibold text-emerald-400">
                      &lt; 20 ms
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-500">
                      {isAr ? "زمن الاستجابة الفوري" : "Instant response time"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Simple 2-Card Guide: Choose Your Goal */}
              <div className="grid gap-5 md:grid-cols-2">
                <motion.div
                  whileHover={{ y: -2 }}
                  transition={{ duration: 0.18 }}
                  onClick={() => setActiveTab("match")}
                  className="cursor-pointer rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6 transition hover:border-zinc-700"
                >
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-200">
                    <Briefcase className="h-4 w-4" />
                  </div>
                  <h3 className="text-base font-semibold text-zinc-100">
                    {isAr
                      ? "1. مطابقة السيرة الذاتية مع وظيفة محددة"
                      : "1. Resume-to-Job Matching"}
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">
                    {isAr
                      ? "ارفع ملف الـ CV أو أدخل مهاراتك الحالية مع الوظيفة التي تريد التقديم عليها؛ ستحصل فوراً على نسبة التوافق، المهارات المتوفرة لديك، والمهارات الناقصة مرتبة في خطة تعلم."
                      : "Upload your resume or enter your current skills alongside a target job posting. Get an instant fit score, matched skills, and exact missing skills organized into a learning plan."}
                  </p>
                  <div className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium text-zinc-200">
                    <span>{isAr ? "ابدأ الآن" : "Open Job Matcher"}</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </div>
                </motion.div>

                <motion.div
                  whileHover={{ y: -2 }}
                  transition={{ duration: 0.18 }}
                  onClick={() => setActiveTab("trajectory")}
                  className="cursor-pointer rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6 transition hover:border-zinc-700"
                >
                  <div className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-200">
                    <Compass className="h-4 w-4" />
                  </div>
                  <h3 className="text-base font-semibold text-zinc-100">
                    {isAr
                      ? "2. تخطيط المسار الوظيفي المستقبلي"
                      : "2. Career Trajectory & Roadmap Planner"}
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">
                    {isAr
                      ? "أدخل مسماك الوظيفي الحالي وسنوات خبرتك والمنصب الذي تطمح إليه؛ سيقوم النظام بحساب احتمالية الانتقال، وتصنيف الخطوة (ترقية، انتقال جانبي، أو تغيير مسار) مع خارطة طريق من 3 مراحل."
                      : "Enter your current title, years of experience, and desired future role. View transition feasibility, move category (promotion, lateral pivot, or career shift), and a 3-phase skill roadmap."}
                  </p>
                  <div className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium text-zinc-200">
                    <span>{isAr ? "ابدأ الآن" : "Open Career Planner"}</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </div>
                </motion.div>
              </div>

              {/* 1-Click Ready Examples */}
              <div className="space-y-4">
                <div>
                  <h2 className="text-base font-semibold text-zinc-100">
                    {isAr
                      ? "أمثلة جاهزة للتجربة الفورية بضغطة واحدة"
                      : "Try an Example Scenario with One Click"}
                  </h2>
                  <p className="text-xs text-zinc-500">
                    {isAr
                      ? "اختر أي مثال لتجربة النظام فوراً بدون إدخال بيانات يدوياً."
                      : "Click any profile below to populate all fields and view instant results."}
                  </p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  {DEMO_PRESETS.map((preset) => (
                    <motion.div
                      key={preset.id}
                      whileHover={{ y: -2 }}
                      transition={{ duration: 0.18 }}
                      className="flex flex-col justify-between rounded-2xl border border-zinc-800/80 bg-zinc-900/30 p-5 transition hover:border-zinc-700"
                    >
                      <div>
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <span className="rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-0.5 text-[11px] font-medium text-zinc-300">
                            {preset.badge}
                          </span>
                          <span className="font-mono text-xs text-zinc-500">
                            {preset.tenure_years}{" "}
                            {isAr ? "سنوات خبرة" : "yrs exp"}
                          </span>
                        </div>
                        <h3 className="text-sm font-semibold text-zinc-100">
                          {isAr ? preset.titleAr : preset.titleEn}
                        </h3>
                        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
                          {preset.job_description}
                        </p>
                      </div>

                      <div className="mt-4 flex gap-2 border-t border-zinc-800/70 pt-3.5">
                        <button
                          onClick={() => loadPreset(preset, "match")}
                          className="flex-1 rounded-xl bg-zinc-100 px-3 py-2 text-xs font-semibold text-zinc-950 transition hover:bg-white"
                        >
                          {isAr ? "فحص توافق الوظيفة" : "Check Job Match"}
                        </button>
                        <button
                          onClick={() => loadPreset(preset, "trajectory")}
                          className="flex-1 rounded-xl border border-zinc-800 bg-zinc-900/80 px-3 py-2 text-xs font-medium text-zinc-200 transition hover:border-zinc-700 hover:text-white"
                        >
                          {isAr ? "عرض المسار الوظيفي" : "View Career Path"}
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* ================================================================
              TAB 2: JOB MATCH (RESUME PARSER & JOB FIT)
             ================================================================ */}
          {activeTab === "match" && (
            <motion.div
              key="match"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="space-y-6"
            >
              {/* Quick Example Selector Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-zinc-800/80 bg-zinc-900/40 px-4 py-3">
                <span className="text-xs text-zinc-400">
                  {isAr
                    ? "جرب مثالاً جاهزاً أو ارفع سيرتك الذاتية:"
                    : "Quick-load an example profile or upload your own CV below:"}
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {DEMO_PRESETS.map((p, idx) => (
                    <button
                      key={p.id}
                      onClick={() => loadPreset(p, "match")}
                      className="rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-xs font-medium text-zinc-300 transition hover:border-zinc-600 hover:text-white"
                    >
                      {isAr ? `مثال ${idx + 1}` : `Example ${idx + 1}`}:{" "}
                      {p.current_role.split(" ")[0]}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid gap-8 lg:grid-cols-12">
                {/* Left Column: Inputs */}
                <div className="space-y-5 lg:col-span-5">
                  {/* Step 1: Resume Upload / Paste */}
                  <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5">
                    <div className="mb-3 flex items-center justify-between">
                      <span className="text-xs font-semibold text-zinc-200">
                        {isAr
                          ? "خطوة 1: استخراج بيانات السيرة الذاتية (اختياري)"
                          : "Step 1: Import Resume (Optional)"}
                      </span>
                      <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800/80 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:bg-zinc-800">
                        <Upload className="h-3.5 w-3.5" />
                        {isAr ? "رفع ملف CV" : "Upload PDF / TXT"}
                        <input
                          type="file"
                          accept=".pdf,.txt,.json,.md"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                      </label>
                    </div>

                    <textarea
                      rows={3}
                      value={resumeRawText}
                      onChange={(e) => setResumeRawText(e.target.value)}
                      placeholder={
                        isAr
                          ? "أو الصق نص السيرة الذاتية هنا لاستخراج المسمى الوظيفي والمهارات تلقائياً..."
                          : "Or paste your resume text here to automatically fill your role, experience, and skills..."
                      }
                      className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 p-3 text-xs text-zinc-200 placeholder-zinc-600 focus:border-zinc-600 focus:outline-none"
                    />

                    <div className="mt-2.5 flex items-center justify-between">
                      <button
                        onClick={() => handleParseResumeText(resumeRawText)}
                        disabled={isParsing}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800/90 px-3 py-1.5 text-xs font-medium text-zinc-100 transition hover:bg-zinc-700"
                      >
                        <Sparkles className="h-3.5 w-3.5 text-zinc-300" />
                        {isParsing
                          ? isAr
                            ? "جاري الاستخراج..."
                            : "Extracting..."
                          : isAr
                          ? "استخراج المهارات تلقائياً"
                          : "Extract Skills from Text"}
                      </button>
                      {parseStatusNote && (
                        <span className="text-[11px] text-emerald-400">
                          ✓ {isAr ? "تم الاستخراج" : "Extracted"}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Step 2: Your Profile */}
                  <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5">
                    <h3 className="mb-3.5 text-xs font-semibold text-zinc-200">
                      {isAr
                        ? "خطوة 2: ملفك الحالي ومهاراتك"
                        : "Step 2: Your Current Profile & Skills"}
                    </h3>
                    <div className="grid gap-3 sm:grid-cols-3">
                      <div className="sm:col-span-2">
                        <label className="mb-1 block text-xs text-zinc-400">
                          {isAr ? "المسمى الوظيفي الحالي" : "Your Current Title"}
                        </label>
                        <input
                          type="text"
                          value={currentRole}
                          onChange={(e) => setCurrentRole(e.target.value)}
                          className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-zinc-400">
                          {isAr ? "سنوات الخبرة" : "Experience (Yrs)"}
                        </label>
                        <input
                          type="number"
                          step="0.5"
                          min="0"
                          max="50"
                          value={tenureYears}
                          onChange={(e) =>
                            setTenureYears(parseFloat(e.target.value) || 0)
                          }
                          className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2 font-mono text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                        />
                      </div>
                    </div>

                    <div className="mt-4">
                      <label className="mb-1.5 block text-xs text-zinc-400">
                        {isAr
                          ? `مهاراتك الحالية (${currentSkills.length})`
                          : `Your Skills (${currentSkills.length})`}
                      </label>
                      <div className="flex flex-wrap gap-1.5 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-2.5">
                        {currentSkills.map((sk) => (
                          <span
                            key={sk}
                            className="inline-flex items-center gap-1 rounded-md border border-zinc-700/80 bg-zinc-800/70 px-2 py-0.5 font-mono text-xs text-zinc-200"
                          >
                            {sk}
                            <button
                              onClick={() =>
                                setCurrentSkills(
                                  currentSkills.filter((s) => s !== sk)
                                )
                              }
                              className="text-zinc-500 hover:text-zinc-200"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </span>
                        ))}
                      </div>
                      <div className="mt-2 flex gap-2">
                        <input
                          type="text"
                          list="skill-vocab-list"
                          value={newCurrentSkillInput}
                          onChange={(e) =>
                            setNewCurrentSkillInput(e.target.value)
                          }
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              addCurrentSkill(newCurrentSkillInput);
                            }
                          }}
                          placeholder={
                            isAr ? "أضف مهارة..." : "Add a skill and press Enter..."
                          }
                          className="flex-1 rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-1.5 text-xs text-zinc-100 focus:border-zinc-600 focus:outline-none"
                        />
                        <button
                          onClick={() => addCurrentSkill(newCurrentSkillInput)}
                          className="inline-flex items-center gap-1 rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:border-zinc-700 hover:text-white"
                        >
                          <Plus className="h-3.5 w-3.5" />
                          {isAr ? "إضافة" : "Add"}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Step 3: Target Job */}
                  <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5">
                    <h3 className="mb-3.5 text-xs font-semibold text-zinc-200">
                      {isAr
                        ? "خطوة 3: الوظيفة المستهدفة"
                        : "Step 3: Target Job Posting"}
                    </h3>
                    <div>
                      <label className="mb-1 block text-xs text-zinc-400">
                        {isAr ? "المسمى الوظيفي المطلوب" : "Target Job Title"}
                      </label>
                      <input
                        type="text"
                        value={targetRole}
                        onChange={(e) => setTargetRole(e.target.value)}
                        className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                      />
                    </div>

                    <div className="mt-3.5">
                      <label className="mb-1.5 block text-xs text-zinc-400">
                        {isAr
                          ? `المهارات المطلوبة في الوظيفة (${requiredSkills.length})`
                          : `Required Job Skills (${requiredSkills.length})`}
                      </label>
                      <div className="flex flex-wrap gap-1.5 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-2.5">
                        {requiredSkills.map((sk) => (
                          <span
                            key={sk}
                            className="inline-flex items-center gap-1 rounded-md border border-zinc-700/80 bg-zinc-800/70 px-2 py-0.5 font-mono text-xs text-zinc-200"
                          >
                            {sk}
                            <button
                              onClick={() =>
                                setRequiredSkills(
                                  requiredSkills.filter((s) => s !== sk)
                                )
                              }
                              className="text-zinc-500 hover:text-zinc-200"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </span>
                        ))}
                      </div>
                      <div className="mt-2 flex gap-2">
                        <input
                          type="text"
                          list="skill-vocab-list"
                          value={newReqSkillInput}
                          onChange={(e) => setNewReqSkillInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              addRequiredSkill(newReqSkillInput);
                            }
                          }}
                          placeholder={
                            isAr
                              ? "أضف مهارة مطلوبة..."
                              : "Add required skill..."
                          }
                          className="flex-1 rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-1.5 text-xs text-zinc-100 focus:border-zinc-600 focus:outline-none"
                        />
                        <button
                          onClick={() => addRequiredSkill(newReqSkillInput)}
                          className="inline-flex items-center gap-1 rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:border-zinc-700 hover:text-white"
                        >
                          <Plus className="h-3.5 w-3.5" />
                          {isAr ? "إضافة" : "Add"}
                        </button>
                      </div>
                    </div>

                    <div className="mt-3.5">
                      <label className="mb-1 block text-xs text-zinc-400">
                        {isAr
                          ? "وصف الوظيفة (اختياري)"
                          : "Job Description Snippet (Optional)"}
                      </label>
                      <textarea
                        rows={2}
                        value={jobDescription}
                        onChange={(e) => setJobDescription(e.target.value)}
                        className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 p-2.5 text-xs text-zinc-200 focus:border-zinc-600 focus:outline-none"
                      />
                    </div>

                    <button
                      onClick={() => runJobMatchEvaluation()}
                      disabled={matchLoading}
                      className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-100 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-white"
                    >
                      <Briefcase className="h-3.5 w-3.5" />
                      {matchLoading
                        ? isAr
                          ? "جاري التحليل..."
                          : "Analyzing Match..."
                        : isAr
                        ? "تحليل مدى توافق الوظيفة"
                        : "Analyze Job Match"}
                    </button>
                  </div>

                  <datalist id="skill-vocab-list">
                    {SKILL_VOCABULARY.map((s) => (
                      <option key={s} value={s} />
                    ))}
                  </datalist>
                </div>

                {/* Right Column: Clean Results */}
                <div className="space-y-5 lg:col-span-7">
                  {matchResult && (
                    <motion.div
                      key={`${matchResult.current_role}-${matchResult.target_job_title}-${matchResult.fit_score_percentage}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.25 }}
                      className="space-y-5"
                    >
                      {/* Main Score Card */}
                      <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800/80 pb-4">
                          <div>
                            <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                              {isAr
                                ? "نتيجة توافق السيرة الذاتية"
                                : "Resume-to-Job Alignment"}
                            </span>
                            <h2 className="mt-0.5 text-lg font-semibold text-zinc-100">
                              {matchResult.current_role}{" "}
                              <span className="text-zinc-500">➔</span>{" "}
                              {matchResult.target_job_title}
                            </h2>
                          </div>
                          <span className="inline-flex items-center gap-1 rounded-full border border-zinc-800 bg-zinc-950 px-2.5 py-1 font-mono text-[11px] text-zinc-400">
                            <Clock className="h-3 w-3" />
                            {matchResult.latency_ms} ms
                          </span>
                        </div>

                        <div className="mt-6 grid items-center gap-6 sm:grid-cols-12">
                          <div className="sm:col-span-5">
                            <MinimalCircularGauge
                              percentage={matchResult.fit_score_raw * 100}
                              tone={
                                matchResult.predicted_match_tier === "High"
                                  ? "emerald"
                                  : matchResult.predicted_match_tier ===
                                    "Medium"
                                  ? "slate"
                                  : "amber"
                              }
                              label={isAr ? "نسبة التوافق" : "Match Score"}
                              sublabel={`${
                                isAr ? "تغطية المهارات" : "Skill Coverage"
                              }: ${matchResult.skill_match_ratio}`}
                            />
                          </div>

                          <div className="space-y-4 sm:col-span-7">
                            <div className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3.5">
                              <span className="text-xs text-zinc-400">
                                {isAr ? "مستوى التوافق" : "Alignment Level"}
                              </span>
                              <span
                                className={`rounded-lg border px-2.5 py-1 text-xs font-semibold ${
                                  matchResult.predicted_match_tier === "High"
                                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                                    : matchResult.predicted_match_tier ===
                                      "Medium"
                                    ? "border-zinc-700 bg-zinc-800 text-zinc-200"
                                    : "border-amber-500/30 bg-amber-500/10 text-amber-300"
                                }`}
                              >
                                {matchResult.predicted_match_tier} Fit
                              </span>
                            </div>

                            <div className="space-y-2.5 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3.5">
                              <div className="text-xs font-medium text-zinc-400">
                                {isAr
                                  ? "توزيع الثقة الإحصائية"
                                  : "Confidence Distribution"}
                              </div>
                              {(["High", "Medium", "Low"] as const).map(
                                (tier) => {
                                  const valStr =
                                    matchResult.tier_probabilities[tier] ||
                                    "0%";
                                  const num = parseFloat(valStr) || 0;
                                  return (
                                    <div key={tier} className="space-y-1">
                                      <div className="flex justify-between font-mono text-xs">
                                        <span className="text-zinc-400">
                                          {tier}
                                        </span>
                                        <span className="text-zinc-200">
                                          {valStr}
                                        </span>
                                      </div>
                                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                                        <motion.div
                                          initial={{ width: 0 }}
                                          animate={{
                                            width: `${Math.min(100, num)}%`,
                                          }}
                                          transition={{ duration: 0.5 }}
                                          className={`h-full ${
                                            tier === "High"
                                              ? "bg-emerald-400"
                                              : tier === "Medium"
                                              ? "bg-zinc-300"
                                              : "bg-zinc-500"
                                          }`}
                                        />
                                      </div>
                                    </div>
                                  );
                                }
                              )}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Matched vs Missing Skills */}
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5">
                          <div className="mb-3 flex items-center justify-between">
                            <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
                              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                              {isAr
                                ? "المهارات المتوفرة لديك"
                                : "Matched Skills"}
                            </div>
                            <span className="rounded-full bg-zinc-800 px-2 py-0.5 font-mono text-xs text-zinc-300">
                              {matchResult.matched_skills.length}
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {matchResult.matched_skills.length > 0 ? (
                              matchResult.matched_skills.map((sk) => (
                                <span
                                  key={sk}
                                  className="rounded-md border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 font-mono text-xs text-emerald-300"
                                >
                                  ✓ {sk}
                                </span>
                              ))
                            ) : (
                              <span className="text-xs text-zinc-500">
                                {isAr
                                  ? "لا توجد مهارات مشتركة مباشرة"
                                  : "No direct skill overlap found."}
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5">
                          <div className="mb-3 flex items-center justify-between">
                            <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
                              <AlertCircle className="h-4 w-4 text-amber-400" />
                              {isAr
                                ? "مهارات تحتاج لاكتسابها"
                                : "Skills to Acquire"}
                            </div>
                            <span className="rounded-full bg-zinc-800 px-2 py-0.5 font-mono text-xs text-zinc-300">
                              {matchResult.missing_skill_gap_count}
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {matchResult.missing_skill_gap.length > 0 ? (
                              matchResult.missing_skill_gap.map((sk) => (
                                <span
                                  key={sk}
                                  className="rounded-md border border-zinc-700 bg-zinc-800/90 px-2.5 py-1 font-mono text-xs text-zinc-200"
                                >
                                  + {sk}
                                </span>
                              ))
                            ) : (
                              <span className="text-xs text-emerald-400">
                                {isAr
                                  ? "جميع المهارات المطلوبة متوفرة لديك!"
                                  : "All required skills are covered!"}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* 3-Phase Action Plan */}
                      <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
                        <div className="mb-4 flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Layers className="h-4 w-4 text-zinc-400" />
                            <h3 className="text-sm font-semibold text-zinc-100">
                              {isAr
                                ? "خطة استكمال المهارات المقترحة"
                                : "Recommended Learning Plan"}
                            </h3>
                          </div>
                          <button
                            onClick={() => {
                              runTrajectoryEvaluation(
                                currentRole,
                                tenureYears,
                                targetRole,
                                stepIndex
                              );
                              setActiveTab("trajectory");
                            }}
                            className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:bg-zinc-800"
                          >
                            <span>
                              {isAr
                                ? "عرض المسار الوظيفي الكامل"
                                : "View Career Trajectory"}
                            </span>
                            <ArrowRight className="h-3.5 w-3.5" />
                          </button>
                        </div>

                        <div className="grid gap-3.5 md:grid-cols-3">
                          {Object.entries(matchResult.upskilling_roadmap).map(
                            ([phase, skills], idx) => (
                              <div
                                key={phase}
                                className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-4"
                              >
                                <div className="mb-2 flex items-center gap-2">
                                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-800 font-mono text-[11px] font-semibold text-zinc-200">
                                    {idx + 1}
                                  </span>
                                  <span className="text-xs font-semibold text-zinc-200">
                                    {phase}
                                  </span>
                                </div>
                                <ul className="space-y-1.5 text-xs text-zinc-400">
                                  {(Array.isArray(skills)
                                    ? skills
                                    : [String(skills)]
                                  ).map((s) => (
                                    <li
                                      key={s}
                                      className="flex items-center gap-1.5 font-mono"
                                    >
                                      <ChevronRight className="h-3 w-3 shrink-0 text-zinc-500" />
                                      <span>{s}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* ================================================================
              TAB 3: CAREER PATH (TRAJECTORY & ROADMAP)
             ================================================================ */}
          {activeTab === "trajectory" && (
            <motion.div
              key="trajectory"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="grid gap-8 lg:grid-cols-12"
            >
              {/* Left Inputs */}
              <div className="space-y-5 lg:col-span-4">
                <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
                  <h3 className="mb-4 text-sm font-semibold text-zinc-100">
                    {isAr
                      ? "تخطيط الانتقال الوظيفي"
                      : "Career Transition Planner"}
                  </h3>

                  <div className="space-y-4">
                    <div>
                      <label className="mb-1 block text-xs text-zinc-400">
                        {isAr ? "المسمى الوظيفي الحالي" : "Current Role"}
                      </label>
                      <input
                        type="text"
                        value={currentRole}
                        onChange={(e) => setCurrentRole(e.target.value)}
                        className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3.5 py-2 text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-xs text-zinc-400">
                        {isAr
                          ? `سنوات الخبرة (${tenureYears} سنة)`
                          : `Years of Experience (${tenureYears} yrs)`}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="20"
                        step="0.5"
                        value={Math.min(20, tenureYears)}
                        onChange={(e) =>
                          setTenureYears(parseFloat(e.target.value))
                        }
                        className="w-full accent-zinc-200"
                      />
                    </div>

                    <div>
                      <label className="mb-1 block text-xs text-zinc-400">
                        {isAr ? "المنصب المستقبلي المستهدف" : "Target Future Role"}
                      </label>
                      <input
                        type="text"
                        value={targetRole}
                        onChange={(e) => setTargetRole(e.target.value)}
                        className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3.5 py-2 text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                      />
                    </div>

                    <button
                      onClick={() =>
                        runTrajectoryEvaluation(
                          currentRole,
                          tenureYears,
                          targetRole,
                          stepIndex
                        )
                      }
                      disabled={trajectoryLoading}
                      className="flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-100 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-white"
                    >
                      <Compass className="h-3.5 w-3.5" />
                      {trajectoryLoading
                        ? isAr
                          ? "جاري الحساب..."
                          : "Analyzing Path..."
                        : isAr
                        ? "تحليل المسار وخارطة الطريق"
                        : "Analyze Career Path"}
                    </button>
                  </div>
                </div>

                {/* Quick Examples */}
                <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5">
                  <div className="mb-2.5 text-xs font-medium text-zinc-400">
                    {isAr ? "أمثلة مسارات شائعة" : "Popular Career Paths"}
                  </div>
                  <div className="space-y-1.5">
                    {[
                      [
                        "Junior Software Engineer",
                        2.5,
                        "Senior Software Engineer",
                      ],
                      ["Data Analyst", 3.0, "Machine Learning Engineer"],
                      ["Junior Frontend Developer", 2.0, "Full Stack Engineer"],
                      [
                        "Customer Support Specialist",
                        2.0,
                        "Sales Account Manager",
                      ],
                    ].map(([c, ten, t]) => (
                      <button
                        key={`${c}-${t}`}
                        onClick={() => {
                          setCurrentRole(String(c));
                          setTenureYears(Number(ten));
                          setTargetRole(String(t));
                          runTrajectoryEvaluation(
                            String(c),
                            Number(ten),
                            String(t),
                            stepIndex
                          );
                        }}
                        className="flex w-full items-center justify-between rounded-xl border border-zinc-800/70 bg-zinc-950/60 px-3 py-2 text-left text-xs text-zinc-300 transition hover:border-zinc-700 hover:text-white"
                      >
                        <span>
                          {c} ➔ <strong className="text-zinc-100">{t}</strong>
                        </span>
                        <span className="font-mono text-[10px] text-zinc-500">
                          {ten}y
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Output */}
              <div className="space-y-5 lg:col-span-8">
                {trajectoryResult && (
                  <motion.div
                    key={`${trajectoryResult.current_role}-${trajectoryResult.target_future_role}-${trajectoryResult.transition_probability}`}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                    className="space-y-5"
                  >
                    <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
                      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800/80 pb-4">
                        <div>
                          <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                            {isAr
                              ? "تقييم قابلية الانتقال الوظيفي"
                              : "Career Transition Assessment"}
                          </span>
                          <h2 className="mt-0.5 text-lg font-semibold text-zinc-100">
                            {trajectoryResult.current_role}{" "}
                            <span className="text-zinc-500">➔</span>{" "}
                            {trajectoryResult.target_future_role}
                          </h2>
                        </div>
                        <span className="inline-flex items-center gap-1 rounded-full border border-zinc-800 bg-zinc-950 px-2.5 py-1 font-mono text-[11px] text-zinc-400">
                          <Clock className="h-3 w-3" />
                          {trajectoryResult.latency_ms} ms
                        </span>
                      </div>

                      <div className="mt-6 grid items-center gap-6 sm:grid-cols-12">
                        <div className="sm:col-span-5">
                          <MinimalCircularGauge
                            percentage={
                              trajectoryResult.transition_probability_raw * 100
                            }
                            tone={
                              trajectoryResult.transition_probability_raw >=
                              0.45
                                ? "emerald"
                                : trajectoryResult.transition_probability_raw >=
                                  0.18
                                ? "slate"
                                : "amber"
                            }
                            label={isAr ? "احتمالية النجاح" : "Feasibility"}
                            sublabel={trajectoryResult.feasibility_assessment}
                          />
                        </div>

                        <div className="space-y-3.5 sm:col-span-7">
                          <div className="grid grid-cols-2 gap-3">
                            <div className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3">
                              <div className="text-[11px] text-zinc-500">
                                {isAr ? "نوع الخطوة" : "Transition Type"}
                              </div>
                              <div className="mt-1 text-xs font-semibold text-zinc-100">
                                {trajectoryResult.predicted_transition_type.replace(
                                  "_",
                                  " "
                                )}
                              </div>
                            </div>
                            <div className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3">
                              <div className="text-[11px] text-zinc-500">
                                {isAr ? "تدرج الأقدمية" : "Seniority Path"}
                              </div>
                              <div className="mt-1 font-mono text-xs font-semibold text-zinc-200">
                                {trajectoryResult.current_seniority} ➔{" "}
                                {trajectoryResult.target_seniority} (
                                <span className="text-emerald-400">
                                  {trajectoryResult.seniority_delta}
                                </span>
                                )
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3.5">
                            <div className="flex items-center justify-between text-xs text-zinc-400">
                              <span>
                                {isAr
                                  ? "احتمالات تصنيف الانتقال"
                                  : "Transition Category Breakdown"}
                              </span>
                              <span className="font-mono text-[11px] text-zinc-300">
                                ~{trajectoryResult.estimated_skill_gap_count}{" "}
                                skills required
                              </span>
                            </div>
                            {(
                              [
                                "Vertical_Promotion",
                                "Adjacent_Pivot",
                                "Career_Shift",
                              ] as const
                            ).map((tKey) => {
                              const valStr =
                                trajectoryResult.type_distribution[tKey] ||
                                "0%";
                              const num = parseFloat(valStr) || 0;
                              return (
                                <div key={tKey} className="space-y-1">
                                  <div className="flex justify-between font-mono text-xs">
                                    <span className="text-zinc-400">
                                      {tKey.replace("_", " ")}
                                    </span>
                                    <span className="text-zinc-200">
                                      {valStr}
                                    </span>
                                  </div>
                                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                                    <motion.div
                                      initial={{ width: 0 }}
                                      animate={{
                                        width: `${Math.min(100, num)}%`,
                                      }}
                                      transition={{ duration: 0.5 }}
                                      className="h-full bg-zinc-200"
                                    />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* 3-Phase Roadmap */}
                    <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
                      <div className="mb-4">
                        <h3 className="text-sm font-semibold text-zinc-100">
                          {isAr
                            ? "خارطة الطريق المهارية للوصول إلى المنصب المستهدف"
                            : "Step-by-Step Competency Roadmap"}
                        </h3>
                        <p className="text-xs text-zinc-500">
                          {isAr
                            ? "مسترجعة من قاعدة بيانات تضم 230,333 خارطة طريق وظيفية موثقة"
                            : "Retrieved from 230,333 verified career progression roadmaps"}
                        </p>
                      </div>

                      <div className="grid gap-4 md:grid-cols-3">
                        {Object.entries(
                          trajectoryResult.actionable_roadmap
                        ).map(([phase, skills], idx) => (
                          <div
                            key={phase}
                            className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-4"
                          >
                            <div className="mb-2.5 flex items-center gap-2">
                              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-800 font-mono text-[11px] font-semibold text-zinc-200">
                                {idx + 1}
                              </span>
                              <span className="text-xs font-semibold text-zinc-200">
                                {phase}
                              </span>
                            </div>
                            <ul className="space-y-2 text-xs text-zinc-400">
                              {(Array.isArray(skills)
                                ? skills
                                : [String(skills)]
                              ).map((s) => (
                                <li
                                  key={s}
                                  className="flex items-start gap-1.5 leading-snug"
                                >
                                  <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
                                  <span>{s}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>

                      {/* Top-3 Retrieved RAG Knowledge Base Records */}
                      {ragMatches.length > 0 && (
                        <div className="mt-5 border-t border-zinc-800/80 pt-4">
                          <div className="mb-2.5 flex items-center justify-between">
                            <span className="text-xs font-semibold text-zinc-300">
                              {isAr
                                ? "أقرب السجلات المسترجعة من قاعدة المعرفة (230,333 RAG Index)"
                                : "Top Retrieved Records from 230,333-Roadmap RAG Knowledge Base"}
                            </span>
                            <span className="font-mono text-[11px] text-emerald-400">
                              TF-IDF Cosine Retrieval
                            </span>
                          </div>
                          <div className="grid gap-3 md:grid-cols-3">
                            {ragMatches.map((m, i) => (
                              <div
                                key={`${m.target_future_role}-${i}`}
                                className="rounded-xl border border-zinc-800/70 bg-zinc-950/40 p-3 text-xs"
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="truncate font-semibold text-zinc-200">
                                    {m.target_future_role}
                                  </span>
                                  <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-emerald-400">
                                    sim: {m.similarity}
                                  </span>
                                </div>
                                <p className="mt-1 line-clamp-2 text-[11px] text-zinc-500">
                                  {m.target_job_description ||
                                    "Verified occupational skill-gap record"}
                                </p>
                                <div className="mt-1.5 font-mono text-[10px] text-zinc-400">
                                  {m.skill_gap_count} indexed skills
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )}

          {/* ================================================================
              TAB 4: NEXT STEPS (CAREER RECOMMENDER)
             ================================================================ */}
          {activeTab === "recommend" && (
            <motion.div
              key="recommend"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="grid gap-8 lg:grid-cols-12"
            >
              {/* Left Config */}
              <div className="space-y-5 lg:col-span-4">
                <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
                  <h3 className="mb-4 text-sm font-semibold text-zinc-100">
                    {isAr
                      ? "استكشاف أفضل الخطوات الوظيفية التالية"
                      : "Next-Step Role Explorer"}
                  </h3>

                  <div className="space-y-4">
                    <div>
                      <label className="mb-1 block text-xs text-zinc-400">
                        {isAr ? "المسمى الوظيفي الحالي" : "Current Role"}
                      </label>
                      <input
                        type="text"
                        value={currentRole}
                        onChange={(e) => setCurrentRole(e.target.value)}
                        className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3.5 py-2 text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="mb-1 block text-xs text-zinc-400">
                          {isAr ? "سنوات الخبرة" : "Experience (Yrs)"}
                        </label>
                        <input
                          type="number"
                          step="0.5"
                          value={tenureYears}
                          onChange={(e) =>
                            setTenureYears(parseFloat(e.target.value) || 0)
                          }
                          className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2 font-mono text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-zinc-400">
                          {isAr ? "عدد النتائج" : "Top Results"}
                        </label>
                        <input
                          type="number"
                          min="2"
                          max="8"
                          value={topN}
                          onChange={(e) =>
                            setTopN(parseInt(e.target.value, 10) || 4)
                          }
                          className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2 font-mono text-sm text-zinc-100 focus:border-zinc-600 focus:outline-none"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="mb-1.5 block text-xs text-zinc-400">
                        {isAr
                          ? `الوظائف المرشحة للمقارنة (${candidateRoles.length})`
                          : `Candidate Roles to Compare (${candidateRoles.length})`}
                      </label>
                      <div className="flex flex-wrap gap-1.5 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-2.5">
                        {candidateRoles.map((cr) => (
                          <span
                            key={cr}
                            className="inline-flex items-center gap-1 rounded-md border border-zinc-700/80 bg-zinc-800/70 px-2 py-0.5 text-xs text-zinc-300"
                          >
                            {cr}
                            <button
                              onClick={() =>
                                setCandidateRoles(
                                  candidateRoles.filter((r) => r !== cr)
                                )
                              }
                              className="text-zinc-500 hover:text-zinc-200"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </span>
                        ))}
                      </div>
                      <div className="mt-2 flex gap-2">
                        <input
                          type="text"
                          value={newCandidateRoleInput}
                          onChange={(e) =>
                            setNewCandidateRoleInput(e.target.value)
                          }
                          onKeyDown={(e) => {
                            if (
                              e.key === "Enter" &&
                              newCandidateRoleInput.trim()
                            ) {
                              e.preventDefault();
                              setCandidateRoles([
                                ...candidateRoles,
                                newCandidateRoleInput.trim(),
                              ]);
                              setNewCandidateRoleInput("");
                            }
                          }}
                          placeholder={
                            isAr ? "أضف وظيفة للمقارنة..." : "Add role to rank..."
                          }
                          className="flex-1 rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-1.5 text-xs text-zinc-100 focus:border-zinc-600 focus:outline-none"
                        />
                        <button
                          onClick={() => {
                            if (newCandidateRoleInput.trim()) {
                              setCandidateRoles([
                                ...candidateRoles,
                                newCandidateRoleInput.trim(),
                              ]);
                              setNewCandidateRoleInput("");
                            }
                          }}
                          className="rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:border-zinc-700 hover:text-white"
                        >
                          {isAr ? "إضافة" : "Add"}
                        </button>
                      </div>
                    </div>

                    <button
                      onClick={() =>
                        runRecommendationEvaluation(
                          currentRole,
                          tenureYears,
                          candidateRoles,
                          topN
                        )
                      }
                      disabled={recLoading}
                      className="flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-100 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-white"
                    >
                      <TrendingUp className="h-3.5 w-3.5" />
                      {recLoading
                        ? isAr
                          ? "جاري الترتيب..."
                          : "Ranking..."
                        : isAr
                        ? "ترتيب أفضل الخطوات التالية"
                        : "Rank Best Next Roles"}
                    </button>
                  </div>
                </div>
              </div>

              {/* Right Ranked Cards */}
              <div className="space-y-3.5 lg:col-span-8">
                {recResult &&
                  recResult.recommendations.map((rec, idx) => {
                    const isExpanded = expandedRecIdx === idx;
                    const pct = rec.transition_probability_raw * 100;
                    return (
                      <motion.div
                        key={rec.target_future_role}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.2, delay: idx * 0.05 }}
                        className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-4">
                          <div className="flex items-center gap-3.5">
                            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-zinc-700 bg-zinc-800 font-mono text-xs font-semibold text-zinc-100">
                              #{idx + 1}
                            </div>
                            <div>
                              <div className="flex flex-wrap items-center gap-2">
                                <h3 className="text-sm font-semibold text-zinc-100">
                                  {rec.target_future_role}
                                </h3>
                                <span className="rounded-md border border-zinc-700/80 bg-zinc-800/80 px-2 py-0.5 text-[11px] text-zinc-300">
                                  {rec.predicted_transition_type.replace(
                                    "_",
                                    " "
                                  )}
                                </span>
                              </div>
                              <p className="mt-0.5 text-xs text-zinc-400">
                                {rec.feasibility_assessment} •{" "}
                                {rec.current_seniority} ➔ {rec.target_seniority}{" "}
                                ({rec.seniority_delta})
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-2.5">
                            <div className="text-right">
                              <div className="font-mono text-lg font-semibold text-zinc-100">
                                {rec.transition_probability}
                              </div>
                              <div className="text-[10px] uppercase text-zinc-500">
                                {isAr ? "الاحتمالية" : "Feasibility"}
                              </div>
                            </div>
                            <button
                              onClick={() =>
                                setExpandedRecIdx(isExpanded ? null : idx)
                              }
                              className="rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:border-zinc-700 hover:text-white"
                            >
                              {isExpanded
                                ? isAr
                                  ? "إخفاء الخطة"
                                  : "Hide Plan"
                                : isAr
                                ? "عرض الخطة"
                                : "View Plan"}
                            </button>
                            <button
                              onClick={() => {
                                setTargetRole(rec.target_future_role);
                                runJobMatchEvaluation({
                                  target_role: rec.target_future_role,
                                });
                                setActiveTab("match");
                              }}
                              className="rounded-lg bg-zinc-100 px-3 py-1.5 text-xs font-semibold text-zinc-950 hover:bg-white"
                            >
                              {isAr ? "فحص التوافق" : "Check Fit"}
                            </button>
                          </div>
                        </div>

                        <div className="mt-3.5 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(100, pct)}%` }}
                            transition={{ duration: 0.5 }}
                            className="h-full bg-zinc-200"
                          />
                        </div>

                        <AnimatePresence initial={false}>
                          {isExpanded && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.22 }}
                              className="mt-4 grid gap-3 border-t border-zinc-800/80 pt-4 md:grid-cols-3"
                            >
                              {Object.entries(rec.actionable_roadmap).map(
                                ([phase, skills]) => (
                                  <div
                                    key={phase}
                                    className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3.5"
                                  >
                                    <div className="mb-1.5 text-xs font-semibold text-zinc-200">
                                      {phase}
                                    </div>
                                    <ul className="space-y-1 text-xs text-zinc-400">
                                      {(Array.isArray(skills)
                                        ? skills
                                        : [String(skills)]
                                      ).map((s) => (
                                        <li key={s}>• {s}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )
                              )}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    );
                  })}
              </div>
            </motion.div>
          )}

          {/* ================================================================
              TAB 5: TECHNICAL REPORT
             ================================================================ */}
          {activeTab === "report" && <DefenseReportTab lang={lang} />}
        </AnimatePresence>
      </main>
    </div>
  );
}
