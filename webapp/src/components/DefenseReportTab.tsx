"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Printer,
  ShieldCheck,
  Database,
  Cpu,
  BarChart3,
  Scale,
  HelpCircle,
  ChevronDown,
  CheckCircle2,
  Layers,
  Lock,
} from "lucide-react";

interface DefenseReportTabProps {
  lang: "en" | "ar";
}

const QA_ITEMS = [
  {
    qEn: "Q1: How did you guarantee Zero Data Leakage across the 1,140,000 career transition records and 221,903 resume-matching records?",
    qAr: "س1: كيف ضمنتم عدم وجود أي تسريب للبيانات (Zero Data Leakage) عبر 1,140,000 سجل للمسار الوظيفي و221,903 سجل لمطابقة السير الذاتية؟",
    aEn: "Both models enforce two-stage Person/Candidate-Level GroupShuffleSplit (SEED=42). In the Career Trajectory Model, records are grouped by person_id (801,603 unique individuals) into Train (798,192), Validation (170,996), and Held-Out Test (170,812) with 0 shared individuals. In the Resume-Job Matching Model, records are grouped by candidate_id (37,044 unique resume signatures) into Train (155,087), Validation (32,794), and Held-Out Test (34,022) with 0 shared resumes. All frequency tables, TF-IDF vectorizers, 24-D TruncatedSVD projections, and StandardScaler statistics are fitted strictly on the training split.",
    aAr: "يستخدم كلا النموذجين تقسيم GroupShuffleSplit على مرحلتين (SEED=42) على مستوى الشخص أو السيرة الذاتية. في نموذج المسار الوظيفي تم التجميع حسب person_id (801,603 شخص فريد) إلى تدريب (798,192)، تحقق (170,996)، واختبار معزول (170,812) بتقاطع صفر. وفي نموذج مطابقة الوظائف تم التجميع حسب candidate_id (37,044 سيرة ذاتية فريدة) إلى تدريب (155,087)، تحقق (32,794)، واختبار معزول (34,022) بتقاطع صفر.",
  },
  {
    qEn: "Q2: What preprocessing improvement unlocked 82.73% accuracy on the 221,903-row Resume-Job Matching dataset?",
    qAr: "س2: ما هو التحسين الهندسي في معالجة البيانات الذي رفع دقة نموذج مطابقة السير الذاتية إلى 82.73%؟",
    aEn: "In the initial preprocessing pipeline, applying a rare-category cutoff (<50 occurrences) to target job titles collapsed over 85% of the 84,508 unique job titles into a generic 'other' token. Restoring all 84,508 unmasked normalized job_title strings enabled fine-grained role-to-job TF-IDF cosine alignment and raised 3-class test accuracy on the 34,022 held-out test records to 82.73% (Macro F1 = 0.8259).",
    aAr: "في المعالجة الأولية، أدى تطبيق حد التكرار (<50) على المسميات الوظيفية المستهدفة إلى تحويل أكثر من 85% من أصل 84,508 مسمى وظيفي فريد إلى كلمة 'other'. قمنا باسترجاع جميع الـ 84,508 مسمى وظيفي الحقيقي، مما أتاح حساب التشابه الدلالي الدقيق ورفع دقة التصنيف الثلاثي على بيانات الاختبار (34,022 سجل) إلى 82.73% (Macro F1 = 0.8259).",
  },
  {
    qEn: "Q3: Why combine a PyTorch Residual Multi-Task Neural Network with Gradient Boosted Decision Trees?",
    qAr: "س3: لماذا قمتم بدمج شبكة عصبية عميقة متعددة المهام (PyTorch Multi-Task DNN) مع أشجار القرار المعززة (HistGradientBoosting)؟",
    aEn: "Deep Residual MLPs excel at learning smooth representations over dense continuous SVD embeddings ([c_svd, t_svd, |c_svd - t_svd|, c_svd * t_svd]) and joint multi-task representation sharing between regression and classification heads. Conversely, HistGradientBoosting excels at sharp threshold splits on tabular features (tenure_years, skill_gap_count, skill_match_ratio, seniority_level_change). Blending them followed by validation-set IsotonicRegression calibration reduces variance and improves reliability.",
    aAr: "تتفوق الشبكات العصبية العميقة ذات الوصلات المتبقية (Residual MLP) في تمثيل التفاعلات الدلالية الكثيفة (SVD Embeddings)، بينما تتفوق أشجار HistGradientBoosting في التقسيمات الحادة للميزات الجدولية مثل سنوات الخبرة ونسبة تطابق المهارات. دمج النموذجين يقلل التباين ويحقق أفضل معايرة احتمالية.",
  },
  {
    qEn: "Q4: Why do you report Within-Source Cohort MAE alongside Pooled R² on the 1.14M Career Trajectory dataset?",
    qAr: "س4: لماذا تعرضون خطأ MAE داخل كل مصدر بيانات إلى جانب معامل R² الإجمالي في بيانات المسار الوظيفي (1.14 مليون سجل)؟",
    aEn: "The 1.14M dataset merges empirical occupational transition frequencies from ESCO/JobHop (mean probability 0.011–0.019) with LinkedIn promotion feasibility scores (mean probability 0.461). Pooled R² (0.9898) largely reflects separation across these regimes, so reporting Within-Source Cohort MAE (ESCO MAE = 0.0041, JobHop MAE = 0.0036, LinkedIn MAE = 0.0304) and 3-class Transition Accuracy (86.85%) provides transparent within-source evaluation.",
    aAr: "تدمج قاعدة البيانات البالغة 1.14 مليون سجل بين احتمالات الانتقال الإحصائية في ESCO/JobHop ودرجات قابلية الترقية في LinkedIn؛ لذلك نعرض خطأ MAE داخل كل مصدر على حدة بشفافية علمية كاملة (ESCO MAE = 0.0041، JobHop MAE = 0.0036، LinkedIn MAE = 0.0304) إلى جانب دقة التصنيف (86.85%).",
  },
  {
    qEn: "Q5: Why use a 230,333-roadmap TF-IDF Cosine Index instead of a generative LLM for skill roadmaps?",
    qAr: "س5: لماذا استخدمتم فهرس استرجاع TF-IDF Cosine على 230,333 خارطة طريق بدلاً من نموذج لغوي توليدي؟",
    aEn: "1) Zero Hallucination: Every returned curriculum comes verbatim from 230,333 verified occupational roadmaps (103,207 unique target roles). 2) Sub-20ms Latency & Zero External Cost: Sparse cosine retrieval over a 230,333 × 20,000 float32 matrix executes in <20ms locally. 3) Skill-Gap Alignment: Selects the curriculum whose skill_gap_count best matches the candidate's expected seniority and domain progression.",
    aAr: "1) انعدام الهلوسة (Zero Hallucination): جميع المهارات المسترجعة موثقة من 230,333 خارطة طريق حقيقية. 2) سرعة استجابة أقل من 20 مللي ثانية. 3) مطابقة حجم الفجوة المهارية مع مستوى الأقدمية والمجال المستهدف.",
  },
];

export default function DefenseReportTab({ lang }: DefenseReportTabProps) {
  const [openQa, setOpenQa] = useState<number | null>(0);
  const isAr = lang === "ar";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6 pb-16"
    >
      {/* Top Banner + Print Button */}
      <div className="print-card flex flex-col justify-between gap-4 rounded-2xl border border-zinc-800/80 bg-zinc-900/50 p-6 md:flex-row md:items-center">
        <div>
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-zinc-700 bg-zinc-800/80 px-3 py-1 text-xs font-medium text-zinc-300">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
            {isAr
              ? "التوثيق الهندسي والعلمي الشامل"
              : "System Architecture, Validation & Fairness Report"}
          </div>
          <h2 className="print-text-dark text-2xl font-semibold tracking-tight text-zinc-100">
            {isAr
              ? "تقرير الأداء والمعمارية: مطابقة الوظائف والتنبؤ بالمسار المهني"
              : "Technical Report: Resume-Job Matching & Career Trajectory Models"}
          </h2>
          <p className="mt-1 text-sm text-zinc-400">
            {isAr
              ? "تقييم شامل على 1,361,903 سجل، معمارية الشبكات العصبية الهجينة، المعايرة الإحصائية، وفحص العدالة."
              : "Validated across 1,361,903 combined records using person-grouped splits, hybrid multi-task learning, and isotonic calibration."}
          </p>
        </div>
        <button
          onClick={() => window.print()}
          className="no-print inline-flex items-center gap-2 self-start rounded-xl bg-zinc-100 px-4 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-white md:self-center"
        >
          <Printer className="h-4 w-4" />
          {isAr ? "طباعة / تصدير PDF" : "Print / Export PDF"}
        </button>
      </div>

      {/* Section 1: Dataset Provenance & Zero-Leakage GroupShuffleSplit */}
      <div className="print-card rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
        <div className="mb-4 flex items-center gap-2.5">
          <Database className="h-4 w-4 text-zinc-400" />
          <h3 className="print-text-dark text-base font-semibold text-zinc-100">
            {isAr
              ? "1. حجم البيانات وتقسيم GroupShuffleSplit الخالي من التسريب"
              : "1. Dataset Provenance & Leak-Free GroupShuffleSplit Partitions"}
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-400">
                <th className="py-3 pr-4 font-medium">Model System</th>
                <th className="px-4 py-3 font-medium">Dataset Size</th>
                <th className="px-4 py-3 font-medium">Grouping Key (0 Overlap)</th>
                <th className="px-4 py-3 font-medium">Coverage &amp; Index</th>
                <th className="px-4 py-3 font-medium">Train (70%)</th>
                <th className="px-4 py-3 font-medium">Validation (15%)</th>
                <th className="py-3 pl-4 font-medium">Held-Out Test (15%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/70 text-zinc-300">
              <tr>
                <td className="py-3.5 pr-4 font-semibold text-zinc-100">
                  Model 1: Resume-to-Job Matching
                </td>
                <td className="px-4 py-3.5 font-mono">221,903 × 26</td>
                <td className="px-4 py-3.5">
                  <span className="font-mono text-emerald-400">candidate_id</span>{" "}
                  (37,044 resumes)
                </td>
                <td className="px-4 py-3.5">
                  114 categories • 84,508 job titles • 588-skill vocabulary
                </td>
                <td className="px-4 py-3.5 font-mono">155,087</td>
                <td className="px-4 py-3.5 font-mono">32,794</td>
                <td className="py-3.5 pl-4 font-mono font-semibold text-emerald-400">
                  34,022 (Overlap: 0)
                </td>
              </tr>
              <tr>
                <td className="py-3.5 pr-4 font-semibold text-zinc-100">
                  Model 2: Career Trajectory &amp; Roadmap
                </td>
                <td className="px-4 py-3.5 font-mono">1,140,000 × 18</td>
                <td className="px-4 py-3.5">
                  <span className="font-mono text-emerald-400">person_id</span>{" "}
                  (801,603 individuals)
                </td>
                <td className="px-4 py-3.5">
                  58,253 current roles • 103,207 target roles • 230,333 roadmaps
                </td>
                <td className="px-4 py-3.5 font-mono">798,192</td>
                <td className="px-4 py-3.5 font-mono">170,996</td>
                <td className="py-3.5 pl-4 font-mono font-semibold text-emerald-400">
                  170,812 (Overlap: 0)
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 2: Hybrid Multi-Task Architecture */}
      <div className="print-card rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
        <div className="mb-4 flex items-center gap-2.5">
          <Cpu className="h-4 w-4 text-zinc-400" />
          <h3 className="print-text-dark text-base font-semibold text-zinc-100">
            {isAr
              ? "2. المعمارية الهجينة متعددة المهام والمعايرة الإحصائية"
              : "2. Hybrid Multi-Task Architecture & Probability Calibration"}
          </h3>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-4">
            <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold text-zinc-200">
              <Layers className="h-3.5 w-3.5 text-zinc-400" />
              Stage 1: Dense Semantic Feature Vector
            </div>
            <p className="text-xs leading-relaxed text-zinc-400">
              Combines 14–18 scaled numerical features (tenure, seniority delta,
              skill gap count, match ratio, lexical Jaccard, TF-IDF cosine
              similarity) with 96 dense semantic interaction features from 24-D{" "}
              <code>TruncatedSVD</code>:{" "}
              <code>[c_svd, t_svd, |c_svd - t_svd|, c_svd * t_svd]</code>.
            </p>
          </div>
          <div className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-4">
            <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold text-zinc-200">
              <Cpu className="h-3.5 w-3.5 text-zinc-400" />
              Stage 2: Multi-Task Residual DNN + GBDT
            </div>
            <p className="text-xs leading-relaxed text-zinc-400">
              PyTorch Residual MLP (<code>Linear ➔ LayerNorm ➔ GELU ➔ 2×ResBlock(256)</code>)
              with dual Sigmoid Regression and 3-Class Classification heads,
              blended with <code>HistGradientBoostingRegressor</code> and{" "}
              <code>HistGradientBoostingClassifier</code>.
            </p>
          </div>
          <div className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-4">
            <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold text-zinc-200">
              <Scale className="h-3.5 w-3.5 text-zinc-400" />
              Stage 3: Isotonic Calibration &amp; Retrieval
            </div>
            <p className="text-xs leading-relaxed text-zinc-400">
              Validation-fitted <code>IsotonicRegression</code> calibrates
              predicted scores to empirical probabilities (reducing Career
              Trajectory Test ECE to <strong>0.01%</strong> and Job Matching
              Test ECE to <strong>1.96%</strong>), paired with a 230,333-roadmap
              TF-IDF retrieval index.
            </p>
          </div>
        </div>
      </div>

      {/* Section 3: Held-Out Test Benchmarks Side-by-Side */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Model 1 Benchmarks */}
        <div className="print-card rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-zinc-400" />
              <h3 className="print-text-dark text-sm font-semibold text-zinc-100">
                Model 1: Resume-Job Matching Test Results (n = 34,022)
              </h3>
            </div>
            <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-xs font-medium text-emerald-400">
              82.73% Acc
            </span>
          </div>
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-400">
                <th className="py-2 font-medium">Architecture</th>
                <th className="py-2 font-medium">Score MAE</th>
                <th className="py-2 font-medium">Score RMSE</th>
                <th className="py-2 font-medium">Score R²</th>
                <th className="py-2 font-medium">Tier Acc</th>
                <th className="py-2 font-medium">Macro F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/70 text-zinc-300">
              <tr>
                <td className="py-2.5">Multi-Task Deep Neural Net</td>
                <td className="py-2.5 font-mono">0.0826</td>
                <td className="py-2.5 font-mono">0.1337</td>
                <td className="py-2.5 font-mono">0.7392</td>
                <td className="py-2.5 font-mono">82.73%</td>
                <td className="py-2.5 font-mono">0.8263</td>
              </tr>
              <tr>
                <td className="py-2.5">HistGradientBoosting Trees</td>
                <td className="py-2.5 font-mono">0.1131</td>
                <td className="py-2.5 font-mono">0.1549</td>
                <td className="py-2.5 font-mono">0.6499</td>
                <td className="py-2.5 font-mono">80.36%</td>
                <td className="py-2.5 font-mono">0.8022</td>
              </tr>
              <tr className="font-semibold text-zinc-100">
                <td className="py-2.5">Calibrated Hybrid Ensemble</td>
                <td className="py-2.5 font-mono">0.0899</td>
                <td className="py-2.5 font-mono">0.1373</td>
                <td className="py-2.5 font-mono">0.7250</td>
                <td className="py-2.5 font-mono text-emerald-400">82.73%</td>
                <td className="py-2.5 font-mono text-emerald-400">0.8259</td>
              </tr>
            </tbody>
          </table>
          <div className="mt-4 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3 text-xs text-zinc-400">
            <div className="font-medium text-zinc-200">
              Per-Class F1 &amp; Calibration (n = 34,022):
            </div>
            <div className="mt-1 grid grid-cols-3 gap-2 font-mono text-zinc-300">
              <div>Low F1: 0.8526</div>
              <div>Medium F1: 0.7597</div>
              <div>High F1: 0.8090</div>
            </div>
            <div className="mt-1">
              Test ECE: <code>2.74%</code> ➔{" "}
              <span className="font-medium text-emerald-400">1.96%</span> |
              Brier Score: <code>0.1510</code>
            </div>
          </div>
        </div>

        {/* Model 2 Benchmarks */}
        <div className="print-card rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-zinc-400" />
              <h3 className="print-text-dark text-sm font-semibold text-zinc-100">
                Model 2: Career Trajectory Test Results (n = 170,812)
              </h3>
            </div>
            <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-xs font-medium text-emerald-400">
              86.85% Acc • ECE 0.01%
            </span>
          </div>
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-400">
                <th className="py-2 font-medium">Architecture</th>
                <th className="py-2 font-medium">Prob MAE</th>
                <th className="py-2 font-medium">Prob RMSE</th>
                <th className="py-2 font-medium">Pooled R²</th>
                <th className="py-2 font-medium">Type Acc</th>
                <th className="py-2 font-medium">Macro F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/70 text-zinc-300">
              <tr>
                <td className="py-2.5">Multi-Task Deep Neural Net</td>
                <td className="py-2.5 font-mono">0.0125</td>
                <td className="py-2.5 font-mono">0.0252</td>
                <td className="py-2.5 font-mono">0.9849</td>
                <td className="py-2.5 font-mono">85.56%</td>
                <td className="py-2.5 font-mono">0.7964</td>
              </tr>
              <tr>
                <td className="py-2.5">HistGradientBoosting Trees</td>
                <td className="py-2.5 font-mono">0.0100</td>
                <td className="py-2.5 font-mono">0.0203</td>
                <td className="py-2.5 font-mono">0.9902</td>
                <td className="py-2.5 font-mono">87.47%</td>
                <td className="py-2.5 font-mono">0.8184</td>
              </tr>
              <tr className="font-semibold text-zinc-100">
                <td className="py-2.5">Calibrated Hybrid Ensemble</td>
                <td className="py-2.5 font-mono">0.0107</td>
                <td className="py-2.5 font-mono">0.0207</td>
                <td className="py-2.5 font-mono">0.9898</td>
                <td className="py-2.5 font-mono text-emerald-400">86.85%</td>
                <td className="py-2.5 font-mono text-emerald-400">0.8107</td>
              </tr>
            </tbody>
          </table>
          <div className="mt-4 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3 text-xs text-zinc-400">
            <div className="font-medium text-zinc-200">
              Within-Source Cohort Breakdown (n = 170,812):
            </div>
            <ul className="mt-1 space-y-1 font-mono text-[11px] text-zinc-300">
              <li>
                • ESCO/ISCO (&lt;1M, n=84,588): MAE=0.0041 | RMSE=0.0114 |
                Acc=84.86%
              </li>
              <li>
                • JobHop (2B..3B, n=42,524): MAE=0.0036 | RMSE=0.0135 |
                Acc=77.29%
              </li>
              <li>
                • LinkedIn (&gt;=3B, n=43,700): MAE=0.0304 | RMSE=0.0354 |
                Acc=100.00%
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Section 4: Fairness & Security */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="print-card rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <Scale className="h-4 w-4 text-zinc-400" />
            <h3 className="print-text-dark text-sm font-semibold text-zinc-100">
              3. Subgroup Parity &amp; Fairness Audit (EEOC 80% Rule)
            </h3>
          </div>
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-400">
                <th className="py-1.5 font-medium">Subgroup</th>
                <th className="py-1.5 font-medium">Test Count</th>
                <th className="py-1.5 font-medium">Score MAE</th>
                <th className="py-1.5 font-medium">Macro F1</th>
                <th className="py-1.5 font-medium">Parity Ratio</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/70 font-mono text-zinc-300">
              <tr>
                <td className="py-1.5">Tenure: &lt;=5 yrs</td>
                <td>4,773</td>
                <td>0.1192</td>
                <td>0.7394</td>
                <td className="text-emerald-400">0.948 (PASS)</td>
              </tr>
              <tr>
                <td className="py-1.5">Tenure: 5–12 yrs</td>
                <td>7,373</td>
                <td>0.1103</td>
                <td>0.7730</td>
                <td className="text-emerald-400">1.000 (PASS)</td>
              </tr>
              <tr>
                <td className="py-1.5">Tenure: 12–20 yrs</td>
                <td>12,796</td>
                <td>0.0970</td>
                <td>0.8239</td>
                <td className="text-emerald-400">0.998 (PASS)</td>
              </tr>
              <tr>
                <td className="py-1.5">Seniority: Junior</td>
                <td>3,696</td>
                <td>0.1175</td>
                <td>0.7386</td>
                <td className="text-emerald-400">0.921 (PASS)</td>
              </tr>
              <tr>
                <td className="py-1.5">Seniority: Senior</td>
                <td>3,732</td>
                <td>0.1046</td>
                <td>0.8011</td>
                <td className="text-emerald-400">0.945 (PASS)</td>
              </tr>
              <tr>
                <td className="py-1.5">Sector: Software Engineering</td>
                <td>6,183</td>
                <td>0.0272</td>
                <td>0.9066</td>
                <td className="text-emerald-400">0.871 (PASS)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="print-card rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
          <div className="mb-3 flex items-center gap-2">
            <Lock className="h-4 w-4 text-zinc-400" />
            <h3 className="print-text-dark text-sm font-semibold text-zinc-100">
              4. Production Engineering &amp; Reliability Verification
            </h3>
          </div>
          <ul className="space-y-2.5 text-xs text-zinc-300">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              <span>
                <strong>Prometheus 0.0.4 Metrics (/metrics):</strong> Standard
                text exposition format tracking requests, error counters,
                uptime, and latency.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              <span>
                <strong>API Security &amp; Rate Limiting:</strong> Constant-time{" "}
                <code>hmac.compare_digest</code> API key authentication (HTTP
                401) and sliding-window per-IP rate limiting (HTTP 429).
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              <span>
                <strong>Sanitized Error Handling &amp; Strict CORS:</strong>{" "}
                Server-side traceback logging with correlation IDs and explicit
                origin allowlists.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              <span>
                <strong>Deterministic Containerization:</strong>{" "}
                <code>python:3.12-slim</code> non-root Docker runtime with
                pinned dependencies (<code>scikit-learn==1.7.2</code>,{" "}
                <code>torch==2.13.0</code>) and 11/11 automated checks passing.
              </span>
            </li>
          </ul>
        </div>
      </div>

      {/* Section 5: Interactive Q&A Accordion */}
      <div className="print-card rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6">
        <div className="mb-4 flex items-center gap-2.5">
          <HelpCircle className="h-4 w-4 text-zinc-400" />
          <h3 className="print-text-dark text-base font-semibold text-zinc-100">
            {isAr
              ? "5. الأسئلة التقنية والعلمية الشائعة (Technical Q&A)"
              : "5. Technical & Architectural Q&A"}
          </h3>
        </div>
        <div className="space-y-2.5">
          {QA_ITEMS.map((item, idx) => {
            const isOpen = openQa === idx;
            return (
              <div
                key={idx}
                className="overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-950/60"
              >
                <button
                  onClick={() => setOpenQa(isOpen ? null : idx)}
                  className="flex w-full items-center justify-between px-4 py-3.5 text-left text-xs font-semibold text-zinc-200 transition hover:text-white"
                >
                  <span>{isAr ? item.qAr : item.qEn}</span>
                  <motion.div
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronDown className="h-4 w-4 shrink-0 text-zinc-400" />
                  </motion.div>
                </button>
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.22 }}
                      className="border-t border-zinc-800/80 px-4 py-3 text-xs leading-relaxed text-zinc-400"
                    >
                      {isAr ? item.aAr : item.aEn}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}
