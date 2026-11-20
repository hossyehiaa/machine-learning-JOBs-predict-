# CareerAI Studio — FitNet (Model 1) & PathNet0 (Model 2) Web Platform

A complete, production-ready, Vercel-deployable **Next.js 15 (App Router, TypeScript, Tailwind CSS, Lucide Icons)** web application uniting:
1. **FitNet Studio (`Model 1` — Resume-to-Job Matching, `221,903` records)**: Built-in PDF/TXT/JSON resume parser & 588-skill extractor, calibrated Fit Score gauge, 3-class Match Tier (`Low` / `Medium` / `High`), matched vs. missing skill gap audit, and 3-phase upskilling plan.
2. **PathNet0 Trajectory (`Model 2` — Future-Self Career Modeling, `1,140,000` records & `230,333` RAG roadmaps)**: Calibrated transition feasibility probability, transition category (`Vertical_Promotion`, `Adjacent_Pivot`, `Career_Shift`), seniority progression delta, and 3-phase retrieved RAG skill curriculum.
3. **Next-Step Recommender**: Ranks Top-N candidate next career steps descending by calibrated transition probability.
4. **Interactive & Printable Graduation Defense Report**: Full provenance across `1,361,903` combined records, `GroupShuffleSplit` leakage prevention, `110-D / 114-D` hybrid multi-task architecture, held-out test benchmarks, EEOC 80% subgroup fairness audits, MLOps security specs, and interactive thesis committee Q&A.

---

## Quickstart & Vercel Deployment

```bash
# 1. Install dependencies
npm install

# 2. Run local development server on http://localhost:3000
npm run dev

# 3. Verify production build
npm run build
npm run start

# 4. Deploy to Vercel (works 100% standalone via Serverless Edge Calibrated Engine)
npx vercel --prod
```
