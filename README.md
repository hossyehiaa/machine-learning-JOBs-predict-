# Career Intelligence Studio — Full-Stack Monorepo

End-to-end AI Career Guidance platform combining:
1. **`webapp/`** — Next.js 15 (App Router, TypeScript, Tailwind CSS, Framer Motion) user-facing studio with PDF/TXT/JSON resume parsing, bilingual English/Arabic (RTL/LTR) UI, and a printable technical report.
2. **`model1-fitnet/`** — FastAPI microservice for **Model 1 (Resume-to-Job Matching)** trained on `221,903` records (`37,044` unique candidate resumes, `84,508` job titles) using PyTorch + HistGradientBoosting + Isotonic Calibration (`82.73%` test accuracy).
3. **`model2-pathnet/`** — FastAPI microservice for **Model 2 (Career Trajectory + RAG Roadmap Retrieval)** trained on `1,140,000` records (`801,603` individuals) + `230,333`-row TF-IDF RAG Knowledge Base (`86.85%` test accuracy, `0.01%` calibrated ECE).
4. **`deploy/`** — Ready-to-deploy infrastructure templates (`render.yaml`, `fly.model1.toml`, `fly.model2.toml`, `docker-compose.yml`).

---

## Repository Structure

```text
career-ai-monorepo/
├── webapp/                  # Next.js 15 Frontend (Deploy to Vercel)
├── model1-fitnet/           # Model 1 FastAPI Service (:8000) + Dockerfile
├── model2-pathnet/          # Model 2 FastAPI Service + 230k RAG (:8001) + Dockerfile
├── deploy/                  # Render, Fly.io, and Docker Compose deployment configs
├── .env.example             # Environment variable template (no secrets)
├── .gitignore
└── README.md
```

---

## Environment Variables

| Variable | Where Used | Default (Local) | Purpose |
| :--- | :--- | :--- | :--- |
| `FITNET_API_URL` | Vercel / `webapp` | `http://127.0.0.1:8000` | Base URL of deployed Model 1 FastAPI service |
| `PATHNET_API_URL` | Vercel / `webapp` | `http://127.0.0.1:8001` | Base URL of deployed Model 2 + RAG FastAPI service |
| `FITNET_API_KEY` | Vercel + Model 1 | *(empty)* | Optional `X-API-Key` shared secret between Next.js proxy and Model 1 |
| `PATHNET_API_KEY` | Vercel + Model 2 | *(empty)* | Optional `X-API-Key` shared secret between Next.js proxy and Model 2 |
| `ALLOWED_ORIGINS` | Model 1 & Model 2 | `http://localhost:3000` | Comma-separated production CORS origins (e.g. `https://<app>.vercel.app`) |
| `ALLOWED_ORIGIN_REGEX` | Model 1 & Model 2 | `^https://.*\.vercel\.app$` | Regex allowing Vercel preview URLs without wildcard `*` |

---

## Local Verification & Clean Build

```bash
# 1. Verify Next.js Webapp clean build
cd webapp && npm install && npm run build

# 2. Verify Model 1 FastAPI import
cd ../model1-fitnet && python3 -c "import app_fitnet; print('Model 1 OK')"

# 3. Verify Model 2 + RAG FastAPI import
cd ../model2-pathnet && python3 -c "import app; print('Model 2 OK')"
```
