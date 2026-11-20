import { NextResponse } from "next/server";

export async function GET() {
  const fitnetUrl = process.env.FITNET_API_URL || "http://127.0.0.1:8000";
  const pathnetUrl = process.env.PATHNET_API_URL || "http://127.0.0.1:8001";
  let fitnetOnline = false;
  let pathnetOnline = false;

  try {
    const c1 = new AbortController();
    const t1 = setTimeout(() => c1.abort(), 900);
    const r1 = await fetch(`${fitnetUrl.replace(/\/$/, "")}/health`, {
      signal: c1.signal,
    });
    clearTimeout(t1);
    fitnetOnline = r1.ok;
  } catch {
    fitnetOnline = false;
  }

  try {
    const c2 = new AbortController();
    const t2 = setTimeout(() => c2.abort(), 900);
    const r2 = await fetch(`${pathnetUrl.replace(/\/$/, "")}/health`, {
      signal: c2.signal,
    });
    clearTimeout(t2);
    pathnetOnline = r2.ok;
  } catch {
    pathnetOnline = false;
  }

  const fastApiConnected = fitnetOnline || pathnetOnline;

  return NextResponse.json({
    status: "healthy",
    engine_mode: fastApiConnected ? "fastapi_production" : "serverless_edge",
    services: {
      job_match_fastapi_8000: fitnetOnline,
      trajectory_and_rag_fastapi_8001: pathnetOnline,
    },
    models: {
      job_match: {
        records: 221903,
        unique_resumes: 37044,
        unique_jobs: 84508,
        held_out_test_accuracy: "82.73%",
        macro_f1: 0.8259,
      },
      career_trajectory: {
        records: 1140000,
        unique_persons: 801603,
        rag_roadmaps: 230333,
        held_out_test_accuracy: "86.85%",
        calibrated_ece: "0.01%",
      },
    },
  });
}
