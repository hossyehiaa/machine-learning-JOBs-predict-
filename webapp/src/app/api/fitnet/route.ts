import { NextRequest, NextResponse } from "next/server";
import { computeFitNetMatch, FitNetRequest } from "@/lib/career-engine";

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as FitNetRequest;
    const fastApiUrl = process.env.FITNET_API_URL || "http://127.0.0.1:8000";
    const apiKey = process.env.FITNET_API_KEY || "";

    if (fastApiUrl) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 1400);
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (apiKey) {
          headers["X-API-Key"] = apiKey;
        }
        const resp = await fetch(`${fastApiUrl.replace(/\/$/, "")}/v1/predict-fit`, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        clearTimeout(timer);
        if (resp.ok) {
          const data = await resp.json();
          return NextResponse.json({ ...data, engine_mode: "fastapi_production" });
        }
      } catch {
        // Fallback transparently to deterministic Serverless Edge Calibrated Engine
      }
    }

    const result = computeFitNetMatch(body);
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      {
        detail: "Invalid FitNet request payload",
        error: err instanceof Error ? err.message : "Unknown error",
      },
      { status: 422 }
    );
  }
}
