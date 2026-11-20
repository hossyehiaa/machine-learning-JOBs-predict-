import { NextRequest, NextResponse } from "next/server";
import zlib from "zlib";
import { parseResumeText } from "@/lib/career-engine";

function extractTextFromPdfBuffer(buf: Buffer): string {
  const rawLatin1 = buf.toString("latin1");
  const chunks: string[] = [];

  // 1. Decompress FlateDecode streams where possible and scan for PDF text operators
  const streamRegex = /stream[\r\n]+([\s\S]*?)[\r\n]+endstream/g;
  let m: RegExpExecArray | null;
  while ((m = streamRegex.exec(rawLatin1)) !== null) {
    const streamBytes = Buffer.from(m[1], "latin1");
    let decoded = "";
    try {
      decoded = zlib.inflateSync(streamBytes).toString("latin1");
    } catch {
      decoded = m[1];
    }
    const parenMatches = decoded.match(/\(([^()\\]|\\.)*\)/g);
    if (parenMatches) {
      for (const pm of parenMatches) {
        const inner = pm
          .slice(1, -1)
          .replace(/\\n/g, "\n")
          .replace(/\\r/g, " ")
          .replace(/\\\(/g, "(")
          .replace(/\\\)/g, ")")
          .replace(/[^\x20-\x7E\n]/g, " ");
        if (inner.trim().length > 1) {
          chunks.push(inner.trim());
        }
      }
    }
  }

  // 2. Fallback: extract readable ASCII sequences if stream parsing yielded minimal text
  const combined = chunks.join(" ");
  if (combined.length >= 40) {
    return combined;
  }
  const asciiRuns = rawLatin1.match(/[A-Za-z0-9+#./,\-_@ ]{4,}/g) || [];
  return asciiRuns
    .filter(
      (s) =>
        !/^(obj|endobj|stream|endstream|xref|trailer|startxref|\/Type|\/Font|\/Page)/.test(
          s.trim()
        )
    )
    .join(" ");
}

export async function POST(req: NextRequest) {
  try {
    const contentType = req.headers.get("content-type") || "";
    let extractedText = "";

    if (contentType.includes("multipart/form-data")) {
      const formData = await req.formData();
      const file = formData.get("file") as File | null;
      const rawText = formData.get("text") as string | null;

      if (file) {
        const arrayBuf = await file.arrayBuffer();
        const buffer = Buffer.from(arrayBuf);
        const fileName = (file.name || "").toLowerCase();
        if (fileName.endsWith(".pdf") || buffer.subarray(0, 4).toString() === "%PDF") {
          extractedText = extractTextFromPdfBuffer(buffer);
        } else {
          extractedText = buffer.toString("utf-8");
        }
      } else if (rawText) {
        extractedText = rawText;
      }
    } else {
      const body = await req.json();
      extractedText = body.text || body.resume_text || "";
    }

    const parsed = parseResumeText(extractedText);
    return NextResponse.json({
      ...parsed,
      extracted_text_preview: extractedText.slice(0, 1200),
    });
  } catch (err) {
    return NextResponse.json(
      {
        error: "Failed to parse resume content",
        details: err instanceof Error ? err.message : "Unknown error",
      },
      { status: 400 }
    );
  }
}
