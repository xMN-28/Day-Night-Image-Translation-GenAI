import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import { hashToken, newClientToken, verifyAccessCode } from "@/lib/auth";
import { publicJob, saveJob } from "@/lib/jobs";
import { MODEL_IDS, supportsDirection, type Direction } from "@/lib/models";
import { getRedis, QUEUE_KEY } from "@/lib/redis";
import type { StoredJob } from "@/lib/types";

function isBlobUrl(value: string) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname.endsWith(".blob.vercel-storage.com");
  } catch {
    return false;
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    if (!verifyAccessCode(body.accessCode)) {
      return NextResponse.json({ error: "Incorrect demo access code." }, { status: 401 });
    }
    const modelId = String(body.modelId ?? "");
    const direction = String(body.direction ?? "") as Direction;
    if (!MODEL_IDS.has(modelId) || !supportsDirection(modelId, direction)) {
      return NextResponse.json({ error: "Unsupported model or direction." }, { status: 400 });
    }
    const inputUrl = String(body.inputUrl ?? "");
    const inputPathname = String(body.inputPathname ?? "");
    if (!isBlobUrl(inputUrl) || !inputPathname.startsWith("inputs/")) {
      return NextResponse.json({ error: "Invalid uploaded image URL." }, { status: 400 });
    }
    const now = Date.now();
    const id = randomUUID();
    const token = newClientToken();
    const job: StoredJob = {
      id,
      status: "queued",
      tokenHash: hashToken(token),
      modelId,
      direction,
      maximumEdge: Math.max(256, Math.min(1024, Number(body.maximumEdge) || 768)),
      inputUrl,
      inputPathname,
      inputName: String(body.inputName ?? "image"),
      inputWidth: Number(body.inputWidth) || undefined,
      inputHeight: Number(body.inputHeight) || undefined,
      createdAt: now,
      updatedAt: now,
    };
    await saveJob(job);
    await getRedis().lpush(QUEUE_KEY, id);
    return NextResponse.json({ job: publicJob(job), token }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not create job." },
      { status: 500 },
    );
  }
}
