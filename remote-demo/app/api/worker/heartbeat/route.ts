import { NextResponse } from "next/server";
import { verifyWorker } from "@/lib/auth";
import { getRedis, HEARTBEAT_KEY } from "@/lib/redis";
import type { WorkerHeartbeat } from "@/lib/types";

export async function POST(request: Request) {
  if (!verifyWorker(request)) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  const body = await request.json();
  const heartbeat: WorkerHeartbeat = {
    workerName: String(body.workerName ?? "home-gpu"),
    device: String(body.device ?? "unknown"),
    activeJob: body.activeJob ? String(body.activeJob) : null,
    availableModels: Array.isArray(body.availableModels)
      ? body.availableModels.map(String).slice(0, 20)
      : [],
    timestamp: Date.now(),
  };
  await getRedis().set(HEARTBEAT_KEY, heartbeat, { ex: 25 });
  return NextResponse.json({ ok: true });
}
