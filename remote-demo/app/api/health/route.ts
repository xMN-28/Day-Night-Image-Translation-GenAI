import { NextResponse } from "next/server";
import { getRedis, HEARTBEAT_KEY } from "@/lib/redis";
import type { WorkerHeartbeat } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const heartbeat = await getRedis().get<WorkerHeartbeat>(HEARTBEAT_KEY);
    const online = Boolean(heartbeat && Date.now() - heartbeat.timestamp < 20_000);
    return NextResponse.json(
      online ? { online: true, ...heartbeat } : { online: false },
      { headers: { "cache-control": "no-store, max-age=0" } },
    );
  } catch {
    return NextResponse.json({ online: false }, { headers: { "cache-control": "no-store" } });
  }
}
