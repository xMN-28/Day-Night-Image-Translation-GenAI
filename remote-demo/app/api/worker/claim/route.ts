import { NextResponse } from "next/server";
import { verifyWorker } from "@/lib/auth";
import { getJob, saveJob } from "@/lib/jobs";
import { getRedis, PROCESSING_KEY, QUEUE_KEY } from "@/lib/redis";

export async function POST(request: Request) {
  if (!verifyWorker(request)) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  const redis = getRedis();
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const id = await redis.rpop<string>(QUEUE_KEY);
    if (!id) return new Response(null, { status: 204 });
    const job = await getJob(id);
    if (!job || job.status !== "queued") continue;
    job.status = "processing";
    job.claimedAt = Date.now();
    await saveJob(job);
    await redis.sadd(PROCESSING_KEY, id);
    const workerJob = { ...job } as Partial<typeof job>;
    delete workerJob.tokenHash;
    return NextResponse.json({ job: workerJob });
  }
  return new Response(null, { status: 204 });
}
