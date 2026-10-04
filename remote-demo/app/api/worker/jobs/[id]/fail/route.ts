import { NextResponse } from "next/server";
import { verifyWorker } from "@/lib/auth";
import { getJob, saveJob } from "@/lib/jobs";
import { getRedis, PROCESSING_KEY } from "@/lib/redis";

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  if (!verifyWorker(request)) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  const { id } = await context.params;
  const job = await getJob(id);
  if (!job) return NextResponse.json({ error: "Job not found or expired." }, { status: 404 });
  const body = await request.json();
  job.status = "failed";
  job.error = String(body.error ?? "The GPU worker could not complete this job.").slice(0, 600);
  await saveJob(job);
  await getRedis().srem(PROCESSING_KEY, id);
  return NextResponse.json({ ok: true });
}
