import { NextResponse } from "next/server";
import { verifyJobToken } from "@/lib/auth";
import { getJob, publicJob } from "@/lib/jobs";

export const dynamic = "force-dynamic";

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const token = new URL(request.url).searchParams.get("token") ?? "";
  const job = await getJob(id);
  if (!job) return NextResponse.json({ error: "Job not found or expired." }, { status: 404 });
  if (!verifyJobToken(token, job.tokenHash)) {
    return NextResponse.json({ error: "Invalid job token." }, { status: 401 });
  }
  return NextResponse.json(publicJob(job), {
    headers: { "cache-control": "no-store, max-age=0" },
  });
}
