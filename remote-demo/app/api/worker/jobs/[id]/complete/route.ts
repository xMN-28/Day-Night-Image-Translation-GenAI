import { put } from "@vercel/blob";
import { NextResponse } from "next/server";
import { verifyWorker } from "@/lib/auth";
import { getJob, saveJob } from "@/lib/jobs";
import { getRedis, PROCESSING_KEY } from "@/lib/redis";

export const runtime = "nodejs";

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  if (!verifyWorker(request)) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  const { id } = await context.params;
  const job = await getJob(id);
  if (!job) return NextResponse.json({ error: "Job not found or expired." }, { status: 404 });
  try {
    const form = await request.formData();
    const output = form.get("output");
    if (!(output instanceof File) || !output.type.startsWith("image/")) {
      return NextResponse.json({ error: "Missing output image." }, { status: 400 });
    }
    if (output.size > 4 * 1024 * 1024) {
      return NextResponse.json({ error: "Output image is larger than 4 MB." }, { status: 413 });
    }
    const metadataText = String(form.get("metadata") ?? "{}");
    const metadata = JSON.parse(metadataText) as Record<string, unknown>;
    const blob = await put(`outputs/${id}.jpg`, output, {
      access: "public",
      addRandomSuffix: false,
      contentType: "image/jpeg",
    });
    job.status = "complete";
    job.outputUrl = blob.url;
    job.outputPathname = blob.pathname;
    job.metadata = metadata;
    job.error = undefined;
    await saveJob(job);
    await getRedis().srem(PROCESSING_KEY, id);
    return NextResponse.json({ ok: true, outputUrl: blob.url });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not store output." },
      { status: 500 },
    );
  }
}
