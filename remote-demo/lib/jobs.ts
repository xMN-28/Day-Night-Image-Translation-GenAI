import type { StoredJob } from "@/lib/types";
import { getRedis, JOB_TTL_SECONDS, jobKey } from "@/lib/redis";

export async function getJob(id: string) {
  return getRedis().get<StoredJob>(jobKey(id));
}

export async function saveJob(job: StoredJob) {
  job.updatedAt = Date.now();
  await getRedis().set(jobKey(job.id), job, { ex: JOB_TTL_SECONDS });
}

export function publicJob(job: StoredJob) {
  const safe = { ...job } as Partial<StoredJob>;
  delete safe.tokenHash;
  delete safe.inputPathname;
  delete safe.outputPathname;
  return safe;
}
