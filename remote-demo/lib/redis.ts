import { Redis } from "@upstash/redis";

let redis: Redis | null = null;

export function getRedis() {
  if (!redis) {
    const url = process.env.UPSTASH_REDIS_REST_URL ?? process.env.KV_REST_API_URL;
    const token = process.env.UPSTASH_REDIS_REST_TOKEN ?? process.env.KV_REST_API_TOKEN;
    if (!url || !token) {
      throw new Error("Redis is not configured on this deployment.");
    }
    redis = new Redis({ url, token });
  }
  return redis;
}

export const QUEUE_KEY = "lumicycle:jobs:queued";
export const PROCESSING_KEY = "lumicycle:jobs:processing";
export const HEARTBEAT_KEY = "lumicycle:worker:heartbeat";
export const JOB_TTL_SECONDS = 60 * 60 * 24;

export function jobKey(id: string) {
  return `lumicycle:job:${id}`;
}
