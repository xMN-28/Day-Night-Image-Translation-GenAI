import { createHash, randomBytes, timingSafeEqual } from "node:crypto";

function equalSecret(provided: string, expected: string) {
  const left = Buffer.from(provided);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

export function verifyAccessCode(provided: string | null | undefined) {
  const expected = process.env.DEMO_ACCESS_CODE;
  if (!expected) return true;
  return Boolean(provided && equalSecret(provided, expected));
}

export function verifyWorker(request: Request) {
  const expected = process.env.WORKER_SECRET;
  const authorization = request.headers.get("authorization") ?? "";
  const provided = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length)
    : request.headers.get("x-worker-secret") ?? "";
  return Boolean(expected && equalSecret(provided, expected));
}

export function newClientToken() {
  return randomBytes(24).toString("base64url");
}

export function hashToken(token: string) {
  return createHash("sha256").update(token).digest("hex");
}

export function verifyJobToken(provided: string, tokenHash: string) {
  return equalSecret(hashToken(provided), tokenHash);
}
