// Authenticated end-to-end API smoke test. Never logs credentials or job tokens.
// Run inside remote-demo: node scripts/smoke-inference.mjs <image> <model> <direction>
import nextEnv from "@next/env";
import { put } from "@vercel/blob";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { randomUUID } from "node:crypto";

nextEnv.loadEnvConfig(process.cwd());
const [imagePath, modelId = "night_day_experimental", direction = "night_to_day"] = process.argv.slice(2);
if (!imagePath) throw new Error("Provide a JPEG/PNG/WebP sample path");
const base = process.env.LUMICYCLE_REMOTE_URL ?? "https://lumicycle-remote-demo.vercel.app";
const input = await readFile(resolve(imagePath));
const contentType = imagePath.endsWith(".jpg") ? "image/jpeg" : "image/png";
const blob = await put(`inputs/smoke-${randomUUID()}${contentType === "image/jpeg" ? ".jpg" : ".png"}`, input,
  {access: "public", contentType});
const response = await fetch(`${base}/api/jobs`, {method: "POST",
  headers: {"content-type": "application/json"},
  body: JSON.stringify({accessCode: process.env.DEMO_ACCESS_CODE,
    modelId, direction, maximumEdge: 512, inputUrl: blob.url,
    inputPathname: blob.pathname, inputName: "smoke-test"})});
const created = await response.json();
if (!response.ok) throw new Error(`Job creation ${response.status}: ${created.error}`);
let completed;
for (let attempt = 0; attempt < 40; attempt++) {
  await new Promise(resolve => setTimeout(resolve, 1500));
  const poll = await fetch(`${base}/api/jobs/${created.job.id}?token=${encodeURIComponent(created.token)}`);
  const job = await poll.json();
  if (!poll.ok) throw new Error(`Job polling failed (${poll.status})`);
  if (job.status === "failed") throw new Error(job.error);
  if (job.status === "complete") { completed = job; break; }
}
if (!completed) throw new Error("Timed out waiting for home GPU");
const generated = await fetch(completed.outputUrl);
if (!generated.ok || !generated.headers.get("content-type")?.startsWith("image/")) {
  throw new Error("Invalid output download");
}
const folder = resolve("../artifacts/remote_worker");
await mkdir(folder, {recursive: true});
await writeFile(`${folder}/${modelId}-remote.jpg`, Buffer.from(await generated.arrayBuffer()));
const evidence = {modelId, direction, status: completed.status, metadata: completed.metadata,
  verification: "Blob SDK upload → public job API → home GPU worker → result polling/download"};
await writeFile(`${folder}/${modelId}-e2e.json`, JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));
