# LumiCycle Remote Demo

This is the public web interface for LumiCycle. Vercel serves the page, accepts image uploads,
and holds a short-lived job queue. The model weights never leave the project PC: the Python
worker polls the queue over outbound HTTPS, runs inference on the RTX 4070 Super, and uploads the
result.

## Required Vercel resources

1. Link this directory to a Vercel project.
2. Add a Vercel Blob store. Production uses Vercel OIDC; local development may use a read-write
   token.
3. Add an Upstash Redis integration, which supplies `UPSTASH_REDIS_REST_URL` and
   `UPSTASH_REDIS_REST_TOKEN` (or the equivalent `KV_REST_API_*` variables).
4. Set `WORKER_SECRET` to a long random value.
5. Set `DEMO_ACCESS_CODE` to the short code the team will type during the showcase.

## Local web development

Copy `.env.example` to `.env.local`, fill the values, then run:

```powershell
npm install
npm run dev
```

## Start the home GPU worker

From the repository root, with the Python virtual environment active, use the prepared launcher:

```powershell
.\scripts\start_remote_demo_worker.ps1
```

The equivalent manual command is:

```powershell
$env:LUMICYCLE_REMOTE_URL = "https://your-project.vercel.app"
$env:LUMICYCLE_WORKER_SECRET = "the-same-long-worker-secret"
python -m daynight.remote_worker
```

Leave that terminal open and prevent the PC from sleeping. The website's top-right indicator
turns green within a few seconds. No router port-forwarding or tunnel is needed.

Uploaded inputs, outputs, and Redis job state are intended for a private college demonstration.
Job metadata expires after 24 hours; periodically remove old images from the Blob dashboard.
