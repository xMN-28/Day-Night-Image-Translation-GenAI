"use client";

import { upload } from "@vercel/blob/client";
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import { MODEL_OPTIONS, type Direction } from "@/lib/models";

type PublicJob = {
  id: string;
  status: "queued" | "processing" | "complete" | "failed";
  modelId: string;
  direction: Direction;
  inputUrl: string;
  outputUrl?: string;
  metadata?: Record<string, unknown>;
  error?: string;
};

type Health = {
  online: boolean;
  workerName?: string;
  device?: string;
  activeJob?: string | null;
};

const POLL_MS = 1200;

export function Demo({ accessCodeRequired }: { accessCodeRequired: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [direction, setDirection] = useState<Direction>("day_to_night");
  const [modelId, setModelId] = useState("v2_best");
  const [maximumEdge, setMaximumEdge] = useState(768);
  const [accessCode, setAccessCode] = useState("");
  const [job, setJob] = useState<PublicJob | null>(null);
  const [jobToken, setJobToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState<Health>({ online: false });
  const [compare, setCompare] = useState(50);

  const selectedModel = useMemo(
    () => MODEL_OPTIONS.find((model) => model.id === modelId) ?? MODEL_OPTIONS[0],
    [modelId],
  );

  useEffect(() => {
    let live = true;
    async function refreshHealth() {
      try {
        const response = await fetch("/api/health", { cache: "no-store" });
        if (live && response.ok) setHealth(await response.json());
      } catch {
        if (live) setHealth({ online: false });
      }
    }
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 5000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!job || !jobToken || !["queued", "processing"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/jobs/${job.id}?token=${encodeURIComponent(jobToken)}`, {
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Could not read the job status.");
        const updated: PublicJob = await response.json();
        setJob(updated);
        if (updated.status === "complete" || updated.status === "failed") setBusy(false);
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Status polling failed.");
      }
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [job, jobToken]);

  function acceptFile(nextFile: File | null) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (nextFile.size > 12 * 1024 * 1024) {
      setError("The image must be smaller than 12 MB.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setJob(null);
    setError("");
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    acceptFile(event.dataTransfer.files[0] ?? null);
  }

  async function translate() {
    if (!file) return setError("Upload an image first.");
    if (!health.online) return setError("The home GPU worker is offline. Start it before translating.");
    if (accessCodeRequired && !accessCode) return setError("Enter the demo access code.");
    setBusy(true);
    setError("");
    setJob(null);
    try {
      const blob = await upload(`inputs/${crypto.randomUUID()}-${file.name}`, file, {
        access: "public",
        handleUploadUrl: "/api/blob/upload",
        multipart: true,
        clientPayload: JSON.stringify({ accessCode }),
      });
      const image = new Image();
      image.src = preview ?? "";
      await image.decode();
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          accessCode,
          modelId,
          direction,
          maximumEdge,
          inputUrl: blob.url,
          inputPathname: blob.pathname,
          inputName: file.name,
          inputWidth: image.naturalWidth,
          inputHeight: image.naturalHeight,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Could not create the translation job.");
      setJob(payload.job);
      setJobToken(payload.token);
    } catch (submitError) {
      setBusy(false);
      setError(submitError instanceof Error ? submitError.message : "Upload failed.");
    }
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setJob(null);
    setJobToken("");
    setBusy(false);
    setError("");
    setCompare(50);
    if (inputRef.current) inputRef.current.value = "";
  }

  const statusText = job
    ? job.status === "queued"
      ? "Queued — waiting for the home GPU"
      : job.status === "processing"
        ? "Processing on RTX 4070 Super"
        : job.status === "complete"
          ? "Translation complete"
          : "Translation failed"
    : "Upload an image and choose a model.";

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">REMOTE GPU SHOWCASE</p>
          <h1>LumiCycle</h1>
          <p>Structure-aware day ↔ night translation, rendered on our home RTX 4070 Super.</p>
        </div>
        <div className={`worker-pill ${health.online ? "online" : "offline"}`}>
          <span aria-hidden="true" />
          {health.online ? `GPU online · ${health.device ?? "ready"}` : "GPU worker offline"}
        </div>
      </header>

      <section className="workspace">
        <div className="left-column">
          <div className="panel image-panel">
            <div className="panel-label">Input photograph</div>
            <div
              className={`drop-zone ${preview ? "has-image" : ""}`}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event: ChangeEvent<HTMLInputElement>) => acceptFile(event.target.files?.[0] ?? null)}
                hidden
              />
              {preview ? (
                // A local blob URL cannot use next/image.
                // eslint-disable-next-line @next/next/no-img-element
                <img src={preview} alt="Uploaded photograph" />
              ) : (
                <div className="upload-copy">
                  <span className="upload-icon">↥</span>
                  <strong>Drop an image here</strong>
                  <small>or tap to browse · JPEG, PNG, WebP · max 12 MB</small>
                </div>
              )}
            </div>
          </div>

          <div className="panel controls-grid">
            <fieldset>
              <legend>Direction</legend>
              <div className="segmented">
                <button
                  type="button"
                  className={direction === "day_to_night" ? "active" : ""}
                  onClick={() => setDirection("day_to_night")}
                >
                  Day → Night
                </button>
                <button
                  type="button"
                  disabled={!selectedModel.directions.includes("night_to_day")}
                  className={direction === "night_to_day" ? "active" : ""}
                  onClick={() => setDirection("night_to_day")}
                >
                  Night → Day
                </button>
              </div>
            </fieldset>
            <label>
              <span>Model</span>
              <select
                value={modelId}
                onChange={(event) => {
                  const nextModel = MODEL_OPTIONS.find((model) => model.id === event.target.value);
                  setModelId(event.target.value);
                  if (nextModel && !nextModel.directions.includes(direction)) {
                    setDirection(nextModel.directions[0]);
                  }
                }}
              >
                {MODEL_OPTIONS.map((model) => (
                  <option key={model.id} value={model.id}>{model.label}</option>
                ))}
              </select>
              <small>{selectedModel.note}</small>
            </label>
          </div>

          <div className="panel slider-panel">
            <label htmlFor="edge">Maximum inference edge</label>
            <output>{maximumEdge}px</output>
            <input
              id="edge"
              type="range"
              min="256"
              max="1024"
              step="64"
              value={maximumEdge}
              onChange={(event) => setMaximumEdge(Number(event.target.value))}
            />
          </div>

          {accessCodeRequired && (
            <label className="panel access-code">
              <span>Demo access code</span>
              <input
                type="password"
                autoComplete="off"
                value={accessCode}
                onChange={(event) => setAccessCode(event.target.value)}
                placeholder="Ask the project team"
              />
            </label>
          )}

          <div className="actions">
            <button className="primary" type="button" onClick={translate} disabled={busy || !file}>
              {busy ? "Working…" : "Translate"}
            </button>
            <button className="secondary" type="button" onClick={reset}>Reset</button>
          </div>
        </div>

        <div className="right-column">
          <div className="panel image-panel">
            <div className="panel-label">Translated output</div>
            <div className="output-stage">
              {job?.outputUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={job.outputUrl} alt="Night translation output" />
              ) : (
                <div className="output-empty"><span>☾</span><p>The generated image will appear here.</p></div>
              )}
            </div>
          </div>

          <div className="panel compare-panel">
            <div className="panel-label">Before / after</div>
            {preview && job?.outputUrl ? (
              <div className="comparison" style={{ "--split": `${compare}%` } as React.CSSProperties}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img className="after" src={job.outputUrl} alt="Translated result" />
                <div className="before-wrap">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img className="before" src={preview} alt="Original input" />
                </div>
                <div className="divider" aria-hidden="true"><span>↔</span></div>
                <input
                  aria-label="Compare original and translated image"
                  type="range"
                  min="0"
                  max="100"
                  value={compare}
                  onChange={(event) => setCompare(Number(event.target.value))}
                />
              </div>
            ) : (
              <div className="compare-empty">Comparison appears after translation.</div>
            )}
          </div>

          <div className={`status-card ${job?.status ?? "idle"}`}>
            <div className="status-heading">
              <span className="status-dot" />
              <strong>{statusText}</strong>
            </div>
            {error && <p className="error">{error}</p>}
            {job?.metadata && <pre>{JSON.stringify(job.metadata, null, 2)}</pre>}
          </div>
        </div>
      </section>

      <footer>
        <strong>Academic note:</strong> Every selectable checkpoint was trained locally by the team.
        V2.1 and LumiRender remain visible as documented failed experiments, not claimed improvements.
      </footer>
    </main>
  );
}
