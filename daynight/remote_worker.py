from __future__ import annotations

import argparse
import io
import json
import os
import signal
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import torch
from PIL import Image

from .inference import ModelManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RemoteModel:
    label: str
    engine_name: str
    checkpoint: Path
    directions: tuple[str, ...] = ("day_to_night", "night_to_day")


MODEL_CATALOG = {
    "v2_best": RemoteModel(
        "LumiCycle V2 — best (4.5k)",
        "LumiCycle V2",
        PROJECT_ROOT / "runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt",
    ),
    "v2_final": RemoteModel(
        "LumiCycle V2 — final (12k)",
        "LumiCycle V2",
        PROJECT_ROOT / "runs/lumicycle_v2_bdd100k/checkpoints/step_00012000.pt",
    ),
    "v1_best": RemoteModel(
        "LumiCycle V1 — best (13k)",
        "LumiCycle",
        PROJECT_ROOT / "runs/lumicycle_bdd100k/checkpoints/step_00013000.pt",
    ),
    "v1_final": RemoteModel(
        "LumiCycle V1 — final (40k)",
        "LumiCycle",
        PROJECT_ROOT / "runs/lumicycle_bdd100k/checkpoints/step_00040000.pt",
    ),
    "v2_1_best": RemoteModel(
        "V2.1 ablation — best (5.5k)",
        "LumiCycle V2",
        PROJECT_ROOT / "runs/lumicycle_v2_1_bdd100k/checkpoints/step_00005500.pt",
    ),
    "v2_1_final": RemoteModel(
        "V2.1 ablation — final (6k)",
        "LumiCycle V2",
        PROJECT_ROOT / "runs/lumicycle_v2_1_bdd100k/checkpoints/step_00006000.pt",
    ),
    "lumirender_final": RemoteModel(
        "LumiRender — final (34k)",
        "LumiRender",
        PROJECT_ROOT
        / "runs/lumirender_physics_bdd100k/checkpoints/step_00034004.pt",
        ("day_to_night",),
    ),
}


def available_model_ids() -> list[str]:
    return [model_id for model_id, spec in MODEL_CATALOG.items() if spec.checkpoint.is_file()]


class RemoteWorker:
    def __init__(
        self,
        server: str,
        secret: str,
        poll_seconds: float = 1.0,
        worker_name: str | None = None,
    ) -> None:
        self.server = server.rstrip("/")
        parsed = urlparse(self.server)
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("The remote URL must use HTTPS (except localhost development).")
        if not secret:
            raise ValueError("LUMICYCLE_WORKER_SECRET is required.")
        self.poll_seconds = max(0.5, poll_seconds)
        self.worker_name = worker_name or f"{socket.gethostname()}-gpu"
        self.session = requests.Session()
        self.session.headers.update({"authorization": f"Bearer {secret}"})
        self.manager = ModelManager()
        self.stop_event = threading.Event()
        self.active_job: str | None = None
        self._state_lock = threading.Lock()

    @property
    def device_label(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    def _post(self, path: str, **kwargs: Any) -> requests.Response:
        return self.session.post(f"{self.server}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)

    def heartbeat(self) -> None:
        with self._state_lock:
            active_job = self.active_job
        response = self._post(
            "/api/worker/heartbeat",
            json={
                "workerName": self.worker_name,
                "device": self.device_label,
                "activeJob": active_job,
                "availableModels": available_model_ids(),
            },
        )
        response.raise_for_status()

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.heartbeat()
            except requests.RequestException as error:
                print(f"[worker] heartbeat failed: {error}", flush=True)
            self.stop_event.wait(5)

    def claim(self) -> dict[str, Any] | None:
        response = self._post("/api/worker/claim")
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()["job"]

    def fail(self, job_id: str, error: str) -> None:
        try:
            response = self._post(
                f"/api/worker/jobs/{job_id}/fail",
                json={"error": error[:600]},
            )
            response.raise_for_status()
        except requests.RequestException as report_error:
            print(f"[worker] could not report failure for {job_id}: {report_error}", flush=True)

    def process(self, job: dict[str, Any]) -> None:
        job_id = str(job["id"])
        model_id = str(job["modelId"])
        spec = MODEL_CATALOG.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown model id: {model_id}")
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint is missing: {spec.checkpoint}")
        direction = str(job["direction"])
        if direction not in spec.directions:
            raise ValueError(f"{spec.label} does not support {direction}.")

        download = requests.get(str(job["inputUrl"]), timeout=45)
        download.raise_for_status()
        if len(download.content) > 12 * 1024 * 1024:
            raise ValueError("Input image exceeds the 12 MB worker limit.")
        image = Image.open(io.BytesIO(download.content))
        image.load()

        print(f"[worker] {job_id[:8]} · {spec.label} · {direction}", flush=True)
        output, metadata = self.manager.translate_custom(
            image=image,
            direction=direction,
            model_name=spec.engine_name,
            maximum_edge=int(job.get("maximumEdge", 768)),
            checkpoint_override=spec.checkpoint,
            display_name=spec.label,
        )
        metadata["worker"] = self.worker_name
        metadata["checkpoint_step"] = spec.checkpoint.stem.removeprefix("step_")

        encoded = io.BytesIO()
        output.save(encoded, format="JPEG", quality=94, subsampling=0, optimize=True)
        encoded.seek(0)
        if encoded.getbuffer().nbytes > 4 * 1024 * 1024:
            encoded = io.BytesIO()
            output.save(encoded, format="JPEG", quality=88, optimize=True)
            encoded.seek(0)

        response = self._post(
            f"/api/worker/jobs/{job_id}/complete",
            files={"output": ("output.jpg", encoded, "image/jpeg")},
            data={"metadata": json.dumps(metadata)},
            timeout=60,
        )
        response.raise_for_status()
        print(f"[worker] {job_id[:8]} · complete in {metadata['seconds']}s", flush=True)

    def run(self) -> None:
        models = available_model_ids()
        if not models:
            raise RuntimeError("No configured model checkpoints were found.")
        print(f"[worker] server: {self.server}", flush=True)
        print(f"[worker] device: {self.device_label}", flush=True)
        print(f"[worker] models: {', '.join(models)}", flush=True)
        print("[worker] waiting for jobs; press Ctrl+C to stop", flush=True)
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        backoff = self.poll_seconds
        while not self.stop_event.is_set():
            try:
                job = self.claim()
                backoff = self.poll_seconds
                if job is None:
                    self.stop_event.wait(self.poll_seconds)
                    continue
                job_id = str(job["id"])
                with self._state_lock:
                    self.active_job = job_id
                try:
                    self.process(job)
                # A bad image, checkpoint, CUDA failure, or upload error must fail only this job;
                # the long-running showcase worker should stay available for the next request.
                except Exception as error:  # noqa: BLE001
                    print(f"[worker] {job_id[:8]} · failed: {error}", flush=True)
                    self.fail(job_id, str(error))
                finally:
                    with self._state_lock:
                        self.active_job = None
            except requests.RequestException as error:
                print(f"[worker] connection error: {error}; retrying in {backoff:.0f}s", flush=True)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 30)
        print("[worker] stopped", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LumiCycle home-PC inference worker.")
    parser.add_argument(
        "--server",
        default=os.getenv("LUMICYCLE_REMOTE_URL", ""),
        help="Deployed site URL (or set LUMICYCLE_REMOTE_URL).",
    )
    parser.add_argument(
        "--secret",
        default=os.getenv("LUMICYCLE_WORKER_SECRET", ""),
        help="Worker secret (prefer LUMICYCLE_WORKER_SECRET).",
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.server:
        raise SystemExit("Set LUMICYCLE_REMOTE_URL or pass --server.")
    worker = RemoteWorker(args.server, args.secret, args.poll_seconds, args.name)

    def stop(_signum: int, _frame: Any) -> None:
        worker.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    worker.run()


if __name__ == "__main__":
    main()
