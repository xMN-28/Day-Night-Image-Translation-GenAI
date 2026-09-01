from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "lumirender_physics_bdd100k"
CONTROL_DIR = RUN_DIR / "control"
STATUS_PATH = CONTROL_DIR / "status.json"
STOP_PATH = CONTROL_DIR / "stop.request"
PID_PATH = CONTROL_DIR / "trainer.pid"
LOG_PATH = RUN_DIR / "training.log"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return str(pid) in result.stdout and "No tasks" not in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _trainer_pid() -> int:
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return 0


def start_or_resume() -> str:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    pid = _trainer_pid()
    if _process_running(pid):
        return f"Training is already running as process {pid}."
    STOP_PATH.unlink(missing_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    command = [
        sys.executable,
        "-m",
        "daynight.train_lumirender",
        "--config",
        "configs/lumirender.yaml",
        "--resume",
        "auto",
        "--continuous",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    log.close()
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return f"Started/resumed training as process {process.pid}."


def save_and_stop() -> str:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    pid = _trainer_pid()
    if not _process_running(pid):
        return "Trainer is not running. The latest checkpoint remains safe."
    STOP_PATH.write_text(f"requested_at={time.time()}\n", encoding="utf-8")
    return "Save & Stop requested. Finishing the current optimizer step, then saving everything."


def _tail(path: Path, lines: int = 24) -> str:
    if not path.exists():
        return "No training output yet."
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=lines)).replace("\r", "\n")[-8000:]


def current_status() -> tuple[str, str, str]:
    status = _read_json(STATUS_PATH)
    latest = _read_json(RUN_DIR / "checkpoints" / "latest.json")
    pid = _trainer_pid()
    running = _process_running(pid)
    step = int(status.get("step", latest.get("step", 0)))
    state = "RUNNING" if running else str(status.get("state", "READY")).upper()
    stage = status.get("stage", "not started")
    checkpoint = status.get("checkpoint") or latest.get("filename") or "none yet"
    metrics = status.get("metrics", {})
    metric_text = " · ".join(
        f"{key}={float(value):.4f}" for key, value in metrics.items() if isinstance(value, (int, float))
    )
    summary = f"{state} · global step {step:,} · stage: {stage}"
    details = f"Checkpoint: {checkpoint}\n{metric_text}".strip()
    return summary, details, _tail(LOG_PATH)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LumiRender Training Control") as demo:
        gr.Markdown("# LumiRender Training Control\nGlobal steps persist across every normal resume.")
        summary = gr.Textbox(label="Live status", interactive=False)
        details = gr.Textbox(label="Checkpoint and latest metrics", interactive=False, lines=3)
        with gr.Row():
            start = gr.Button("Start / Resume", variant="primary")
            stop = gr.Button("Save & Stop", variant="stop")
            refresh = gr.Button("Refresh")
        action = gr.Textbox(label="Last action", interactive=False)
        log = gr.Textbox(label="Live text log", interactive=False, lines=18, autoscroll=True)
        timer = gr.Timer(2.0)
        outputs = [summary, details, log]
        timer.tick(current_status, outputs=outputs)
        refresh.click(current_status, outputs=outputs)
        start.click(start_or_resume, outputs=action).then(current_status, outputs=outputs)
        stop.click(save_and_stop, outputs=action).then(current_status, outputs=outputs)
        demo.load(current_status, outputs=outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LumiRender training controller")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--start", action="store_true", help="Start/resume the trainer before serving")
    args = parser.parse_args()
    if args.start:
        print(start_or_resume(), flush=True)
    build_app().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=False)


if __name__ == "__main__":
    main()
