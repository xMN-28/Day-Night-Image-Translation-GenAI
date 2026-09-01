from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_ROOT = Path("runs/lumicycle_bdd100k")
EVENT_ROOT = RUN_ROOT / "tensorboard"
_event_path: Path | None = None
_events: EventAccumulator | None = None


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def _gpu_status() -> tuple[str, str, str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        utilization, used, total, temperature = [
            value.strip() for value in result.stdout.splitlines()[0].split(",")
        ]
        return utilization, used, total, temperature
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return "?", "?", "?", "?"


def _load_events() -> EventAccumulator | None:
    global _event_path, _events
    paths = sorted(EVENT_ROOT.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not paths:
        return None
    newest = paths[-1]
    if _events is None or newest != _event_path:
        _event_path = newest
        _events = EventAccumulator(str(newest), size_guidance={"scalars": 0, "images": 1})
    _events.Reload()
    return _events


def _latest(events: EventAccumulator, tag: str) -> tuple[int, float, float] | None:
    if tag not in events.Tags().get("scalars", []):
        return None
    values = events.Scalars(tag)
    if not values:
        return None
    value = values[-1]
    return value.step, value.value, value.wall_time


def status_text() -> str:
    events = _load_events()
    utilization, memory_used, memory_total, temperature = _gpu_status()
    now = time.time()
    if events is None:
        return (
            "# LumiCycle live monitor\n\n"
            "🟡 **Waiting for the first training update…**\n\n"
            f"GPU: {utilization}% · VRAM: {memory_used}/{memory_total} MiB · {temperature}°C"
        )

    generator = _latest(events, "G/total")
    discriminator = _latest(events, "D/total")
    accuracy = _latest(events, "D/accuracy")
    variance = _latest(events, "health/output_variance")
    cycle = _latest(events, "G/cycle")
    edge = _latest(events, "G/edge")
    if generator is None:
        return "# LumiCycle live monitor\n\n🟡 **Training is starting…**"

    step, generator_loss, wall_time = generator
    config_path = RUN_ROOT / "resolved_config.json"
    max_steps = 40000
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        max_steps = int(config.get("train", {}).get("max_steps", max_steps))

    history = events.Scalars("G/total")
    first = history[0]
    recent = history[-min(20, len(history))]
    step_delta = max(1, step - recent.step)
    time_delta = max(0.001, wall_time - recent.wall_time)
    steps_per_second = step_delta / time_delta
    elapsed = wall_time - first.wall_time + first.step / max(steps_per_second, 0.001)
    eta_seconds = (max_steps - step) / max(steps_per_second, 0.001)
    percent = min(100.0, 100.0 * step / max_steps)
    filled = min(20, int(percent / 5))
    bar = "█" * filled + "░" * (20 - filled)
    update_age = max(0.0, now - wall_time)
    active = update_age < 45 and utilization not in {"0", "?"}

    if active:
        headline = "🟢 **TRAINING IS ACTIVE**"
    elif update_age < 90:
        headline = "🟡 **Waiting for the next logged training update**"
    else:
        headline = "🔴 **No recent update — training may be paused or stopped**"

    d_accuracy = accuracy[1] if accuracy else float("nan")
    output_variance = variance[1] if variance else float("nan")
    if d_accuracy > 0.95:
        balance = "⚠️ Discriminator is temporarily very strong; watch whether this persists."
    elif d_accuracy < 0.55:
        balance = "⚠️ Discriminator is weak; the generator currently has the advantage."
    else:
        balance = "✅ Generator/discriminator competition is in a usable range."
    collapse = (
        "⚠️ Output diversity is very low; possible collapse."
        if output_variance < 0.001
        else "✅ Output diversity is healthy; no mode-collapse signal."
    )

    d_loss = discriminator[1] if discriminator else float("nan")
    cycle_loss = cycle[1] if cycle else float("nan")
    edge_loss = edge[1] if edge else float("nan")
    return f"""# LumiCycle live text monitor

{headline}  
Last model update: **{update_age:.0f} seconds ago** · Page refreshed: **{datetime.now(UTC).astimezone().strftime("%I:%M:%S %p")}**

## Progress

`{bar}` **{percent:.2f}%**  
**Step {step:,} of {max_steps:,}** · Elapsed **{_duration(elapsed)}** · ETA **{_duration(eta_seconds)}**  
Current speed: **{steps_per_second:.2f} steps/second**

## What the numbers mean

| Reading | Current value | Plain-English meaning |
|---|---:|---|
| Generator loss | {generator_loss:.3f} | How much the translator is still getting wrong; trend matters more than one value. |
| Discriminator loss | {d_loss:.3f} | How hard it is to distinguish generated images from real ones. |
| Discriminator accuracy | {d_accuracy * 100:.1f}% | Whether the critic is overpowering or underperforming. |
| Cycle loss | {cycle_loss:.3f} | Lower generally means scene content survives day ↔ night conversion. |
| Edge loss | {edge_loss:.3f} | Lower generally means roads, cars and building outlines are preserved. |
| Output variance | {output_variance:.4f} | Confirms the model is not producing the same image repeatedly. |

{balance}  
{collapse}

## GPU right now

**Usage {utilization}%** · **VRAM {memory_used}/{memory_total} MiB** · **Temperature {temperature}°C**

_This page refreshes itself every 2 seconds. Model-loss values are written every 25 steps, so they normally change about every 12–18 seconds._
"""


CSS = """
.gradio-container { max-width: 980px !important; margin: auto !important; }
#status-card { padding: 22px; border: 1px solid #334155; border-radius: 16px; }
"""


with gr.Blocks() as demo:
    status = gr.Markdown(status_text(), elem_id="status-card")
    timer = gr.Timer(value=2.0, active=True)
    timer.tick(status_text, outputs=status, show_progress="hidden", queue=False)
    demo.load(status_text, outputs=status, show_progress="hidden")


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True, css=CSS)
