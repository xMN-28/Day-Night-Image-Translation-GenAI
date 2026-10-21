"""Check two uninterrupted updates against one update + restart + one update.

Uses copied trusted local checkpoints in a new directory; never overwrites the
source run. This narrow audit is not proof of long-run numerical equivalence.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from .compare_sky_gan import file_digest
from .utils import atomic_json_dump


def compare_state(a, b, path="", differences=None, errors=None):
    if differences is None:
        differences = []
    if isinstance(a, torch.Tensor):
        if a.is_floating_point():
            differences.append(float((a - b).abs().max()))
        try:
            torch.testing.assert_close(
                a, b, rtol=1e-6, atol=1e-7, msg=lambda msg: path + ": " + msg
            )
        except AssertionError as error:
            if errors is None:
                raise
            errors.append(str(error)[:500])
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare_state(a[key], b[key], f"{path}/{key}", differences, errors)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            compare_state(left, right, f"{path}/{i}", differences, errors)
    else:
        assert a == b, path
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--previous-targets", type=Path)
    parser.add_argument("--sky-matte", type=Path)
    parser.add_argument("--vgg-cache", type=Path, default=Path("E:/LumiCycleCache/torchvision"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--repeat-control", action="store_true")
    args = parser.parse_args()
    # Refuse reuse so an old audit cannot accidentally be reported as a new one.
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = file_digest(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    settings = saved["distillation"]
    start = saved["adaptation_step"]
    torch.set_num_threads(6)
    names = ["uninterrupted", "resumed"]
    if args.repeat_control:
        names.append("repeat_uninterrupted")
    folders = {name: args.output / name for name in names}
    for folder in folders.values():
        folder.mkdir()
        if args.deterministic:
            # Explicitly enable a backend-only setting in the audit forks. All
            # source weights/optimizers/RNG remain unchanged; source is untouched.
            fork = dict(saved)
            fork["distillation"] = {**settings, "deterministic": True}
            torch.save(fork, folder / "latest.pt")
        else:
            shutil.copyfile(args.checkpoint, folder / "latest.pt")

    def run(name, steps, log_name):
        command = [
            sys.executable,
            "-m",
            "daynight.train_teacher_student",
            "--parent",
            saved["parent_checkpoint"],
            "--targets",
            str(args.targets),
            "--output",
            str(folders[name]),
            "--steps",
            str(steps),
            "--max-minutes",
            "5",
            "--batch-size",
            str(settings.get("batch_size", 1)),
            "--lr",
            str(settings["learning_rate"]),
            "--highlight-weight",
            str(settings.get("highlight_weight", 0)),
            "--adversarial-weight",
            str(settings.get("adversarial_weight", 0)),
            "--footprint-weight",
            str(settings.get("footprint_weight", 0)),
            "--overfit-images",
            str(settings.get("overfit_images", 0)),
            "--semantic-weight",
            str(settings.get("semantic_weight", 0.25)),
            "--train-scope",
            settings.get("train_scope", "all"),
        ]
        if settings.get("continue_state", False):
            command.append("--continue-state")
        if settings.get("add_detail_skip", False):
            command.append("--add-detail-skip")
        if settings.get("resolution_curriculum_from_sha256"):
            if not args.previous_targets:
                raise ValueError("Curriculum resume audit requires --previous-targets")
            command.extend(("--previous-targets", str(args.previous_targets)))
        if settings.get("sky_target_weight", 0):
            if not args.sky_matte:
                raise ValueError("Sky-target resume audit requires --sky-matte")
            command.extend(
                (
                    "--sky-target-weight",
                    str(settings["sky_target_weight"]),
                    "--sky-matte",
                    str(args.sky_matte),
                )
            )
        if settings.get("perceptual_target_weight", 0):
            command.extend(
                (
                    "--perceptual-target-weight",
                    str(settings["perceptual_target_weight"]),
                    "--vgg-cache",
                    str(args.vgg_cache),
                )
            )
        if args.deterministic or settings.get("deterministic", False):
            command.append("--deterministic")
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
        )
        (args.output / log_name).write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Resume audit subprocess failed: {log_name}")

    run("uninterrupted", start + 2, "uninterrupted.log")
    run("resumed", start + 1, "first_update.log")
    shutil.copyfile(folders["resumed"] / "latest.pt", folders["resumed"] / "after_first_update.pt")
    run("resumed", start + 2, "restart_second_update.log")
    a, b = (
        torch.load(folders[name] / "latest.pt", map_location="cpu", weights_only=False)
        for name in ("uninterrupted", "resumed")
    )
    differences, errors = [], []
    keys = (
        "models",
        "ema",
        "optimizer",
        "conditional_critic",
        "critic_optimizer",
        "rng",
        "cuda_rng",
        "python_rng",
        "step",
        "adaptation_step",
    )
    per_state_max = {}
    for key in keys:
        per_key = []
        compare_state(a[key], b[key], key, per_key, errors)
        per_state_max[key] = max(per_key, default=0)
        differences.extend(per_key)
    repeat_errors, repeat_differences = [], []
    if args.repeat_control:
        run("repeat_uninterrupted", start + 2, "repeat_uninterrupted.log")
        repeat = torch.load(
            folders["repeat_uninterrupted"] / "latest.pt", map_location="cpu", weights_only=False
        )
        for key in keys:
            compare_state(a[key], repeat[key], key, repeat_differences, repeat_errors)
    assert a["step"] == saved["step"] + 2
    assert file_digest(args.checkpoint) == source_hash, "Source checkpoint changed during audit"
    result = {
        "passed": not errors and not repeat_errors,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "source": str(args.checkpoint),
        "source_sha256": source_hash,
        "source_global_step": saved["step"],
        "final_global_step": a["step"],
        "maximum_float_state_difference": max(differences, default=0),
        "floating_tensors_checked": len(differences),
        "rtol": 1e-6,
        "atol": 1e-7,
        "deterministic_backend_enabled_in_audit_forks": args.deterministic,
        "restart_per_state_maximum_difference": per_state_max,
        "restart_failed_tensor_count": len(errors),
        "restart_first_errors": errors[:5],
        "repeat_uninterrupted_control_run": args.repeat_control,
        "repeat_maximum_float_state_difference": max(repeat_differences, default=0),
        "repeat_failed_tensor_count": len(repeat_errors),
        "scope": "Two updates uninterrupted versus one update plus process restart plus one update; generator, EMA, critic, both optimizers and RNG. Not a long-run guarantee.",
    }
    atomic_json_dump(result, args.output / "result.json")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
