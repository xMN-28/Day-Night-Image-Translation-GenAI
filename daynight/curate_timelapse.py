"""Review corrected DNIM clock labels without using warped RAFT targets.

Generate raw-frame review sheets and a camera-disjoint candidate manifest.
Clock-derived labels remain candidates until visual inspection, not ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .prepare_amos import _records
from .utils import atomic_json_dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("D:/datasets/kagglehub/datasets/stevemark/daynight-dataset/versions/1/DNIM"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/curated_timelapse_v1"))
    parser.add_argument(
        "--review", type=Path, default=Path("outputs/sky_pilot/timelapse_corrected_clock")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.review.mkdir(parents=True, exist_ok=True)
    cameras = sorted(p.stem for p in (args.root / "time_stamp").glob("*.txt"))
    ordered = sorted(cameras, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    validation = set(ordered[:3])
    rows, report = [], []
    for camera in cameras:
        records = _records(args.root / "Image", args.root / "time_stamp" / f"{camera}.txt")
        days = [(t, p) for t, p in records if 10 <= t.hour <= 14]
        nights = [(t, p) for t, p in records if t.hour >= 22 or t.hour <= 3]
        split = "val" if camera in validation else "train"
        sheet = Image.new("RGB", (1280, 408), "#101010")
        draw = ImageDraw.Draw(sheet)
        inspected = []
        for r, group in enumerate((days, nights)):
            ids = np.linspace(0, len(group) - 1, min(4, len(group)), dtype=int) if group else []
            for col, i in enumerate(ids):
                t, path = group[i]
                with Image.open(path) as image:
                    thumb = image.convert("RGB").resize((320, 180), Image.Resampling.LANCZOS)
                    lum = float(np.asarray(thumb).mean() / 255)
                label = "day" if r == 0 else "night"
                draw.text(
                    (col * 320 + 4, r * 204 + 4),
                    f"{camera} {label} {t:%H:%M} {split}",
                    fill="white",
                )
                sheet.paste(thumb, (col * 320, r * 204 + 24))
                inspected.append(
                    {"label": label, "path": str(path), "clock": t.isoformat(), "mean_rgb": lum}
                )
        sheet.save(args.review / f"camera_{camera}.jpg", quality=95)
        for nt, npth in nights:
            if not days:
                continue
            # Match the same fixed camera's temporally closest annotated midday.
            dt, dp = min(days, key=lambda v: abs((v[0] - nt).total_seconds()))
            rows.append(
                {
                    "camera": camera,
                    "split": split,
                    "day_path": str(dp),
                    "night_path": str(npth),
                    "day_clock": dt.isoformat(),
                    "night_clock": nt.isoformat(),
                    "alignment": "raw_fixed_camera_not_optical_flow",
                    "review_status": "candidate_not_yet_approved",
                }
            )
        report.append(
            {
                "camera": camera,
                "split": split,
                "days": len(days),
                "nights": len(nights),
                "review_samples": inspected,
            }
        )
    # This is generated data, not a source-code edit; original labels are untouched.
    with (args.output / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    atomic_json_dump(
        {
            "clock_source": "DNIM time_stamp metadata hour/minute, not filename hour",
            "validation_cameras": sorted(validation),
            "cameras": report,
            "pairs": len(rows),
            "training_ready": False,
            "reference": "https://zhhoper.github.io/version_0/dnim.html",
        },
        args.output / "provenance.json",
    )
    print(
        {
            "pairs": len(rows),
            "validation_cameras": sorted(validation),
            "review_folder": str(args.review),
            "training_ready": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
