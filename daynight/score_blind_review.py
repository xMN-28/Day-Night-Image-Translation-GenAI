from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .utils import atomic_json_dump

CRITERIA = ("night_realism", "unchanged_geometry", "light_placement", "reflections", "artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a blinded LumiRender-versus-V2 team review")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", default="outputs/lumirender/human_review.json")
    args = parser.parse_args()
    wins = ties = comparisons = 0
    per_criterion = {criterion: {"lumirender": 0, "v2": 0, "tie": 0} for criterion in CRITERIA}
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for criterion in CRITERIA:
                vote = row[criterion].strip().lower()
                if vote not in {"lumirender", "v2", "tie"}:
                    raise ValueError(f"Invalid {criterion} vote: {vote!r}")
                per_criterion[criterion][vote] += 1
                comparisons += 1
                wins += vote == "lumirender"
                ties += vote == "tie"
    preference = (wins + 0.5 * ties) / max(1, comparisons)
    result = {
        "blinded": True,
        "comparisons": comparisons,
        "lumirender_preference_rate": preference,
        "criteria": per_criterion,
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
