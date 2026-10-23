"""Reclaim identical checkpoint bytes with same-volume NTFS hardlinks.

All checkpoint names and contents remain intact. Atomic saves are compatible;
never modify a linked checkpoint in-place. Default is an audit, not a mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    roots = [project / "runs", Path("E:/LumiCycleCache/experiments")]
    report = {"created_utc": datetime.now(UTC).isoformat(), "applied": args.apply,
              "reclaimed_bytes": 0, "links": []}
    for root in roots:
        root = root.resolve(strict=True)
        sizes = defaultdict(list)
        for path in sorted(root.rglob("*.pt")):
            if path.is_file() and not path.is_symlink():
                sizes[path.stat().st_size].append(path)
        for size, paths in sizes.items():
            if len(paths) < 2:
                continue
            hashes = {}
            for path in paths:
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"Checkpoint escaped audited root: {path}")
                checksum = digest(path)
                original = hashes.setdefault(checksum, path)
                if original == path or os.path.samefile(original, path):
                    continue
                record = {"original": str(original), "duplicate": str(path),
                          "sha256": checksum, "bytes": size}
                if args.apply:
                    temp = path.with_name(path.name + ".dedup-link.tmp")
                    if temp.exists():
                        raise FileExistsError(temp)
                    # Both paths have the same root/volume. Verify immediately
                    # before replacement; original remains independently named.
                    if digest(path) != checksum or digest(original) != checksum:
                        raise RuntimeError("Checkpoint changed during cleanup")
                    os.link(original, temp)
                    os.replace(temp, path)
                    assert os.path.samefile(original, path)
                report["links"].append(record)
                report["reclaimed_bytes"] += size
    folder = project / "artifacts/cleanup"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("dedup_applied.json" if args.apply else "dedup_audit.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), "duplicate_files": len(report["links"]),
                      "GiB": round(report["reclaimed_bytes"] / 2**30, 3),
                      "applied": args.apply}))


if __name__ == "__main__":
    main()
