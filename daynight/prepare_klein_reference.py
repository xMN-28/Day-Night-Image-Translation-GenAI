"""Prepare a pinned EXTERNAL edit-model reference under the approved E: cache.

Downloads only required diffusers components, not duplicate single-file weights.
No package installation or production model registration is performed here.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
from pathlib import Path

from .utils import atomic_json_dump

MODEL = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
ROOT = Path("E:/LumiCycleCache/official_models/flux2_klein4b")
PATTERNS = [
    "model_index.json",
    "transformer/*",
    "text_encoder/*",
    "tokenizer/*",
    "vae/*",
    "scheduler/*",
    "LICENSE.md",
    "README.md",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_download()
        return
    os.environ["HF_HOME"] = "E:/LumiCycleCache/klein_huggingface"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION or info.gated:
        raise ValueError(
            "Require the pinned ungated model; do not accept new gated terms automatically"
        )
    files = [
        item
        for item in info.siblings
        if any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in PATTERNS)
    ]
    expected = sum(item.size for item in files)
    missing = sum(
        item.size
        for item in files
        if not (ROOT / item.rfilename).is_file()
        or (ROOT / item.rfilename).stat().st_size != item.size
    )
    if shutil.disk_usage("E:/").free < missing + 20 * 1024**3:
        raise RuntimeError("Keep at least 20 GiB free on E: after the reference download")
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "external_pretrained": True,
        "expected_bytes": expected,
        "state": "downloading",
        "model_card": f"https://huggingface.co/{MODEL}",
        "files": [
            {
                "path": item.rfilename,
                "bytes": item.size,
                "sha256": getattr(item.lfs, "sha256", None),
            }
            for item in files
        ],
    }
    atomic_json_dump(metadata, ROOT / "reference_provenance.json")
    print(f"Pinned external reference: {expected / 1e9:.2f} GB; keeping 20 GiB free", flush=True)
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=str(ROOT),
        allow_patterns=PATTERNS,
        max_workers=3,
        token=False,
    )
    for item in files:
        if (ROOT / item.rfilename).stat().st_size != item.size:
            raise RuntimeError(f"Incomplete pinned file: {item.rfilename}")
    metadata["state"] = "downloaded_size_verified"
    atomic_json_dump(metadata, ROOT / "reference_provenance.json")
    print(f"Reference prepared at {ROOT}. No inference/quality acceptance yet.", flush=True)


def verify_download():
    metadata_path = ROOT / "reference_provenance.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["revision"] != REVISION:
        raise ValueError("Unexpected reference revision")
    checked = 0
    for item in metadata["files"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Incomplete reference component: {path}")
        if item["sha256"]:
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise RuntimeError(f"Reference SHA-256 mismatch: {path}")
            checked += 1
            print(f"Verified SHA-256 {item['path']}", flush=True)
    metadata["state"] = "downloaded_sha256_verified"
    metadata["verified_lfs_files"] = checked
    atomic_json_dump(metadata, metadata_path)


if __name__ == "__main__":
    main()
