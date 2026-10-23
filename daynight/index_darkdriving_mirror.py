"""Metadata-only inventory; no image downloads or dataset-license acceptance."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import atomic_json_dump

REF = "benxelua/darkdriving"


def summarize_files(files):
    images, prefixes = [], Counter()
    for row in files:
        path = PurePosixPath(row["name"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe mirror filename")
        prefixes["/".join(path.parts[:-1][:5])] += 1
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.append(row)
    domains = {}
    for row in images:
        parts = PurePosixPath(row["name"]).parts
        # Only explicit train-folder memberships qualify; never infer split by
        # appearance, filename timestamp or order.
        if "train" not in parts:
            continue
        domain = next((x for x in ("day", "night", "low", "high") if x in parts), None)
        if domain:
            normalized = tuple("DOMAIN" if x == domain else x for x in parts)
            domains.setdefault(domain, {})[normalized] = row["name"]
    pair_counts = {}
    examples = {}
    for a, b in (("day", "night"), ("low", "high")):
        shared = sorted(set(domains.get(a, {})) & set(domains.get(b, {})))
        pair_counts[f"{a}_{b}"] = len(shared)
        examples[f"{a}_{b}"] = [[domains[a][key], domains[b][key]] for key in shared[:3]]
    return {"file_count": len(files), "image_count": len(images),
        "total_listed_bytes": sum(r["bytes"] for r in files), "prefix_counts": dict(prefixes),
        "explicit_train_images": {k: len(v) for k, v in domains.items()},
        "exact_path_matched_train_pairs": pair_counts, "matched_examples": examples,
        "pairing_warning": "Name agreement is only a candidate correspondence, not verified image alignment"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier inventory")
    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.5,
        status_forcelist=(404, 429, 500, 502, 503, 504))))
    response = sess.get(f"https://www.kaggle.com/api/v1/datasets/view/{REF}", timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata["currentVersionNumber"] != 2:
        raise ValueError("Mirror version changed; re-review")
    partial = args.output.with_suffix(".partial.json")
    files, token, seen, start_page = [], None, set(), 0
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved["ref"] != REF or saved["version"] != 2:
            raise ValueError("Partial inventory version mismatch")
        files, token, start_page = saved["files"], saved["next_token"], saved["pages"]
        seen = set(saved["seen_tokens"])
    for page_number in range(start_page, 500):
        params = {"pageSize": 1000}
        if token:
            if token in seen:
                raise RuntimeError("Repeated pagination token")
            seen.add(token)
            params["pageToken"] = token
        response = sess.get(f"https://www.kaggle.com/api/v1/datasets/list/{REF}",
            params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        files.extend({"name": r["name"], "bytes": r["totalBytes"]} for r in page["datasetFiles"])
        print(f"Inventoried metadata page{page_number + 1}, files{len(files)}", flush=True)
        token = page.get("nextPageToken") or page.get("nextPageTokenNullable")
        atomic_json_dump({"ref": REF, "version": 2, "files": files,
            "next_token": token, "seen_tokens": sorted(seen), "pages": page_number + 1,
            "complete_listing": not bool(token), "image_bytes_downloaded": 0}, partial)
        if not token:
            break
    else:
        raise RuntimeError("Inventory incomplete")
    if len(files) != len({row["name"] for row in files}):
        raise ValueError("Duplicate listing names")
    result = {"ref": REF, "version": 2, "complete_listing": True,
        "mirror_metadata": metadata, "summary": summarize_files(files), "files": files,
        "image_bytes_downloaded": 0, "training_performed": False,
        "canonical_license": "Academic/noncommercial, attribution, no redistribution incl derivatives; mirror Apache label not authoritative",
        "canonical_license_url": "https://github.com/DriveMindLab/DarkDriving-ICRA-2026/blob/main/LICENSE",
        "mirror_authorization_verified": False, "approved_for_training": False,
        "public_demo_unchanged": True}
    atomic_json_dump(result, args.output)
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
