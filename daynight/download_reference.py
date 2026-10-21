"""Resumable bounded-concurrency HTTP range download for an official checkpoint.

Writes disjoint ranges directly to one partial file, avoiding a second full
checkpoint copy on disk. Sidecar progress, not preallocated file size, is the
authority for resumption. Source URL/ETag/range responses are checked.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .utils import atomic_json_dump


def ranged_download(url: str, destination: Path, workers: int = 4):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".download")
    progress_path = destination.with_suffix(".ranges.json")
    with requests.get(
        url,
        headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(20, 60),
    ) as response:
        response.raise_for_status()
        if response.status_code != 206 or "Content-Range" not in response.headers:
            raise RuntimeError("Server does not support verified range downloads")
        total = int(response.headers["Content-Range"].split("/")[-1])
        etag = response.headers.get("ETag")
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        if (state["url"], state["size"], state["etag"]) != (url, total, etag):
            raise RuntimeError("Remote checkpoint changed; partial preserved, not mixed")
        if not partial.exists() or partial.stat().st_size != total:
            raise RuntimeError(
                "Partial file disagrees with range manifest; inspect before resuming"
            )
    else:
        # An existing sequential-download prefix contains actual received bytes.
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > total:
            raise RuntimeError("Partial prefix exceeds remote checkpoint size")
        width = max(1, (total - prefix + workers - 1) // workers)
        ranges = [
            {"start": start, "end": min(start + width, total), "done": 0}
            for start in range(prefix, total, width)
        ]
        state = {"url": url, "size": total, "etag": etag, "prefix_bytes": prefix, "ranges": ranges}
        with partial.open("r+b" if partial.exists() else "w+b") as handle:
            handle.truncate(total)
        atomic_json_dump(state, progress_path)
    lock = threading.Lock()
    last_report = [time.monotonic()]

    def transfer(index):
        part = state["ranges"][index]
        if part["done"] == part["end"] - part["start"]:
            return
        for attempt in range(3):
            start = part["start"] + part["done"]
            end = part["end"] - 1
            if start > end:
                return
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
            if etag:
                headers["If-Range"] = etag
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{total}"
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise RuntimeError("Server returned a mismatched byte range")
                    with partial.open("r+b", buffering=0) as handle:
                        handle.seek(start)
                        for block in response.iter_content(1024 * 1024):
                            if not block:
                                continue
                            if part["start"] + part["done"] + len(block) > part["end"]:
                                raise RuntimeError("Server returned too many bytes")
                            count = handle.write(block)
                            if count != len(block):
                                raise OSError("Short local checkpoint write")
                            with lock:
                                part["done"] += count
                                atomic_json_dump(state, progress_path)
                                if time.monotonic() - last_report[0] >= 10:
                                    downloaded = state["prefix_bytes"] + sum(
                                        p["done"] for p in state["ranges"]
                                    )
                                    print(
                                        json.dumps(
                                            {
                                                "phase": "ranged_official_download",
                                                "bytes": downloaded,
                                                "total": total,
                                            }
                                        ),
                                        flush=True,
                                    )
                                    last_report[0] = time.monotonic()
                if part["done"] != part["end"] - part["start"]:
                    raise OSError("Incomplete response; range will resume")
                return
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise
        raise RuntimeError("Range did not finish")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(transfer, range(len(state["ranges"]))))
    completed = state["prefix_bytes"] + sum(p["done"] for p in state["ranges"])
    if completed != total:
        raise RuntimeError("Checkpoint ranges are not complete")
    partial.replace(destination)
    atomic_json_dump({**state, "complete": True}, progress_path)
