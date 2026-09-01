from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

QUERIES = (
    "day night image translation",
    "single image relighting inverse rendering",
    "intrinsic image decomposition reflectance illumination",
    "night image synthesis camera noise glare reflection",
    "unpaired image translation structure preservation",
)


def _abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words = sorted(
        ((position, word) for word, positions in index.items() for position in positions),
        key=lambda item: item[0],
    )
    return " ".join(word for _, word in words)


def _fetch(query: str, per_page: int) -> list[dict[str, Any]]:
    parameters = urllib.parse.urlencode(
        {"search": query, "per-page": per_page, "select": "id,title,publication_year,doi,primary_location,abstract_inverted_index,cited_by_count"}
    )
    request = urllib.request.Request(
        f"https://api.openalex.org/works?{parameters}",
        headers={"User-Agent": "LumiRender college research project (mailto:student@example.com)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["results"]


def screen(output: Path, per_query: int = 50) -> int:
    records: dict[str, dict[str, Any]] = {}
    for query in QUERIES:
        for work in _fetch(query, per_query):
            record = records.setdefault(
                work["id"],
                {
                    "openalex_id": work["id"],
                    "title": work.get("title", ""),
                    "year": work.get("publication_year", ""),
                    "doi": work.get("doi") or "",
                    "url": (work.get("primary_location") or {}).get("landing_page_url") or "",
                    "citations": work.get("cited_by_count", 0),
                    "abstract": _abstract(work.get("abstract_inverted_index")),
                    "matched_queries": set(),
                },
            )
            record["matched_queries"].add(query)
    keywords = (
        "night", "relight", "illumination", "intrinsic", "reflectance", "translation",
        "reflection", "glare", "noise", "inverse render", "low-light", "low light",
    )
    rows = []
    for record in records.values():
        haystack = f"{record['title']} {record['abstract']}".lower()
        matches = [keyword for keyword in keywords if keyword in haystack]
        record["decision"] = "retain" if matches else "exclude"
        record["reason"] = ", ".join(matches[:6]) if matches else "no mechanism relevant to scope"
        record["matched_queries"] = " | ".join(sorted(record["matched_queries"]))
        rows.append(record)
    rows.sort(key=lambda row: (row["decision"] != "retain", -int(row["citations"] or 0)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce LumiRender's OpenAlex abstract screen")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/LUMIRENDER_LITERATURE_SCREEN.csv")
    )
    parser.add_argument("--per-query", type=int, default=50)
    args = parser.parse_args()
    count = screen(args.output, args.per_query)
    if count < 200:
        raise SystemExit(f"Only {count} unique records; increase --per-query.")
    print(f"Wrote {count} unique screened records to {args.output}")


if __name__ == "__main__":
    main()
