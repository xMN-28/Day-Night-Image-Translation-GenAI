"""Bundle existing college artifacts plus offline HTML study/demo indexes."""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import zipfile
from pathlib import Path

import markdown

PROJECT = Path(__file__).resolve().parents[2]
PACK = Path("E:/LumiCycleCache/share_packages/college_presentation_2026-10-24")
DOWNLOADS = Path("C:/Users/Admin/Downloads")
ARCHIVE = DOWNLOADS / "entire project docs and stuff.zip"
if ARCHIVE.exists():
    raise FileExistsError(ARCHIVE)


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)


STYLE = """
body{font:17px/1.65 system-ui,sans-serif;max-width:1060px;margin:40px auto;padding:0 24px;color:#202936;background:#fafafa}
h1,h2,h3{line-height:1.25;color:#122238}h1{margin-top:30px}h2{margin-top:38px}
a{color:#125bad}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #cfd6de;padding:10px;text-align:left}
th{background:#eaf0f6}pre{overflow:auto;background:#edf0f4;padding:16px}code{font-size:.9em}
img{max-width:100%;height:auto}aside{background:#fff2cf;padding:12px 16px;border-left:4px solid #cc9c23}
nav{border-bottom:1px solid #ccc;padding-bottom:12px}figure{margin:25px 0 40px;padding:16px;background:white;border:1px solid #ddd}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%}figcaption{font-size:14px}
@media(max-width:650px){.pair{grid-template-columns:1fr}body{margin:20px auto}}
"""


def document(title: str, body: str, prefix: str = "") -> str:
    return (f'<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{STYLE}</style><body>'
            f'<nav><a href="{prefix}START_HERE.html">Start here</a> · '
            f'<a href="{prefix}CURRENT_FACTS_AND_VIVA.html">Current facts</a> · '
            f'<a href="{prefix}BACKUP_GALLERY.html">Offline results</a> · '
            '<a href="https://lumicycle-remote-demo.vercel.app/">Live demo</a></nav>'
            + body + '</body></html>')


presentation = PACK / "01_Presentation_Files"
for name in ("lumicycle_architecture.pdf", "Day_Night_Image_Translation_Report Final.pdf",
             "Logbook final.docx", "Day_Night_Image_Translation_Group24.pptx"):
    copy(DOWNLOADS / name, presentation / name)
copy(PROJECT / "output/pdf/report_cleanup_verified/Day_Night_Image_Translation_Report_Final.cleaned.pdf",
     presentation / "Day_Night_Image_Translation_Report_Final.metadata-cleaned.pdf")

for source in sorted(PROJECT.glob("*.md")):
    copy(source, PACK / source.name)
for source in sorted((PROJECT / "docs").rglob("*")):
    if source.is_file() and source.suffix.lower() in {".md", ".csv", ".png", ".jpg", ".jpeg", ".svg"}:
        copy(source, PACK / "docs" / source.relative_to(PROJECT / "docs"))
copy(PROJECT / "LICENSE", PACK / "LICENSE")
copy(PROJECT / "remote-demo/README.md", PACK / "remote-demo/README.md")

showcase = PROJECT / "outputs/demo_showcase"
inputs = sorted((showcase / "daytime_inputs").glob("*.jpg"))
assert len(inputs) == 14
for folder in ("daytime_inputs", "generated_nights", "comparisons"):
    for source in sorted((showcase / folder).iterdir()):
        if source.is_file():
            copy(source, PACK / "02_Demo_Images" / folder / source.name)
copy(showcase / "SHOWCASE_CONTACT_SHEET.jpg", PACK / "02_Demo_Images/SHOWCASE_CONTACT_SHEET.jpg")
copy(showcase / "README.md", PACK / "02_Demo_Images/README.md")

figure = PROJECT / "output/pdf/lumicycle_architecture"
for name in ("lumicycle_architecture.png", "lumicycle_architecture.svg", "README.md", "provenance.json"):
    copy(figure / name, PACK / "03_Architecture_Diagram" / name)
copy(PROJECT / "outputs/night_to_day_experimental/showcase_collages/bdd_four_night_to_day_latest_16500.png",
     PACK / "04_Experimental_Inverse/Night_to_Day_EXPERIMENTAL_16500.png")

for source in sorted(PACK.rglob("*.md")):
    prefix = "../" * (len(source.relative_to(PACK).parts) - 1)
    body = markdown.markdown(source.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "toc"])
    def local_links(match):
        link = match.group(1)
        return match.group(0) if "://" in link else f'href="{link}.html'
    body = re.sub(r'href="([^"#]+)\.md(?=["#])', local_links, body)
    if source.parts[-2] == "docs" or source.name in {"README.md", "TEAM_STUDY_INDEX.md", "FINAL_DAY_NIGHT_ARCHITECTURE.md"}:
        body = ('<aside>Study archive: documents may describe historical releases. '
                f'Use <a href="{prefix}CURRENT_FACTS_AND_VIVA.html">current facts</a> '
                'for today\'s selected model and demo behavior.</aside>' + body)
    source.with_suffix(".html").write_text(document(source.stem, body, prefix), encoding="utf-8")

gallery = ['<h1>Offline demo backup gallery</h1><p>14 curated validation examples. '
           'Left: original upload. Right: saved student + sky repair output. '
           'Pre-generated results, not live requests or a success-rate benchmark. '
           'Click an input to open the upload JPEG.</p>']
for source in inputs:
    day = Path("02_Demo_Images/daytime_inputs") / source.name
    night = Path("02_Demo_Images/generated_nights") / source.with_suffix(".png").name
    assert (PACK / night).is_file()
    gallery.append(f'<figure><div class="pair"><a href="{day.as_posix()}">'
                   f'<img src="{day.as_posix()}" alt="Daytime input {source.stem}"></a>'
                   f'<a href="{night.as_posix()}"><img src="{night.as_posix()}" '
                   f'alt="Generated nighttime output {source.stem}"></a></div>'
                   f'<figcaption>{source.name} · Day input / generated night</figcaption></figure>')
(PACK / "BACKUP_GALLERY.html").write_text(document("Offline results", "".join(gallery)), encoding="utf-8")

files = []
for source in sorted(PACK.rglob("*")):
    if source.is_file():
        files.append({"path": source.relative_to(PACK).as_posix(), "bytes": source.stat().st_size,
                      "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
(PACK / "PACKAGE_MANIFEST.json").write_text(json.dumps({"prepared_date": "2026-10-24",
    "purpose": "Private college presentation/study pack; home-PC remote inference",
    "daytime_inputs": 14, "contains_weights": False, "contains_credentials": False,
    "original_supplied_documents_unchanged": True, "files": files}, indent=2), encoding="utf-8")

with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for source in sorted(PACK.rglob("*")):
        if source.is_file():
            archive.write(source, "entire project docs and stuff/" + source.relative_to(PACK).as_posix())
with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert len([n for n in names if "/daytime_inputs/" in n and n.endswith(".jpg")]) == 14
    assert not any(n.endswith((".pt", ".pth", ".env", ".safetensors")) for n in names)
    for entry in files:
        data = archive.read("entire project docs and stuff/" + entry["path"])
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
print(json.dumps({"zip": str(ARCHIVE), "files": len(names), "MiB": round(ARCHIVE.stat().st_size / 2**20, 2),
                  "inputs": 14, "archive_integrity": "CRC + SHA256 for all packaged files verified"}, indent=2))
