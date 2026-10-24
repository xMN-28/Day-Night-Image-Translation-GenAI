"""Package the visually reviewed, explicitly curated experimental inverse set."""
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("E:/LumiCycleCache/experiments/night_day_demo_screen_20261024")
DEST = Path("E:/LumiCycleCache/share_packages/night_to_day_shortlist_20261024")
ZIP = Path("C:/Users/Admin/Downloads/LumiCycle_Handpicked_Night_to_Day_10.zip")
if ZIP.exists() or DEST.exists():
    raise FileExistsError("Preserve existing shortlist; choose a new version")
manifest = json.loads((ROOT / "manifest.json").read_text())
rows = {r["index"]: r for r in manifest["records"]}
selections = [
    (299, "Open residential junction", "Blue-hour/night source; grain and a remaining lit streetlight."),
    (269, "Service station and overhead cables", "Late-evening source; soft detail and small retained light points."),
    (300, "Tree-lined urban street", "Blue-hour/night source; grain and a retained light hotspot."),
    (187, "Open divided road", "Some streetlamp glow, residual grain and modest tonal inconsistency."),
    (288, "Open road with painted divider", "Slightly noisy sky, retained light points and softened distant detail."),
    (253, "Residential street with parked cars", "Retained streetlight flare and visibly imperfect fine textures."),
    (475, "Ordinary city street", "Hazy light/sky transition, softened detail and remaining headlights."),
    (411, "Urban street with apartment buildings", "Bright haze near the right facade and small retained lamp hotspots."),
    (137, "Broad urban intersection", "Retained lamps and signals; grain and small fine-structure artifacts."),
    (480, "Quiet street with parked cars", "Less convincing sky/foliage boundary and remaining streetlight glow."),
]
for folder in ("night_inputs", "generated_days", "comparisons"):
    (DEST / folder).mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
contact = Image.new("RGB", (1024, 920), "#15171b")
draw = ImageDraw.Draw(contact)
draw.text((12, 8), "LumiCycle | curated experimental night-to-day examples", font=font, fill="white")
draw.text((12, 35), "Night input / actual model output. Relative shortlist, not flawless results or an accuracy estimate.", font=small, fill="#bdc7d6")
gallery = []
selected = []
for number, (index, title, limitation) in enumerate(selections, 1):
    row = rows[index]
    stem = f"{number:02d}_val_{index:04d}"
    source = DEST / "night_inputs" / f"{stem}.jpg"
    output = DEST / "generated_days" / f"{stem}.png"
    shutil.copy2(row["input"], source)
    shutil.copy2(row["output"], output)
    with Image.open(source) as im:
        night = im.convert("RGB")
    with Image.open(output) as im:
        day = im.convert("RGB")
    assert night.size == day.size == (512, 288)
    panel = Image.new("RGB", (1024, 320), "#15171b")
    pd = ImageDraw.Draw(panel)
    pd.text((8, 5), f"{number:02d} | VAL {index:04d} | NIGHT INPUT", font=font, fill="white")
    pd.text((520, 5), "MODEL OUTPUT — EXPERIMENTAL 16.5k", font=font, fill="white")
    panel.paste(night, (0, 32))
    panel.paste(day, (512, 32))
    panel.save(DEST / "comparisons" / f"{stem}.png")
    x, y = (number - 1) % 2 * 512, 72 + (number - 1) // 2 * 168
    draw.text((x + 5, y), f"{number:02d}: VAL {index:04d} | input → output", font=small, fill="white")
    contact.paste(night.resize((256, 144), Image.Resampling.LANCZOS), (x, y + 24))
    contact.paste(day.resize((256, 144), Image.Resampling.LANCZOS), (x + 256, y + 24))
    gallery.append(f'<section><h2>{number:02d}. {html.escape(title)}</h2><div class="pair">'
                   f'<a href="night_inputs/{stem}.jpg"><img src="night_inputs/{stem}.jpg" alt="Night input"></a>'
                   f'<a href="generated_days/{stem}.png"><img src="generated_days/{stem}.png" alt="Model output"></a>'
                   f'</div><p>Known limitations: {html.escape(limitation)}</p></section>')
    selected.append({"order": number, "index": index, "title": title, "limitations": limitation,
        "input": source.relative_to(DEST).as_posix(), "output": output.relative_to(DEST).as_posix(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
contact.save(DEST / "CONTACT_SHEET.png")
notes = """# Curated experimental night-to-day demo set

10 relatively stronger examples from the current local 16,500-step inverse.
These are NOT ten flawless or validated photorealistic daylight results. Noise,
retained lamps/headlights, haze and imperfect sky boundaries remain. The inverse
has not passed its acceptance gate. Three easy sources (299,269,300) are visibly
late-evening/blue-hour despite the BDD nighttime annotation; do not call them
pitch-black-night conversions.

## Use

Extract the ZIP. Open GALLERY.html for offline before/after previews.
Upload the JPEGs in night_inputs to the website, select Night → Day
(experimental), and use the 16.5k model at 512 px maximum edge.
Use files 01–05 first; 06–10 are more visibly imperfect alternatives.
generated_days contains the untouched raw PNG model outputs. The website
compresses completed outputs to JPEG, so minor compression differences are normal.
No denoising, brightness edits, Photoshop, teacher output substitution or new
training was used. Inputs were resized/encoded as JPEG, then those exact saved
JPEGs were reopened before model inference.

## Selection, not an evaluation claim

Generated 487 validation previews, excluding the 12 reserved lockbox cases and
the previously rejected under-bridge case142. Heuristics only ordered the queue;
192 pairs were visually reviewed as contact sheets and 20 candidates at native
512×288 size before selecting these ten. The other generated outputs were not
all manually reviewed. The test set was not accessed; the fixed quality gate
was not changed. No bridges/underpasses or already-daytime examples were selected.
Known limitations for each selection are included in GALLERY.html and manifest.json.

Label these as cherry-picked examples in the presentation, not typical model
performance or a success rate. Sources include BDD100K licensed imagery: private
academic demonstration only, not a public dataset redistribution.

This is a separate supplement; the earlier college-document ZIP is unchanged.
"""
(DEST / "README.md").write_text(notes, encoding="utf-8")
(DEST / "GALLERY.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Night-to-day shortlist</title>'
    '<style>body{max-width:1100px;margin:30px auto;padding:20px;font:17px/1.6 system-ui;background:#f6f7f9;color:#1b2533}'
    '.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}section{margin:36px 0;padding:18px;background:white}'
    '@media(max-width:600px){.pair{grid-template-columns:1fr}}</style><h1>Curated night-to-day preview</h1>'
    '<p>Left: upload JPEG. Right: untouched local model output. Experimental 16.5k; visible artifacts remain. '
    'Files 01–05 are the preferred relative examples; this is not an unbiased performance evaluation. '
    'Some sources are blue-hour/evening. Click a source to open the exact upload file.</p>'
    '<p><a href="https://lumicycle-remote-demo.vercel.app/">Live website</a> · <a href="CONTACT_SHEET.png">Overview</a></p>'
    + ''.join(gallery) + '</html>', encoding="utf-8")
(DEST / "manifest.json").write_text(json.dumps({"date": "2026-10-24", "model": manifest["model"],
    "step": manifest["step"], "checkpoint_sha256": manifest["checkpoint_sha256"],
    "curated_not_benchmark": True, "test_used": False, "lockbox_excluded": manifest["lockbox_excluded"],
    "quality_gate_passed": False, "selections": selected}, indent=2), encoding="utf-8")
with zipfile.ZipFile(ZIP, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(DEST.rglob("*")):
        if path.is_file():
            archive.write(path, "LumiCycle_Night_to_Day_Shortlist/" + path.relative_to(DEST).as_posix())
with zipfile.ZipFile(ZIP) as archive:
    assert archive.testzip() is None
    for row in selected:
        for kind in ("input", "output"):
            content = archive.read("LumiCycle_Night_to_Day_Shortlist/" + row[kind])
            assert hashlib.sha256(content).hexdigest() == row[kind + "_sha256"]
print(json.dumps({"zip": str(ZIP), "MiB": round(ZIP.stat().st_size / 2**20, 2),
                  "count": len(selected), "contact_sheet": str(DEST / "CONTACT_SHEET.png")}))
