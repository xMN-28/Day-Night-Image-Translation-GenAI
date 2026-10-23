"""Draw an editable SVG and vector PDF from verified, real model intermediates.

Uses a shared drawing scene for both formats: PDF is not a screenshot of SVG.
Requires Pillow, reportlab, pypdf, and either pdftoppm or pypdfium2 for preview.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import shutil
import subprocess
import zipfile
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path

from PIL import Image, ImageFont
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/pdf/lumicycle_architecture"
W, H = 1800, 1200
INK = "#25343E"
MUTED = "#65737C"
RULE = "#DBE2E6"
TEAL = "#82B6AD"
TEAL_DARK = "#397B73"
LILAC = "#B6ABCF"
LILAC_DARK = "#79658F"
AMBER = "#D7B580"
AMBER_DARK = "#A67A3E"


def mix(color: str, white: float) -> str:
    values = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(v + (255 - v) * white):02x}" for v in values)


class Figure:
    """Simple vector scene with parallel SVG and PDF backends."""

    def __init__(self, output: Path, font_dir: Path):
        self.output = output
        self.scale = (180 / 25.4 * 72) / W  # Standard 180 mm two-column figure.
        self.svg = [
            '<?xml version="1.0" encoding="utf-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'width="180mm" height="120mm" viewBox="0 0 {W} {H}" role="img">'
            ),
            "<title>LumiCycle: current day-to-night hybrid architecture</title>",
            (
                "<desc>Editable vector network illustrations with embedded genuine BDD validation "
                "input, student prediction, semantic maps, composition tensors and final output. "
                "No diffusion model executes in this inference path.</desc>"
            ),
        ]
        self.fonts = {}
        for name, filename in (
            ("regular", "arial.ttf"),
            ("bold", "arialbd.ttf"),
            ("italic", "ariali.ttf"),
        ):
            path = font_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing font {path}; provide --font-dir with Arial TTF files"
                )
            pdfmetrics.registerFont(TTFont("Figure-" + name, str(path)))
            self.fonts[name] = path
        self.pdf = canvas.Canvas(
            str(output / "lumicycle_architecture.pdf"),
            pagesize=(W * self.scale, H * self.scale),
            pageCompression=1,
        )
        self.pdf.setTitle("LumiCycle: distilled RGB translation with source-guided sky repair")
        self.pdf.setAuthor("LumiCycle project")
        self.pdf.setSubject("Implemented current day-to-night architecture, 22 October 2026")
        self.pdf.scale(self.scale, self.scale)
        self.text_boxes = []
        self.image_count = 0

    @contextmanager
    def group(self, name):
        self.svg.append(f'<g id="{html.escape(name)}">')
        yield
        self.svg.append("</g>")

    def line(self, points, color=INK, width=3.2, dashed=False, arrow=False):
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash}/>'
        )
        self.pdf.setStrokeColor(HexColor(color))
        self.pdf.setLineWidth(width)
        self.pdf.setDash([7, 5] if dashed else [])
        path = self.pdf.beginPath()
        path.moveTo(points[0][0], H - points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, H - y)
        self.pdf.drawPath(path)
        self.pdf.setDash([])
        if arrow:
            (x0, y0), (x1, y1) = points[-2:]
            angle = math.atan2(y1 - y0, x1 - x0)
            length, spread = 12, 5.3
            a = (
                x1 - length * math.cos(angle) + spread * math.sin(angle),
                y1 - length * math.sin(angle) - spread * math.cos(angle),
            )
            b = (
                x1 - length * math.cos(angle) - spread * math.sin(angle),
                y1 - length * math.sin(angle) + spread * math.cos(angle),
            )
            self.polygon([(x1, y1), a, b], color, None)

    def polygon(self, points, fill, stroke=INK, width=1):
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(
            f'<polygon points="{coords}" fill="{fill or "none"}" '
            f'stroke="{stroke or "none"}" stroke-width="{width}"/>'
        )
        p = self.pdf.beginPath()
        p.moveTo(points[0][0], H - points[0][1])
        for x, y in points[1:]:
            p.lineTo(x, H - y)
        p.close()
        if fill:
            self.pdf.setFillColor(HexColor(fill))
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.setLineWidth(width)
        self.pdf.drawPath(p, fill=int(bool(fill)), stroke=int(bool(stroke)))

    def rect(self, x, y, w, h, fill=None, stroke=RULE, width=1):
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill or "none"}" stroke="{stroke or "none"}" '
            f'stroke-width="{width}"/>'
        )
        if fill:
            self.pdf.setFillColor(HexColor(fill))
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.setLineWidth(width)
        self.pdf.rect(x, H - y - h, w, h, fill=int(bool(fill)), stroke=int(bool(stroke)))

    def circle(self, x, y, r, fill="#FFFFFF", stroke=INK, width=1.5):
        self.svg.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"/>'
        )
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.setLineWidth(width)
        self.pdf.circle(x, H - y, r, fill=1, stroke=1)

    def port(self, x, y, label, color=LILAC_DARK):
        """Named shared-tensor terminal, replacing a cross-panel routing wire."""
        self.circle(x, y, 16, "#FFFFFF", color, 2.2)
        self.text(x, y + 7, label, 23, "italic", color, "middle")

    def text(self, x, y, text, size=23, weight="regular", color=INK, anchor="start"):
        font = ImageFont.truetype(str(self.fonts[weight]), round(size * 4))
        length = font.getlength(text) / 4
        offset = {"start": 0, "middle": length / 2, "end": length}[anchor]
        left = x - offset
        if left < -0.1 or left + length > W + 0.1 or y > H or y - size < 0:
            raise ValueError(
                f"Text outside figure: {text!r}, bounds {(left, y - size, left + length, y)}"
            )
        self.text_boxes.append({"text": text, "bounds": [left, y - size, left + length, y]})
        style = f'font-family="Arial, Helvetica, sans-serif" font-size="{size}"'
        if weight == "bold":
            style += ' font-weight="700"'
        elif weight == "italic":
            style += ' font-style="italic"'
        self.svg.append(
            f'<text x="{x}" y="{y}" {style} fill="{color}" '
            f'text-anchor="{anchor}">{html.escape(text)}</text>'
        )
        self.pdf.setFont("Figure-" + weight, size)
        self.pdf.setFillColor(HexColor(color))
        if anchor == "middle":
            self.pdf.drawCentredString(x, H - y, text)
        elif anchor == "end":
            self.pdf.drawRightString(x, H - y, text)
        else:
            self.pdf.drawString(x, H - y, text)

    def image(self, path, x, y, w, h, border=True):
        with Image.open(path) as image:
            # No contrast boost, retouching, resynthesis, or undisclosed image recoloring.
            if abs(image.width / image.height - w / h) > 0.015:
                raise ValueError(f"Image aspect ratio mismatch: {path}")
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            self.svg.append(
                f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
                f'xlink:href="data:image/png;base64,{encoded}" preserveAspectRatio="xMidYMid meet"/>'
            )
            self.pdf.drawImage(ImageReader(image), x, H - y - h, width=w, height=h)
        self.image_count += 1
        if border:
            self.rect(x, y, w, h)

    def tensor(self, x, cy, w, h, depth, color, channels=None, label_y=337):
        """Axonometric feature tensor schematic, not a fabricated activation map."""
        y, dy = cy - h / 2, -depth * 0.48
        self.polygon(
            [(x, y), (x + depth, y + dy), (x + w + depth, y + dy), (x + w, y)],
            mix(color, 0.60),
            color,
        )
        self.polygon(
            [(x + w, y), (x + w + depth, y + dy), (x + w + depth, y + h + dy), (x + w, y + h)],
            color,
            color,
        )
        self.rect(x, y, w, h, mix(color, 0.27), color)
        if channels:
            self.text(
                x + (w + depth) / 2, label_y, str(channels), size=21, color=MUTED, anchor="middle"
            )

    def bracket(self, x1, x2, y, label, color=MUTED):
        self.line([(x1, y - 5), (x1, y), (x2, y), (x2, y - 5)], color, 1)
        self.text((x1 + x2) / 2, y + 27, label, size=21, anchor="middle", color=color)

    def finish(self):
        self.svg.append("</svg>")
        (self.output / "lumicycle_architecture.svg").write_text(
            "\n".join(self.svg), encoding="utf-8"
        )
        self.pdf.showPage()
        self.pdf.save()


def draw_figure(output: Path, font_dir: Path):
    assets = output / "assets"
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["assets"].items():
        actual = hashlib.sha256((assets / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Asset changed since extraction: {name}")
    f = Figure(output, font_dir)
    asset = lambda kind: assets / f"val_0012_{kind}.png"

    with f.group("paper-background"):
        f.rect(0, 0, W, H, fill="#FFFFFF", stroke=None)

    with f.group("figure-heading"):
        f.text(44, 49, "LumiCycle", 38, "bold")
        f.text(265, 47, "Distilled RGB translation with source-guided sky repair", 27)
        f.line([(44, 70), (1756, 70)], RULE, 1)

    with f.group("a-learned-rgb-translation"):
        f.text(44, 111, "(a)", 25, "bold")
        f.text(91, 111, "Learned RGB translation", 25, "bold")
        f.text(44, 163, "Day input  x", 24)
        f.image(asset("source"), 44, 180, 280, 157.5)
        f.text(44, 369, "EXIF / RGB / resize / reflect-pad", 20, color=MUTED)
        f.text(663, 163, "Gθ  ·  attention ResNet", 24, anchor="middle", color=TEAL_DARK)
        f.line([(324, 259), (354, 259)], arrow=True)
        f.line([(383, 259), (1036, 259)], color=TEAL_DARK, arrow=True)
        f.tensor(360, 259, 20, 112, 11, TEAL, 64)
        f.tensor(429, 259, 25, 83, 15, TEAL, 128)
        f.tensor(499, 259, 31, 57, 21, TEAL, 256)
        for index in range(9):
            f.tensor(575 + index * 21, 259, 12, 57, 8, mix(TEAL, 0.03 * index))
        f.text(668, 337, "256", 21, anchor="middle", color=MUTED)
        f.tensor(808, 259, 25, 83, 15, TEAL, 128)
        f.tensor(878, 259, 20, 112, 11, TEAL, 64)
        f.tensor(951, 259, 9, 124, 7, TEAL)
        f.bracket(355, 548, 355, "7×7 stem + 2× down")
        f.bracket(571, 770, 355, "9 attention blocks")
        f.bracket(801, 982, 355, "2× resize-conv + RGB")
        f.text(1054, 163, "Raw prediction  B", 24)
        f.image(asset("raw"), 1054, 180, 280, 157.5)
        f.line([(1334, 259), (1373, 259)], arrow=True)
        f.circle(1396, 259, 22, mix(AMBER, 0.85), AMBER_DARK, 1.8)
        f.text(1396, 268, "C", 25, "italic", AMBER_DARK, "middle")
        f.line([(1418, 259), (1456, 259)], arrow=True)
        f.text(1460, 163, "Hybrid night output  Y", 24, "bold")
        f.image(asset("repaired"), 1460, 180, 296, 166.5)
        f.text(1608, 378, "Crop padding / export", 21, anchor="middle", color=MUTED)
        f.text(1194, 369, "Single forward pass", 20, anchor="middle", color=MUTED)

    with f.group("source-guidance-and-routing"):
        # Matching labels carry the same tensors without perimeter loops.
        f.text(44, 414, "Repeated x / α ports denote shared tensors", 21, color=MUTED)
        f.port(264, 562, "x")
        f.line([(280, 562), (296, 562)], color=LILAC_DARK, arrow=True)
        f.port(934, 570, "x")
        f.line([(934, 586), (934, 603)], color=LILAC_DARK, arrow=True)
        f.port(1348, 324, "x")
        f.line([(1358, 311), (1382, 278)], color=LILAC_DARK, arrow=True)
        f.port(1396, 339, "α")
        f.line([(1396, 323), (1396, 282)], color=LILAC_DARK, arrow=True)

    with f.group("b-source-semantic-matting"):
        f.text(44, 468, "(b)", 25, "bold")
        f.text(91, 468, "Source-guided semantic matting", 25, "bold")
        f.text(264, 625, "384×216", 22, anchor="middle", color=MUTED)
        f.text(450, 499, "SkyStudent", 24, anchor="middle", color=LILAC_DARK)
        # Accurate three-level U-Net with concatenation skips, not a generator U-Net.
        positions = [
            (302, 562, 18, 78, 9, 24),
            (350, 598, 22, 58, 10, 48),
            (399, 630, 26, 42, 12, 96),
            (449, 662, 30, 31, 14, 192),
            (507, 630, 26, 42, 12, 96),
            (557, 598, 22, 58, 10, 48),
            (607, 562, 18, 78, 9, 24),
        ]
        for (x1, y1, w1, _, depth1, _), (x2, y2, *_rest) in pairwise(positions):
            f.line([(x1 + w1 + depth1, y1), (x2, y2)], color=LILAC_DARK, arrow=True)
        for x, cy, w, h, depth, _channels in positions:
            f.tensor(x, cy, w, h, depth, LILAC)
        for xa, xb, yy in ((320, 612, 521), (373, 560, 565), (425, 509, 603)):
            f.line(
                [(xa, yy + 5), (xa, yy - 10), (xb, yy - 10), (xb, yy + 5)],
                color=LILAC_DARK,
                width=2.1,
                dashed=True,
                arrow=True,
            )
        f.text(465, 709, "24 / 48 / 96 / 192 / 96 / 48 / 24", 20, anchor="middle", color=MUTED)
        f.text(
            465, 741, "GroupNorm + SiLU  ·  skip concatenation", 20, anchor="middle", color=MUTED
        )
        f.line([(634, 562), (695, 572)], color=LILAC_DARK, arrow=True)
        f.line([(634, 562), (695, 716)], color=LILAC_DARK, arrow=True)
        f.text(700, 505, "Sky probability", 22)
        f.image(asset("sky_probability"), 700, 521, 180, 101.25)
        f.text(700, 648, "Vegetation probability", 21)
        f.image(asset("vegetation_probability"), 700, 665, 180, 101.25)
        f.line([(880, 571), (922, 610)], color=LILAC_DARK, arrow=True)
        f.line([(880, 716), (927, 635)], color=LILAC_DARK, arrow=True)
        f.circle(934, 621, 16, mix(LILAC, 0.8), LILAC_DARK)
        f.text(934, 628, "R", 21, "italic", LILAC_DARK, "middle")
        f.line([(950, 621), (975, 621)], color=LILAC_DARK, arrow=True)
        f.text(1097, 535, "Refined sky matte  α", 24, anchor="middle")
        f.image(asset("alpha"), 975, 552, 245, 137.8125)
        f.text(1097, 725, "Color unmixing + hole recovery", 20, anchor="middle", color=MUTED)
        f.line([(1220, 621), (1246, 621)], color=LILAC_DARK, arrow=True)
        f.port(1262, 621, "α")

    with f.group("c-explicit-analytic-composition"):
        f.text(1323, 468, "(c)", 25, "bold")
        f.text(1370, 468, "Inside compositor C", 25, "bold")
        f.text(1323, 520, "Unmixed foreground U", 21)
        f.text(1549, 520, "Sky contribution αS", 21)
        f.image(asset("unmixed_foreground"), 1323, 538, 201, 113.0625)
        f.text(1537, 607, "+", 24, anchor="middle", color=AMBER_DARK)
        f.image(asset("sky_contribution"), 1549, 538, 201, 113.0625)
        f.text(1536, 692, "Y = clip[(1 - M)B + M(U + αS)]", 23, anchor="middle", color=AMBER_DARK)
        f.text(
            1536, 728, "M: repair support   S: analytic night sky", 20, anchor="middle", color=MUTED
        )
        f.text(
            1536,
            757,
            "Explicit composition, not learned relighting",
            20,
            anchor="middle",
            color=MUTED,
        )

    with f.group("d-attention-residual-block-detail"):
        f.line([(44, 790), (1756, 790)], RULE, 1)
        f.text(44, 834, "(d)", 25, "bold")
        f.text(91, 834, "One attention residual block", 25, "bold")
        f.text(44, 945, "z", 25, "italic")
        f.line([(64, 937), (112, 937)], color=TEAL_DARK, arrow=True)
        f.rect(116, 901, 20, 72, mix(TEAL, 0.18), TEAL_DARK)
        f.line([(136, 937), (234, 937)], color=TEAL_DARK, arrow=True)
        f.rect(238, 901, 20, 72, mix(TEAL, 0.18), TEAL_DARK)
        f.line([(258, 937), (348, 937)], color=TEAL_DARK, arrow=True)
        # Channel gate: schematic per-channel weights, intentionally not data activations.
        for index, height in enumerate((22, 44, 32, 55, 27)):
            f.rect(
                354 + index * 11, 966 - height, 7, height, mix(TEAL, index * 0.09), TEAL_DARK, 0.7
            )
        f.line([(409, 937), (464, 937)], color=TEAL_DARK, arrow=True)
        # Spatial gate icon with symbolic selected spatial cells.
        for row in range(4):
            for col in range(4):
                selected = (row, col) in ((0, 2), (1, 2), (2, 1), (2, 2), (3, 1))
                f.rect(
                    472 + col * 13,
                    911 + row * 13,
                    11,
                    11,
                    mix(TEAL, 0.15 if selected else 0.75),
                    mix(TEAL, 0.2),
                    0.5,
                )
        f.line([(526, 937), (639, 937)], color=TEAL_DARK, arrow=True)
        f.circle(659, 937, 18, "#FFFFFF", TEAL_DARK)
        f.text(659, 945, "+", 26, color=TEAL_DARK, anchor="middle")
        f.line([(677, 937), (766, 937)], color=TEAL_DARK, arrow=True)
        f.text(780, 945, "z′", 25, "italic")
        f.line([(80, 937), (80, 869), (659, 869), (659, 917)], color=TEAL_DARK, arrow=True)
        f.text(126, 1008, "Conv 3×3", 21, anchor="middle")
        f.text(126, 1035, "IN + ReLU", 19, anchor="middle", color=MUTED)
        f.text(248, 1008, "Conv 3×3", 21, anchor="middle")
        f.text(248, 1035, "IN", 19, anchor="middle", color=MUTED)
        f.text(376, 1008, "Channel gate", 20, anchor="middle")
        f.text(376, 1035, "pool / MLP / σ", 19, anchor="middle", color=MUTED)
        f.text(514, 1008, "Spatial gate", 20, anchor="middle")
        f.text(514, 1035, "7×7 / σ", 19, anchor="middle", color=MUTED)
        f.text(700, 1035, "CBAM residual addition", 20, anchor="middle", color=MUTED)

    with f.group("e-genuine-thin-boundary-detail"):
        f.text(946, 834, "(e)", 25, "bold")
        f.text(993, 834, "Actual wire / sky detail", 25, "bold")
        for x, kind, label in (
            (990, "source", "Day input"),
            (1250, "raw", "Raw student"),
            (1510, "repaired", "After repair"),
        ):
            f.text(x + 110, 869, label, 22, anchor="middle")
            f.image(assets / f"wire_crop_{kind}.png", x, 886, 220, 205.3333)
        # Actual crop location on each main image, same source coordinates.
        for image_x, image_y, image_w in ((44, 180, 280), (1054, 180, 280), (1460, 180, 296)):
            factor = image_w / 512
            f.rect(
                image_x + 215 * factor, image_y, 120 * factor, 112 * factor, stroke=AMBER, width=1.5
            )

    with f.group("training-provenance-and-caption-key"):
        f.line([(44, 1120), (1756, 1120)], RULE, 1)
        f.text(44, 1153, "Training only", 22, "bold", color=TEAL_DARK)
        f.text(
            219,
            1153,
            "External CycleGAN-Turbo pseudo-targets → student fitting; teacher and critics absent at inference",
            22,
            color=MUTED,
        )
        f.text(
            44, 1186, "EMA student 30,600  /  matte 7,800  /  12.55 M parameters", 19, color=MUTED
        )
        f.text(
            1756,
            1186,
            "BDD validation example 0012  ·  illustrative, not a benchmark",
            19,
            color=MUTED,
            anchor="end",
        )

    overlaps = []
    for index, first in enumerate(f.text_boxes):
        a = first["bounds"]
        for second in f.text_boxes[index + 1 :]:
            b = second["bounds"]
            dx = min(a[2], b[2]) - max(a[0], b[0])
            dy = min(a[3], b[3]) - max(a[1], b[1])
            if dx > 1 and dy > 1:
                overlaps.append([first["text"], second["text"]])
    if overlaps:
        raise ValueError(f"Overlapping text bounds: {overlaps}")
    f.finish()
    return {
        "embedded_image_instances": f.image_count,
        "text_bounds_checked": len(f.text_boxes),
        "overlapping_text_bounds": overlaps,
        "svg_text_elements_editable": True,
        "width_mm": 180,
        "height_mm": 120,
        "minimum_label_font_pt_at_180mm": 19 * f.scale,
        "connector_stroke_px": 3.2,
        "cross_panel_routing": "Shared x/alpha ports; no perimeter-spanning guidance routes",
    }


def render_and_check(output: Path, dpi: int, drawing_report):
    pdf = output / "lumicycle_architecture.pdf"
    preview = output / "lumicycle_architecture.png"
    poppler = shutil.which("pdftoppm")
    if poppler:
        subprocess.run(
            [
                poppler,
                "-r",
                str(dpi),
                "-singlefile",
                "-png",
                str(pdf),
                str(preview.with_suffix("")),
            ],
            check=True,
            capture_output=True,
        )
        renderer = "Poppler pdftoppm"
    else:
        import pypdfium2

        doc = pypdfium2.PdfDocument(str(pdf))
        bitmap = doc[0].render(scale=dpi / 72)
        bitmap.to_pil().save(preview, dpi=(dpi, dpi))
        renderer = "PDFium"
    reader = PdfReader(pdf)
    page = reader.pages[0]
    text = page.extract_text()
    for required in ("LumiCycle", "SkyStudent", "30,600", "7,800", "Training only"):
        assert required in text, f"Missing PDF text: {required}"
    assert len(reader.pages) == 1
    resources = page["/Resources"]
    embedded_fonts = []
    for name, reference in resources["/Font"].items():
        font = reference.get_object()
        descriptor = font.get("/FontDescriptor")
        if descriptor:
            descriptor = descriptor.get_object()
            embedded_fonts.append(
                {
                    "name": str(name),
                    "embedded": any(
                        key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
                    ),
                }
            )
    assert embedded_fonts and all(f["embedded"] for f in embedded_fonts)
    with Image.open(preview) as image:
        size = image.size
        image.thumbnail((1800, 1200), Image.Resampling.LANCZOS)
        image.save(output / "review_preview.png")
    report = {
        **drawing_report,
        "pdf_pages": 1,
        "preview_dpi": dpi,
        "preview_pixels": size,
        "renderer": renderer,
        "pdf_font_checks": embedded_fonts,
        "vector_pdf": "Text and network illustrations remain vectors; only real imagery is raster",
        "provenance": "provenance.json",
        "visual_review_required": True,
    }
    (output / "render_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    report = draw_figure(args.output.resolve(), args.font_dir)
    render_and_check(args.output.resolve(), args.dpi, report)
    write_source_bundle(args.output.resolve())


def write_source_bundle(output: Path):
    """Bundle code and hashed real assets; never include checkpoint/data archives."""
    guide = ROOT / "docs/ARCHITECTURE_FIGURE.md"
    if guide.exists():
        shutil.copyfile(guide, output / "README.md")
    sources = [
        "scripts/figures/extract_architecture_assets.py",
        "scripts/figures/render_architecture_figure.py",
        "scripts/figures/requirements-render.txt",
        "tests/test_architecture_figure.py",
    ]
    package = output / "lumicycle_architecture_source.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            source = ROOT / name
            if source.exists():
                archive.write(source, name)
        for name in (
            "README.md",
            "provenance.json",
            "render_validation.json",
            "lumicycle_architecture.svg",
            "lumicycle_architecture.pdf",
            "lumicycle_architecture.png",
            "asset_review.png",
        ):
            source = output / name
            if source.exists():
                archive.write(source, name)
        for source in sorted((output / "assets").glob("*.png")):
            archive.write(source, "assets/" + source.name)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        print(f"Source bundle: {package} ({len(archive.namelist())} files)")


if __name__ == "__main__":
    main()
