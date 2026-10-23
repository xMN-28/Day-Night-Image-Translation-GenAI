# Current LumiCycle architecture figure

Prepared 22 October 2026 from inspected implementation and freshly extracted
validation-image inference. This figure documents the **selected current
day-to-night hybrid**, not the original V2 training graph or LumiRender proposal.

## Deliverables

Local output directory: `output/pdf/lumicycle_architecture/`.

- `lumicycle_architecture.svg`: editable vector geometry and text, named groups,
  self-contained embedded PNG images, 180 mm × 120 mm view size.
- `lumicycle_architecture.pdf`: same scene exported directly as vector paths and
  embedded-font text. Only genuine image/mask examples are raster objects.
- `lumicycle_architecture.png`: 600 dpi PDF rendering, 4,252 × 2,835 pixels.
- `lumicycle_architecture_source.zip`: source, pinned rendering dependencies,
  real extracted assets, provenance, and instructions for rebuilding.
- `provenance.json`: actual checkpoint hashes, source-image hashes, source-code
  hashes, parameter counts, validation indices, and extraction checks.
- `render_validation.json`: PDF structure, embedded font and bounds checks.

The artifact directory is ignored by Git because the self-contained figure and
source package include licensed BDD100K imagery. Check dataset terms before
external redistribution or publication. The code and this guide contain no
redistributed dataset images or checkpoint weights.

## Reading the figure

The revised layout uses **3.2 px main connectors** (twice the original stroke),
larger arrowheads, and short direct routes. Matching circular `x` and `alpha`
ports denote the same source or refined-matte tensor wherever reused; they replace
long perimeter-spanning connections without removing those inputs. Dashed U-Net
skip paths remain because they depict actual local architectural connections.

**(a) Learned translation.** The EMA RGB student is a nine-block CBAM ResNet with
InstanceNorm and two resize-convolution upsamplers. Its axonometric feature
tensors are network schematics, not invented or mislabeled activation heatmaps.
The highlighted image regions correspond to the exact detail crop in panel (e).

**(b) Learned semantic matting and explicit refinement.** The independent local
SkyStudent U-Net predicts sky and vegetation probabilities at 384×216. Its
three dashed connections represent encoder-to-decoder skip concatenations.
The maps shown are the real predictions, upsampled to the inference dimensions.
R denotes connected support, guided color refinement and hole recovery. Source
RGB is also an input to this refinement, not only to the RGB generator.

**(c) Analytic compositor.** B is raw RGB prediction; alpha is the refined sky
matte; S is the explicit dim sky; U is source-derived foreground after approximate
daylight-background subtraction; M is the production repair support. With
production strength 1, the displayed formula is:

```text
Y = clip[(1 - M) B + M (U + alpha S)]
```

The two images show genuine U and alpha S tensors, without exposure boosts.
They are deliberately dark because that is their actual image-space contribution.
M is an explicit support field, **not identical to alpha**. Detailed formulas are
in `daynight/sky_refinement.py`; the extractor asserts agreement with production.

**(d) Attention residual block.** Two reflected 3×3 convolutions, InstanceNorm,
ReLU after the first normalization, channel then spatial CBAM gates, residual
addition. Gate icons illustrate operations, not measured attention activations.

**(e) Actual wire/sky crop.** The same pixel coordinates are cropped from input,
raw prediction, and repaired output. No detail is painted, inpainted, recolored,
or relit outside the implemented model/compositor. The crop is enlarged to show
boundaries; enlargement does not add source resolution.

The bottom provenance strip separates **training-only external Turbo supervision**
from local inference. No diffusion teacher, critic, VGG/DINO network or inverse
generator executes in this selected inference path.

## Suggested paper caption

**LumiCycle's implemented day-to-night inference pipeline.** (a) An EMA
attention-ResNet student produces a raw night prediction using nine CBAM-equipped
residual blocks and resize-convolution upsampling. (b) A separate local U-Net
estimates sky and vegetation probabilities; source-guided refinement recovers
soft sky boundaries and small sky-colored holes. (c) Explicit compositing combines
source-derived foreground, an analytic night sky, and the raw prediction within
a repair-support field. (d) The student residual block applies channel and spatial
attention before residual addition. (e) Identical-coordinate crops show an actual
wire/sky example before and after repair. All photographic and intermediate
examples are freshly generated from BDD validation source 0012 using selected RGB
EMA step 30,600 and matte step 7,800. External CycleGAN-Turbo supplied student
training targets but is absent at inference. The figure illustrates mechanism,
not general quality or physically correct night reconstruction.

## Reproduction: extract fresh tensors

Run from the complete repository root with the prepared project environment,
licensed local BDD data and both current checkpoints available:

```powershell
.\.venv\Scripts\python.exe -m scripts.figures.extract_architecture_assets
```

This reads validation sources 12, 142 and 23 only; source 12 appears in the figure.
The other sources form a local extraction-review sheet. No test data or training
updates are used. The extractor calls both raw and repaired public-manager paths,
then exposes the compositor intermediates and checks tensor agreement and exact
8-bit equality with the repaired public output. The documented run produced zero
8-bit difference for all three examples.

`--output <directory>` changes the artifact directory. The default figure expects
the default source 12 to remain among extraction indices. PyTorch/NVIDIA runtime
differences may change numerical outputs; hashes and version metadata make that
detectable. Do not claim universal cross-device bitwise reproducibility.

## Reproduction: draw and export

Use a separate lightweight Python environment, or the supplied Codex document
runtime. There is no need to install report-generation packages into the training
environment. Example with a dedicated rendering environment:

```powershell
py -3.12 -m venv .venv-figure
.\.venv-figure\Scripts\python.exe -m pip install -r scripts\figures\requirements-render.txt
.\.venv-figure\Scripts\python.exe scripts\figures\render_architecture_figure.py
```

The renderer uses Arial regular/bold/italic from `C:/Windows/Fonts`. On another
system, supply `--font-dir <directory>` containing `arial.ttf`, `arialbd.ttf`, and
`ariali.ttf`, respecting their license. Fonts are embedded into the PDF; SVG text
remains editable and uses Arial with Helvetica/sans-serif fallbacks. An editor
without Arial can substitute fonts and alter spacing. Paths remain fully editable.

Poppler `pdftoppm` renders the PNG when on PATH; otherwise PDFium from the pinned
dependencies does so. Layout and vector export do not depend on a browser or a
remote API. Default export is 600 dpi; `--dpi 900` makes a larger preview without
increasing the underlying real examples' native resolution.

To redraw from the provided source ZIP without the original project/checkpoints,
unzip it, install `scripts/figures/requirements-render.txt`, and run from its root:

```powershell
python scripts/figures/render_architecture_figure.py --output .
```

This uses the included hashed real assets. Re-extracting new tensors requires
the full project, trained checkpoints, and licensed source dataset.

## Verification and boundaries

- All illustrated probability maps, refined matte, foreground/sky contributions,
  and translated images come from actual inference; no generative illustration
  service or stock architecture template was used.
- Generator: 11,447,541 parameters; matte: 1,097,690; combined: 12,545,231.
- Extraction records the actual model and matte hashes; they match the current
  architecture snapshot, not a newly trained or different inverse checkpoint.
- The renderer verifies asset SHA-256, text/page bounds, one-page PDF structure,
  key text extraction and embedded TrueType fonts. SVG and PDF share vector drawing
  primitives but are separately exported, not screenshots of one another.
- Both PDF and SVG are raster-rendered for visual inspection. Label collisions,
  connector routing and SVG background transparency were corrected during review.
- The illustrative crop is not an all-case sky-fix claim. Sun/reflection removal,
  small-detail matting, invented lights, domain bias and incomplete final quality
  evaluation remain documented limitations.
- No public demo code, weights, model defaults, training state, deployment or
  GitHub remote was changed to create this figure.

See [FINAL_DAY_NIGHT_ARCHITECTURE.md](../FINAL_DAY_NIGHT_ARCHITECTURE.md)
and [PROJECT_MASTER_GUIDE.md](PROJECT_MASTER_GUIDE.md) for the broader explanation.
