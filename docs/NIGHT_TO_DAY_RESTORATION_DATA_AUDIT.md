# Restoration-data audit for the experimental inverse

22 October 2026. Private research preparation, not an accepted model or a new showcase.

## Why this follows the noise experiment

The preceding matched 500-update noise-robustness pilot reduced sensitivity to
synthetic input perturbations but did not fix convincing daytime appearance. That
was progress on a measured failure mode, not achievement of the project goal.
Rather than extend the same 111 flawed pseudo-targets blindly, this audit asks
whether suitable clean photographic supervision is available under the existing
Kaggle-only download preference.

No permission to download DarkDriving from its author-hosted services has yet
been received. That download was NOT started. The two downloads below are from
Kaggle, on E:, and no outside trained image generator was downloaded or run.

## Downloaded datasets and provenance

### LOL v1: real low/normal-exposure pairs, license unresolved

[Kaggle release](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset),
version 1; Kaggle metadata reports 348,058,979 bytes. Download returned normally
with exit 0. Local files:

`E:/LumiCycleCache/kagglehub/datasets/soumikrakshit/lol-dataset/versions/1/lol_dataset`.

The authors' [RetinexNet project](https://daooshee.github.io/BMVC2018website/)
describes paired low-light enhancement and reflectance denoising; its
[official code](https://github.com/weichen582/RetinexNet) documents `our485`
training data. This is exposure restoration, not ground truth for turning off
streetlights in a real nighttime driving scene.

Kaggle's dataset license field is `Unknown`. No LICENSE/README was found in this
download. The code repository's MIT license must NOT be assumed to license the
photographs. This dataset remains inspection-only: no training, redistribution,
public demo inclusion or claim that licensing has been resolved.

### LoLI-Street: author-linked street-scene enhancement pairs

[Kaggle release](https://www.kaggle.com/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street),
version 1; Kaggle metadata reports 2,824,788,499 bytes and CC BY-NC 4.0. The
[authors' repository](https://github.com/tanvirnwu/TriFuse_ACCV_2024) links this
exact Kaggle release. Download and extraction completed with exit 0. Local root:

`E:/LumiCycleCache/kagglehub/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street/versions/1/LoLI-Street Dataset`.

Important provenance limitation: the [primary paper](https://arxiv.org/html/2410.09831v1)
states that the paired training/validation dark images are synthetic, produced
with Photoshop. Its real low-light test data is a separate unpaired set. Thus
the paired data can provide aligned street detail and exposure targets, not
evidence of real day/night lamp-removal supervision. Do not call it 30,000 real
night/day captures. Use is restricted to this private non-commercial experiment
and cite the paper; no images are being pushed to GitHub or the public demo.

## Reproducible preparation

Existing virtual environment and libraries only; no package upgrades. In PowerShell:

```powershell
$env:KAGGLEHUB_CACHE='E:/LumiCycleCache/kagglehub'
.venv/Scripts/python.exe -c "import kagglehub; print(kagglehub.dataset_download('soumikrakshit/lol-dataset/versions/1'))"
.venv/Scripts/python.exe -c "import kagglehub; print(kagglehub.dataset_download('tanvirnwu/loli-street-low-light-image-enhancement-of-street/versions/1'))"
```

`daynight/audit_restoration_pairs.py` matches exact relative names in explicit
training-only directories; mismatched sets or held-out/traversal paths raise
errors. It selects 24 deterministic evenly spaced training examples, hashes
the actual selected source/target files, preserves aspect ratio at 512 width
(height rounded to a multiple of 4), and renders source, real reference and the
existing local 16k EMA inverse. No pretrained enhancement/editor is used.

Audit directories:

- `outputs/night_to_day_experimental/20261022_LOL_training_audit24`
- `outputs/night_to_day_experimental/20261022_LoLI_training_audit24`

Each contains native PNGs, six four-row sheets and `audit.json`. All six sheets
for EACH dataset were opened, totaling 24 diagnostic examples per dataset.
This is a sample review, NOT visual approval of every training pair. No LOL
`eval15`, LoLI `Val`/`Test`, BDD test or BDD lockbox photographs were opened.
Archive extraction is not the same as using test images for model evaluation.

## Actual visual findings

### LOL: strong evidence of corruption/texture problems

The model broadly preserves observable toy/bottle/book/furniture layout but adds
heavy grain and artificial texture. Flat ceilings and walls become mottled;
dark refrigerator backgrounds and drawers gain foliage/sky-like texture not in
the normal-exposure references. White surfaces are overbright and colors change
(for example toys and fabric). A few visible outdoor construction/courtyard
examples are more recognizable, but foliage and flat road/ground still acquire
patchiness. No new accepted night-to-day model is implied by these examples.

Mean local/reference RGB-L1 across these 24 TRAIN diagnostics is 0.145761 in
[0,1]. It is a descriptive exposure-reconstruction statistic, not a validation
score, downstream utility measure or proof of daytime realism. No fit to LOL
has been performed.

### LoLI-Street: useful target detail, but current inverse distorts it

The references contain diverse real street detail: skies/clouds, overhead
structures, wires, foliage, vehicles, signs and road markings. Synthetic inputs
retain daytime shadows and visible clouds. Several light-darkening inputs are
already recognizably daytime; the current inverse nevertheless overexposes and
restyles them. Examples include blocky yellow patches in `dense_1`, blurred
vehicles/buildings in `dense_5694`, smeared/high-contrast skies in `light_6084`
and `light_8432`, a dark horizontal cloud-like patch in `light_2562`, soft
branches and bright road streaks, and a bright spot near the car in
`moderate_2954`. `moderate_178` retains the overhead structure but changes sky
and building contrast strongly. These are observed failures, not evidence that
the dataset itself contains generated RGB targets.

Mean local/reference RGB-L1 across the 24 TRAIN diagnostics is 0.090590. It is
not comparable as a model-quality ranking to the LOL value because scene/input
distributions differ. Both are frozen-model data audits, not new experiments
trained on the respective datasets.

## Leakage checks before any training

All 30,000 LoLI training input/reference filenames match exactly. At least the
inspected `dense`, `light` and `moderate` versions of target IDs 1/100/1000 are
byte-identical across illumination variants. Randomly splitting filenames could
therefore leak the SAME target photo into training and validation.

`daynight/index_loli_training.py` indexes target byte hashes, computes dHash on
TRAIN targets only, and screens against the existing BDD val/test dHash metadata.
It never opens BDD held-out image files. Record duplicate target groups before
any new split, and exclude suspicious groups rather than tuning on them.
64-bit dHash is only a screen: it can miss crops, compression changes and nearby
video frames. It does not prove independent routes/cameras or absence of overlap.

The indexing process finished with actual exit 0: 30,000 paired filenames,
9,457 distinct target byte hashes, zero pairs within dHash distance 4 of the
3,000 saved BDD val/test image hashes. This does NOT prove no overlap. The
read-only reference hashes came from four existing manifests, whose file hashes
are recorded. No held-out BDD image pixels or performance metrics were used.
Index: `E:/LumiCycleCache/teacher_targets/LoLI_train_index_20261022.json`.
It is audit-only, not a ready-to-train approval manifest. The original dataset
files and existing split definitions are not edited.

## Next decision

LoLI-Street may support a bounded, warm-started exposure/detail-preservation
continuation with a matched control. It does not justify restarting training
from scratch, copying the authors' diffusion architecture, or expecting synthetic
darkening alone to remove real-night glare. Require grouped sampling, overlap
screening and a fixed 24-source REAL-night development comparison before accepting
any result. LOL training stays gated on clearer dataset terms. DarkDriving remains
gated on the user's dataset-source choice.

The finished public day-to-night weights, sky repair, routing and worker remain
unchanged. Existing inverse checkpoints remain preserved. No new training or
model promotion occurred during this data-audit pass.

Verification: 129 Python tests passed and relevant Ruff checks passed. Six new
pair-matching tests guard exact names and reject held-out/traversal paths; two
hash tests cover threshold boundaries/high unsigned bits and missing references.
These tests do not establish visual quality or dataset independence. Both
downloads, both 24-image audits and the index returned exit 0. At the final
process check, only the original public remote-worker Python pair remained.

## Subsequent bounded training result

The proposed LoLI continuation has now been executed. All96selectedTRAINpairs
were reviewed before approval; distinct targetSHA groups, one darkening variant
per group, no new heldoutpixels. Two1000-update continuations from16kcompleted:
primary111auditedexternal-target batches plus LoLIauxiliaryRGBweight3versus
matchedweight0control. Original architecture unchanged. Atglobal17k, LoLItraining
RGBL1drops.172399to.071747(control.179225), but all24real-night comparisons
at500and1000still fail on recurrent sky/glare/roadillumination. Neither promoted.
Detailed paths/protocol/reviewscope are in `NIGHT_TO_DAY_VISUAL_REVIEW.md`.
131tests passed and relevantRuffpassed. This demonstrates fit to synthetic
exposure pairs, not solved real-night relighting. Preserve16kparent and test
explicit scene-wide spatial context next; do not blindlyextend thisrecipe.
