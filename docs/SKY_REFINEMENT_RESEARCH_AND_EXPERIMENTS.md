# V2 sky refinement: research audit and experiments

Date: 21 October 2026. Status: experimental; acceptance has not been established.

## Objective and retained foundation

Produce convincing day-to-night images with complete sky conversion and clean branches,
wires and other foreground boundaries. Retain the V2 step-4,500 generator as the foundation.
The original generator checkpoint is never overwritten by these experiments.

The first experiment is a separately trained semantic matte predictor plus an explicit
sky compositor on top of the existing GAN. This is a hybrid learned/analytic method. It
must not be described as a newly trained GAN that learned night lighting from scratch.
Night-to-day remains the existing V2 mapping and is not fixed by this experiment.

## Audit of the supplied other-AI recommendations

| Recommendation | Decision and reason |
|---|---|
| Soft sky matting | Adopt and test. Coarse semantic sky alone misses daylight gaps inside vegetation. The mixed tree/sky region needs color unmixing. |
| `L_sky = abs((G(x)-x)*A)` | Reject this formula for encouraging night conversion: minimizing it preserves the daytime sky. A correct sky objective needs a target nighttime appearance or distribution. |
| Global increase in edge preservation | Reject. It can preserve daytime contrast and frustrate the required lighting change, as in V2.1. |
| Source-chroma preservation on every boundary | Do not adopt blindly. Genuine night illumination also changes chroma; source equality can preserve the problem. |
| Six-channel conditional discriminator on random unpaired images | Not valid as written. A condition requires meaningful real `(source,target)` examples or a carefully designed alternative. Random real pairs teach independence, not scene correspondence. |
| AdaLIN | Valid research candidate. The proposed IN/LN mixture alone omits adaptive affine conditioning from the full U-GAT-IT design. Replacing normalization can shift all pretrained features; use a separately evaluated ablation. |
| Resize-convolution | Valid candidate for a decoder ablation. Existing transposed-convolution weights cannot simply be reinterpreted as equivalent resize-convolution weights. Distillation or adaptation is required. |
| Freeze all GAN critics during a long generator pilot | Avoid as a default. A fixed critic can be exploited by a changing generator. Freeze V2 itself for the first semantic experiment instead. |
| Exact target sky Lab error for unpaired inputs | No aligned target sky exists. Region statistics are distribution diagnostics; calling them pixelwise ground-truth error is misleading. |
| Sky IoU as a realism score | Useful for mask correctness, insufficient for night realism. Segmentation can remain correct when the sky is still bright daytime. |
| Separate restoration research for night-to-day | Agree with the problem distinction. The current goal first addresses day-to-night; this does not imply night-to-day is solved. |

## Verified primary references

- [Efficient and Realistic High Resolution Sky Replacement, CVMP 2019](https://www.cvmp-conference.org/files/2019/short/38.pdf): segmentation, image-guided refinement, and a separate tree/sky mixture with foreground/background color estimates. This directly motivates the prototype matting operation; we implement a simplified local-color approximation, not their exact software.
- [Fast Guided Filter, He and Sun](https://arxiv.org/abs/1505.00996): image-guided filtering. Our local luminance-linear filter follows the standard guided-filter equations; it is not the full fast subsampled algorithm.
- [Mask2Former Cityscapes checkpoint](https://huggingface.co/facebook/mask2former-swin-small-cityscapes-semantic): frozen external segmentation teacher already cached on the project PC. Teacher supervision is externally pretrained and must be attributed. Check its license before redistributing its weights.
- [U-GAT-IT, official implementation](https://github.com/taki0112/UGATIT): reference for adaptive layer/instance normalization, retained for a future controlled normalization ablation.
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/): reference for resize-convolution, retained for a future decoder experiment.
- [SOLO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Tzevelekakis_Sun_Off_Lights_On_Photorealistic_Monocular_Nighttime_Simulation_for_Robust_WACV_2025_paper.html): explicit estimated 3D geometry/material/light-source relighting using ray tracing. It is not a ready-to-use mask loss for our GAN.
- [TReFT](https://arxiv.org/abs/2511.20307): one-step translation by adapting pretrained rectified-flow networks. This is an alternative requiring a different training stack, not a compatible V2 layer replacement.
- [HELIOS](https://arxiv.org/abs/2609.00901): albedo-conditioned diffusion and solar-angle control; a possible larger research direction, not implemented here.

Paper existence does not establish that its claims transfer to our model or hardware.
The other references in the supplied note are not automatically treated as verified.

## Experiment implementation

`daynight/sky_refinement.py` contains:

1. A small U-Net (`SkyStudent`, width 24) with high-resolution skip connections,
   GroupNorm and bilinear resize-convolution. This network predicts a sky matte, not RGB.
2. A guided filter that uses input luminance to align a coarse matte with real edges.
3. Local RGB foreground/background unmixing in vegetation/sky mixtures. This is needed
   because a coarse segmenter often labels an entire leafless tree as vegetation.
4. A compositor that keeps V2 outside the sky neighborhood, changes sky appearance,
   and uses source detail with a local exposure transfer at thin foreground boundaries.

The initial sky color/gradient and exposure limits are explicit hand-set rendering priors.
They are not learned light physics. The method cannot infer actual lamp states, remove
daylight shadows correctly, or recover a true nighttime ground-truth photograph.

`daynight/train_sky_student.py` prepares frozen teacher targets from 1,200 predetermined,
evenly spaced day-training records and 48 separate day-validation records. Full-frame
images are resized to 384 x 216 so the student sees both skyline and road context.
The test split is excluded. The student is trained with boundary-weighted binary cross
entropy, soft Dice and gradient matching to pseudo mattes. That is distillation of semantic
supervision; pseudo labels can still contain teacher errors.

Optimizer: AdamW, learning rate 2e-4, weight decay 1e-4, batch size 6, BF16 autocast,
gradient norm clip 3. Initial pilot: 2,400 steps. Validation every 200 steps chooses the
best matte checkpoint. Visual RGB quality requires an additional gate: a low segmentation
loss cannot prove convincing nighttime output.

Atomic checkpoints store the student, optimizer, random-number states, step, best score,
and parent V2 path. `runs/sky_student_v1/STOP` requests a save at the next completed
optimizer step. Removing that file and rerunning resumes the saved student step.
The step count belongs to this newly initialized sky head; V2's existing weights remain
fixed at their recorded step. A warm start and an exact optimizer continuation are different.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m daynight.train_sky_student --prepare
.\.venv\Scripts\python.exe -m daynight.train_sky_student --steps 2400
.\.venv\Scripts\python.exe -m daynight.sky_pilot --student runs/sky_student_v1/best.pt --output outputs/sky_pilot/student_val
.\.venv\Scripts\python.exe -m pytest tests/test_sky_refinement.py -q
```

`runs/sky_student_v1/status.json` provides plain-text-readable progress. `validation.jsonl`
records every segmentation validation. Teacher cache provenance records the exact source
paths and split indices. Teacher-based and student-based RGB comparisons are written
separately so teacher quality cannot be presented as deployed student quality.

## Visual experiment log

- Initial teacher prototype: rejected. A narrow transition region left bright interiors
  in leafless canopies and created visible hard seams.
- Second teacher prototype: rejected. Smooth local fallback removed some seams but the
  transition neighborhood still failed to reach all canopy gaps.
- Third teacher prototype: expands refinement across the teacher's vegetation region
  above the observed sky horizon. Sky holes improve substantially, but foreground branch
  tone and transition errors remain. Do not call this accepted or deploy it automatically.
- Student training and subsequent visual review: ongoing. Consult the actual run files
  for status and scores; the original V2 remains the showcase default until the RGB gate passes.

## Acceptance checks still required

- Inspect all eight fixed validation examples, including overcast leafless trees,
  dense foliage, wires, bright buildings and no-visible-sky cases.
- Look at full images and boundary crops; reject bright sky holes, seams, glowing
  branches, erasure of objects, obvious black cutouts, or only global darkening.
- Compare teacher and student performance, including skies falsely predicted on buildings.
- Inspect a held-out fixed test subset only after settings are frozen; do not tune on it.
- Record inference latency and memory, and retain exact parent/checkpoint provenance.
- Add to the interactive app only after these checks support the intended appearance.

The goal remains active while the examples fail these checks.

## Completed runs and subsequent findings

The one-output matte student completed 6,000 steps. Its best validation checkpoint is
step 5,800, with objective 0.27647 versus 0.55025 at step 200. This is a pseudo-matte
agreement score, not a night realism score.

An interim 48-image validation diagnostic using the step-5,200 matte reduced mean bright
sky fraction from approximately 0.169 to 0.011 under independent teacher-derived masks.
That diagnostic uses a luminance threshold of 0.25, excludes cases with fewer than 100
reference-sky pixels, and includes 46 sky-containing images. It can be improved by merely
darkening the sky, so it cannot establish realism. Prepared tensor inference on that run
was approximately 0.155 seconds at 512 x 288; model loading and image/browser I/O were excluded.

The broader visual review rejected the first matte/compositor as a final solution:
facade-edge darkening, dark cutout regions, and overly flat skies remain on some scenes.
Increasing semantic support thresholds avoided some building errors but reintroduced
colored fringes around branches. This trade-off motivates a second semantic output.

### Generator-only fine-tuning experiment

`daynight/train_sky_gan.py` warm-starts V2's day-to-night generator from its EMA weights
and fine-tunes it with offline sky mattes. The nighttime discriminator also starts from
V2 and is updated at a lower learning rate. Foreground appearance is anchored to frozen
V2; sky and boundary targets come from the explicitly documented compositor. No semantic
teacher or mask is used by the generator at inference.

- Generator LR 1e-5; discriminator LR 5e-6; Adam betas 0.5/0.999.
- Sky supervision 10; foreground V2 anchor 4; boundary reconstruction 2; nighttime GAN 0.5;
  identity on real night images 0.5.
- Batch 1, BF16, full frames 384 x 216; D updated every second step; EMA decay 0.99.
- Completed 1,600 fine-tuning updates, retaining V2's step lineage as global step 6,100.
- Checkpoint: `runs/lumicycle_v2_skygan_pilot/latest.pt`.
- Raw generator previews: `runs/lumicycle_v2_skygan_pilot/previews`.

The raw generator looks more strongly nighttime but still leaves bright gaps in dense
branches and can produce exaggerated colored highlights. Combining it with the matte
improves sky completion, but some facade and boundary artifacts still prevent acceptance.

### Decoder-phase averaging

An inference ablation averages all sixteen stride-four encoder phases, using reflected
context and correctly aligned crops. This reduces phase-dependent grid texture at the
cost of sixteen generator calls. It does not repair semantic errors or replace training.
The synthetic identity test checks exact spatial alignment. It remains an optional
diagnostic in `daynight/compare_sky_gan.py --phase-average`.

### Region-aware matte extension

The next pilot adds a vegetation output to the trained matte student. Encoder/decoder
weights and the existing sky head are reused; only the added output row starts new.
Frozen teacher vegetation labels are cached separately in
`data/sky_student_fullframe_regions_v2`. The intended use is to apply branch-detail
correction inside vegetation while leaving nearby buildings under the GAN's appearance
mapping. This remains a hypothesis until the rendered images support it.

The old V2 showcase configuration and app default have not been replaced by these pilots.
The held-out test suite has not been used for gradient training or checkpoint selection.

## Saved experimental candidate and current status

The region-aware student completed step 8,200, warm-started from the one-output head's
step 5,800. The selected region-aware checkpoint is step 7,800, with two-output validation
objective 0.30178. This objective includes vegetation and is not comparable numerically
to the one-output head's objective.

The current candidate uses:

```text
Generator: V2 SkyGAN global 6,100
  runs/lumicycle_v2_skygan_pilot/latest.pt
  SHA256 2C9EDE564A06F05E3C894184B7736A5517AC16D10624D87E1A75FDCCDDC3EA4A
Matte: region-aware student 7,800
  runs/sky_student_regions_v2/best.pt
  SHA256 AFB481A22CA789880805398E6A00798BE4D89450A0C657E10191C7C454032C83
Preset: configs/sky_candidate.yaml
API model_name: LumiCycle Sky candidate
```

The 48-image validation diagnostic now has bright-sky fraction approximately 0.00189
versus V2's 0.16875 (46 images contain sufficient reference sky). The warm tensor-only
512 x 288 timing with phase averaging is approximately 0.143 seconds. A real public
`translate` call and the Gradio upload/API round trip both executed successfully; the
first Gradio inference reported 0.551 seconds including image conversion, excluding
model loading. These runtime figures have different measurement boundaries.

The five predetermined test indices `0,249,499,749,999` were then audited at 768 pixels.
All generated images, raw generator outputs, and comparisons are retained under
`outputs/sky_pilot/region_candidate_test_audit`. They now show a substantially stronger
night impression with far fewer bright canopy gaps. Remaining weaknesses include
overemphasized artificial-looking highlights, flat sky appearance, preserved daylight
shadow patterns, and occasional facade/matting artifacts. This is not a claim that the
subjective "very convincing" goal is fully met across the suite; the goal remains active.

A follow-up generator pilot with foreground anchor reduced from 4 to 2 reused global
6,100 weights and completed global 7,700. Its results are preserved in
`runs/lumicycle_v2_skygan_relax` and `outputs/sky_pilot/relaxed_candidate_final_val`.
The change did not clearly resolve the remaining realism problems, so the interactive
candidate retains global 6,100 rather than automatically preferring the newest model.

## Try the candidate

```powershell
.\.venv\Scripts\python.exe app.py
```

Choose **LumiCycle Sky candidate** and **Day → Night**. Original V2 remains available
and remains the default. The candidate rejects Night → Day with a clear message rather
than pretending this direction was improved. It uses locally cached generator/student
weights and does not load a segmentation teacher or call an external image-generation API.

Programmatic interface:

```python
from daynight.inference import translate
output, metadata = translate(image, "Day → Night", "LumiCycle Sky candidate", 768)
```

The candidate is an explicitly labeled hybrid of a fine-tuned local GAN, trained semantic
prediction, and analytic compositing. The Gradio skill informed the existing dropdown and
event integration. The remote Vercel site has not been redeployed or silently changed.

## Next experiment priorities

1. Replace the ad-hoc boundary color prior with foreground-color estimation supervised
   on mixed sky/vegetation regions. Concentrate on facade confusion and pale branches.
2. Train spatial illumination independently of raw RGB residuals so strong highlights
   do not become generic cyan/magenta patches.
3. Add aligned/static-region supervision for shadows and local light placement; the
   current unpaired data cannot identify a physically correct configuration.
4. Test a resize-convolution decoder by distilling from the retained trained generator,
   rather than replacing pretrained transposed-convolution weights as if equivalent.

Use the validation set for further decisions. The inspected five test examples are now
an exposed diagnostic subset and must not be presented later as a freshly untouched
blind test if future settings are changed in response to them.

## Resize-convolution decoder ablation, 21 October 2026

This experiment implements the decoder recommendation independently of additional
illumination/edge-loss changes. The basis is the mechanism discussed in
[Odena et al., Distill 2016](https://distill.pub/2016/deconv-checkerboard/): upsampling
followed by convolution avoids the uneven-overlap construction of transposed convolution.
It is not a guarantee against all artifacts or a solution to scene relighting.

Parent: trained SkyGAN global 6,100, same checkpoint/hash recorded above. The stem,
downsampling encoder, nine residual blocks, and CBAM weights are copied exactly and
remain frozen. Both upsampling stages become nearest-neighbor resize, reflection padding,
and 3×3 convolution. Kernels use an explicitly approximate transposed/flipped initialization;
this is **not** claimed to preserve the original function. The pretrained final RGB head
is retained and adapted with the new decoder. Night-to-day and critics are unchanged.

Training targets are the parent's aligned 16-phase-averaged RGB outputs, cached for
300 predetermined training images and 48 validation images. Only training targets receive
gradient updates. Losses match target RGB (10), pooled appearance at 4/16 scales (2),
and target finite differences (1). No source-edge equality is added. Adam LR is `1e-4`,
BF16, batch one, EMA `0.99`. Frozen-encoder SHA-256 is checked on every checkpoint save.

The run completed 5,000 decoder updates, cumulative lineage step 11,100. This is a
decoder adaptation count, **not** 11,100 steps training the new architecture from scratch.
It resumed its first 1,200 updates from the saved optimizer/RNG state. Validation loss
decreased from 0.64340 at adaptation 200 to 0.35574 at 1,200 and 0.29455 at 5,000.
This measures reproduction of the teacher, not superiority to the teacher.

Checkpoint: `runs/resize_decoder_pilot/best.pt`

```text
SHA256 30C29A4F15D5768D578E08F2D951B47D1734BC5EBA7D59375B63CF25AF4CEBF6
```

### Diagnostics and visual decision

On 48 cached validation scenes, 39 contain at least 100 eligible smooth-sky pixels.
Mean diagnostics over those 39 scenes, RGB scaled to `[0,1]`:

| Diagnostic | Raw parent | 16-phase parent | Resize decoder |
|---|---:|---:|---:|
| Period-4 phase-locked sky high-pass bias | 0.0014404 | 0.0001018 | 0.0001246 |
| RGB L1 to phase-averaged parent | 0.0110458 | 0 | 0.0119314 |
| Gradient L1 to phase-averaged parent | 0.0076826 | 0 | 0.0059516 |
| Mean image luminance | 0.1466411 | 0.1460746 | 0.1473561 |

The phase-bias diagnostic drops about 91% relative to raw parent. It can be gamed by
blurring, which is why gradient fidelity and full-size crops are also inspected. It
is not a FID score, boundary F1, detector-retention score, or human realism measurement.
Warm generator-only CUDA timing at 512×288 was median `0.00963 s` for the resize
decoder versus `0.14822 s` for the 16-pass reference; image I/O, matte, compositing,
and model loading are excluded from these two timings.

Evidence:

- `outputs/sky_pilot/resize_decoder_metrics_5000/metrics.json`
- `outputs/sky_pilot/resize_decoder_controlled_val/comparison_gallery.jpg`
- Individual 768-pixel four-panel images in the same folder.

All eight fixed validation scenes were visually inspected with the **same** matte and
compositor on both sides. The resize decoder retains the strong night appearance and
reduces periodic texture, but is slightly softer in some details. It does not fix the
parent's cyan vehicle/facade highlights, excessive local glow, uniformly dark canopy,
or implausible daylight shadow cues. The default candidate is therefore **not silently
replaced**. `configs/sky_resize_candidate.yaml` provides a separate optional ablation.
Night-to-day remains unchanged and is not considered successful.

### Reproduction

```powershell
.\.venv\Scripts\python.exe -m daynight.train_resize_decoder --steps 5000 --train-images 300
.\.venv\Scripts\python.exe -m daynight.evaluate_resize_decoder --output outputs/sky_pilot/resize_decoder_metrics_5000
.\.venv\Scripts\python.exe -m daynight.compare_sky_gan --checkpoint runs/resize_decoder_pilot/best.pt --student runs/sky_student_regions_v2/best.pt --reference-checkpoint runs/lumicycle_v2_skygan_pilot/latest.pt --reference-phase-average --reference-hybrid --reference-label "Parent + 16-phase + matte" --label "Resize decoder + same matte" --output outputs/sky_pilot/resize_decoder_controlled_val --edge 768
```

The decoder trainer atomically saves weights, EMA, optimizer, parent provenance,
adaptation/global counters, and RNG states. A `STOP` file in its run directory requests
saving after a completed update. This pilot does not establish a full resume-equivalence
benchmark for every interruption point or change the existing overnight trainer.

## Paired-data audit before further lighting supervision

`daynight/audit_paired_lighting.py` inspected the existing Dark Zurich manifest and
generated `outputs/sky_pilot/paired_lighting_audit/pair_contact_sheet.jpg` plus `audit.json`.
There are 2,416 records, three daytime capture sequence groups, all marked training;
all referenced day/night/confidence files exist. File presence alone is not alignment
quality. Several inspected aligned images have heavily distorted roads, warped lights,
and largely black confidence masks. The optical-flow confidence is not a moving-object
or static-class mask. Some sources labeled in the `day` path are visibly twilight.

Across all 2,416 cached confidence maps, median coverage above confidence `0.7` is
only `0.0001703` (about **0.017% of pixels**); 1,897 pairs have under 1% coverage at
that threshold. These values depend on the existing flow-confidence formula and do
not prove every low-confidence region is wrong, but they prohibit describing the
manifest as 2,416 reliable pixel-aligned pairs. There are only 1,426 unique daytime
paths; one daytime frame is reused 70 times.

Do not treat these warped targets as trustworthy full-frame pixel supervision. Before
training a lighting branch against them, audit confidence coverage, exclude moving
classes, reject badly aligned regions, and establish sequence-separated validation.
Do not randomly split neighboring repeated-route frames. The ACDC local summary records
400 unpaired night images and **zero** normal-condition pairs; it is not the originally
planned complete paired ACDC data. None of this audit modifies the source datasets.

The timelapse manifest was separately audited using the same tool. It contains 645
records across 17 cameras but only 60 unique source images; one source is reused 79
times. Median high-confidence coverage is about 14.76%; 239 records have under 1%
coverage. Eight evenly spaced examples were inspected in
`outputs/sky_pilot/timelapse_lighting_audit/pair_contact_sheet.jpg`. They reveal an
additional **domain-label problem**: the first nominal night target is visibly daytime,
and some nominal day sources are already night. A filename's timestamp is not evidence
of local solar time; exact provenance of these mistaken assignments needs auditing.
Some stationary city-square examples look more promising, but the entire cache must
not be accepted as trustworthy day/night pairs. Re-curating actual daytime and actual
nighttime frames per stationary camera is a concrete next step, before training a
paired lighting correction. Hold out whole cameras, not randomly sampled frame pairs.

The optional resize-pipeline smoke test successfully translated a validation input at
512×288 with both weight files local: reported first-call pipeline time `0.407 s`,
excluding model loading. This is not the generator-only timing above. All 38 tests
passed after the decoder and comparison changes; changed Python files passed Ruff.

This turn made measured decoder progress and exposed a supervision-quality problem.
The original goal of convincing nighttime conversion remains active and **unachieved**
across the fixed suite; a faster clean decoder is not a substitute for convincing lighting.

### Follow-up: corrected timelapse labels and controlled lighting pilots

See [CORRECTED_TIMELAPSE_AND_LIGHTING_PILOTS.md](CORRECTED_TIMELAPSE_AND_LIGHTING_PILOTS.md)
for the confirmed filename-clock bug, corrected metadata parsing, camera-disjoint
cache, and three completed 1,000-update pilots (decoder only, late blocks, IN/LN
normalization adapter). All three were visually reviewed and evaluated on all 61
held-out timelapse inputs. None improved aggregate correspondence or earned promotion
to the demo. This is negative experimental evidence, not an accepted model upgrade.

### Follow-up: official reference and warm-started distillation

See [REFERENCE_TEACHER_EVALUATION.md](REFERENCE_TEACHER_EVALUATION.md) for the
pinned official CycleGAN-Turbo benchmark, isolated E-drive environment, local target
cache, and warm-started local student experiments. The official reference is visibly
stronger but remains explicitly external. The local student has not yet matched its
localized lighting. Main showcase weights and deployment remain unchanged.
