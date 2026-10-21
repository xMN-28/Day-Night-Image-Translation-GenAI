# Corrected timelapse supervision and lighting pilots

Date: 21 October 2026. Status: useful data repair; **no accepted realism upgrade**.

## 1. Why this experiment was necessary

The sky/matte candidate improves sky gaps around branches, and the resize decoder
reduces periodic texture. Neither removes all exaggerated facade/vehicle highlights,
daylight shadow patterns, or uniformly dark canopy areas. Those are lighting problems,
not simply upsampling problems. This experiment tests better correspondence supervision
without discarding the previously trained generator.

## 2. Confirmed preparation bug

`daynight/prepare_amos.py` previously parsed the hour from the capture filename and
ignored the adjusted hour/minute in DNIM's `time_stamp/<camera>.txt`. Those clocks
differ by camera. Example from camera `00000850`:

```text
Filename: 20151101_212503.jpg
Metadata: 20151101_212503.jpg 20151101 14 27
```

The old filename rule called this a night image. Visual inspection shows daytime.
Conversely, `20151101_072507.jpg` has annotated time `00:27` and is visibly nighttime.
The parser now uses metadata clock fields, falling back to the filename **date only**
when the metadata uses a relative day counter. It does not invent a timezone for the
camera wall clock. Regression tests cover both cases.

The [original DNIM project](https://zhhoper.github.io/version_0/dnim.html) describes
17 fixed-webcam sequences with timestamps. We independently checked the actual local
metadata and generated image review sheets; the primary page alone does not establish
that every timestamp is accurate or every frame is suitable for paired training.

Existing manifests, source images, and old model checkpoints were not overwritten.
Changing the preparation code does **not** repair already generated old caches.

## 3. New pilot dataset

Scripts:

- `daynight/curate_timelapse.py`: clock-based candidate selection and review sheets.
- `daynight/prepare_timelapse_cache.py`: real-night median targets and static masks.

Selection uses conservative annotated clock bands: day hours 10–14; night hours
22–03. These are clock heuristics, not solar-elevation labels. Four day and four
night samples per camera were visually inspected across all 17 cameras. The corrected
samples show the intended domain separation. This is spot review, not full manual
annotation of every frame.

Camera split is determined by a fixed SHA-256 ordering, before model results:

```text
Validation cameras: 00003548, 00004232, 00011160
Other cameras: training candidates
```

Two training cameras were excluded after data-quality review:

- `00001323`: severe nighttime lens glare.
- `00003837`: strong horizontal nighttime banding.

Validation camera `00011160` also has banding, but it is **retained and flagged**;
it was not silently removed to improve the score. The cache contains 265 unique
daytime inputs from 12 training cameras and 61 daytime inputs from three disjoint
validation cameras. Multiple frames from the same camera are not independent scenes.

For each camera, a temporal median of its selected **real** night frames provides a
stable target and suppresses transient traffic. These are not diffusion-generated
pseudo-targets. A median can still blur motion, create intermediate light states, and
retain camera artifacts; it is not a perfect individual nighttime ground truth.
No dense optical-flow warp is used: the previous RAFT cache had severe distortions.
Fixed cameras can still move slightly, so correspondence remains approximate.

A frozen cached `facebook/mask2former-swin-small-cityscapes-semantic` teacher generates
training-only pseudo-masks. Person/rider/vehicle classes are excluded from correspondence
losses in both source and target, with a small erosion. Masks are imperfect on these
non-dashcam views. The existing installed Transformers loader reports missing Swin
layernorm weight/bias and unexpected relative-position-index buffers; this warning
was not hidden. Masks were spot-inspected, not certified as human labels.

Artifacts:

```text
data/curated_timelapse_v1/provenance.json
data/curated_timelapse_v1/cache_manifest.json
data/curated_timelapse_v1/cache/{train,val}/
outputs/sky_pilot/timelapse_corrected_clock/
outputs/sky_pilot/timelapse_corrected_cache/
```

Downloaded images remain local and are not redistributed.

## 4. Training controls and retained weights

All three pilots start from the same trained resize-decoder checkpoint, cumulative
lineage step 11,100:

```text
runs/resize_decoder_pilot/best.pt
SHA256 30C29A4F15D5768D578E08F2D951B47D1734BC5EBA7D59375B63CF25AF4CEBF6
```

Each pilot runs 1,000 updates and saves cumulative step 12,100 in a **different run**.
Those counters must not be added together as one continuous training run.

Common settings: BF16, batch one, G Adam LR `3e-5`, D LR `5e-6`, EMA `0.99`, gradient
clipping `3`, random flips, and a 50% paired / 50% BDD replay mixture. Existing
night-to-day weights are retained unchanged. Semantic masks are training-only.

Paired loss weights:

| Term | Weight | Purpose |
|---|---:|---|
| Static-region photometric pyramid, scales 4 and 16 | 5 | Approximate scene lighting correspondence |
| Static-region RGB L1 | 0.5 | Local appearance pressure |
| Frozen-parent RGB anchor on excluded dynamic regions | 2 | Avoid copying unrelated nighttime vehicles |
| Night discriminator | 0.25 | Night-domain pressure |
| Identity on real BDD night images | 0.5 | Limit unnecessary nighttime changes |

Pyramid pooling is mask-normalized: masked-out moving pixels are removed **before**
pooling, not just masked after they have contaminated pooled colors. A test covers a
moving region not aligned with the pooling grid. BDD replay retains the previous
sky/foreground/boundary objectives. This means the experiment does not isolate the
paired loss from every other objective or prove that the paired data alone fails.

The three controls are:

1. **Decoder only:** train the two upsampling stages and final RGB head; encoder and
   residual blocks stay frozen.
2. **Late residual blocks:** additionally adapt the last three pretrained residual
   blocks. Earlier encoder/residual weights stay frozen.
3. **Normalization adapter:** decoder plus learned instance/layer-normalization
   mixtures in the nine residual blocks; original convolution/attention weights stay
   frozen. Norm-parameter LR is `0.001`.

### What the normalization adapter is—and is not

The third control follows the IN/LN mixture mechanism in the
[official U-GAT-IT normalization implementation](https://github.com/taki0112/UGATIT/blob/master/ops.py).
It is named `InstanceLayerNorm` in our implementation:

```text
output = gamma * (rho * IN(features) + (1-rho) * LN(features)) + beta
rho is projected into [0,1] after each update.
```

It initializes `rho=1`, `gamma=1`, `beta=0`, preserving the existing InstanceNorm
function. A whole-generator test confirms matching initialization within `1e-6` and
bitwise retention of every existing weight. This is **not full AdaLIN**, because gamma
and beta are learned channelwise parameters, not predicted from input conditioning.
It is also not a wholesale U-GAT-IT implementation. After training, EMA rho min/mean/max
was approximately `0.7877 / 0.9752 / 0.9994`, so the mixture did learn away from its
initialization; lack of benefit cannot be explained by assuming it never changed.

## 5. Full held-out evaluation and decision

`daynight/evaluate_paired_lighting.py` evaluates all 61 cached validation inputs across
the three held-out cameras. Lower correspondence errors are better, but these are
approximate median-target diagnostics, **not** realism, FID, detector preservation,
or a uniquely correct night-lighting reconstruction.

| Model | Static photometric pyramid L1 | Static RGB L1 | Mean image luminance |
|---|---:|---:|---:|
| Retained resize parent | 0.12797 | 0.13589 | 0.11127 |
| Decoder-only paired pilot | 0.13284 | 0.13955 | 0.09530 |
| Late-block paired pilot | 0.13648 | 0.14255 | 0.08629 |
| IN/LN paired pilot | 0.13638 | 0.14254 | 0.08741 |
| Real-night median targets | — | — | 0.21641 |

All three pilots worsen the aggregate correspondence score and tend to darken the
image further. One banded validation camera shows a small correspondence improvement;
that does not reverse the aggregate decision or establish realism. The initial
three-frame previews and full 61-frame statistics are separate evidence and must not
be conflated.

All eight fixed 768-pixel BDD validation comparisons were inspected for each pilot,
with the same sky student/compositor on both sides. They retain the night sky, but
the cyan vehicle/facade highlights, exaggerated local glow, flat dark canopy, and
daylight-like shadow cues remain. Some foreground areas become washed-out or softer.
**None is promoted to the demo.** More recent step numbers are not an acceptance rule.

Checkpoint SHA-256:

```text
Decoder only: 857d37c84c3e6ab74107d89fa1f280937a8e719b662cbbf2c4cb12460b180ca2
Late blocks:  c291d221adb43c02ef69c8a0d0ff21dea5e8e2d7d3081f6361003ecf9ffb9208
IN/LN:        7537ff362dc54487f8fd8cf24beaea58bb0e426ccb06130326cc0411fe1fa002
```

Evidence folders:

```text
outputs/sky_pilot/paired_lighting_metrics/
outputs/sky_pilot/paired_lighting_lateblocks_metrics/
outputs/sky_pilot/paired_lighting_iln_metrics/
outputs/sky_pilot/paired_lighting_controlled_val/
outputs/sky_pilot/paired_lighting_lateblocks_val/
outputs/sky_pilot/paired_lighting_iln_val/
```

## 6. What the failures suggest next

These results do not prove that paired learning or normalization adaptation cannot
work. They rule out promoting **these specific short pilots**. Likely confounds and
next controls include:

- Only 12 training cameras supply independent paired scenes; effective scene diversity
  is tiny despite 265 training frames. More steps can simply memorize those cameras.
- Real-night exposure and light placement vary strongly. A deterministic L1 learner
  cannot infer which unseen lamps/windows switch on; averaging or darkening is an
  easier shortcut. Darker is not automatically more realistic: the actual held-out
  night targets here are substantially brighter than the generated images.
- The RGB anchor intentionally preserves the parent's dynamic-object colors. It cannot
  simultaneously serve as a direct correction target for the parent's cyan vehicle
  artifacts. A selective, geometry-preserving replacement needs its own controlled test.
- BDD replay strongly anchors non-sky appearance, and the discriminator's real examples
  remain BDD night frames even on paired-webcam updates. This domain/objective conflict
  may oppose the corrected webcam lighting targets. Test supervision/domain controls
  before declaring an architectural ceiling.
- A stronger external translator could provide an explicitly attributed reference or
  training-only teacher for a locally warm-started student. It must not be presented
  as the team's pretrained backbone or an original research invention.

No new unseen BDD test subset was used to select these pilots. The earlier exposed
five-scene test audit remains diagnostic, not a newly blind final test.

## 7. Reproduction

```powershell
.\.venv\Scripts\python.exe -m daynight.curate_timelapse
.\.venv\Scripts\python.exe -m daynight.prepare_timelapse_cache
.\.venv\Scripts\python.exe -m daynight.train_sky_gan --init-from runs/resize_decoder_pilot/best.pt --output runs/paired_lighting_pilot --steps 1000 --paired-cache data/curated_timelapse_v1 --decoder-only --generator-lr 0.00003
.\.venv\Scripts\python.exe -m daynight.train_sky_gan --init-from runs/resize_decoder_pilot/best.pt --output runs/paired_lighting_lateblocks --steps 1000 --paired-cache data/curated_timelapse_v1 --decoder-only --late-blocks 3 --generator-lr 0.00003
.\.venv\Scripts\python.exe -m daynight.train_sky_gan --init-from runs/resize_decoder_pilot/best.pt --output runs/paired_lighting_iln --steps 1000 --paired-cache data/curated_timelapse_v1 --decoder-only --adapt-norm --generator-lr 0.00003
.\.venv\Scripts\python.exe -m daynight.evaluate_paired_lighting --checkpoint runs/paired_lighting_iln/latest.pt --output outputs/sky_pilot/paired_lighting_iln_metrics
```

Keep prior outputs for resume; use a new output directory for a changed experiment.
Source data and cached teacher weights are prerequisites, not included in Git.
At review time D: had only about 3.4 GiB free; do not launch long checkpoint-heavy
runs there. No datasets or old checkpoints were deleted. Additional cache-drive
permission has been requested before larger reference-model downloads.

The goal remains active: these experiments repaired supervision and tested specific
mechanisms, but they did not yet deliver consistently convincing nighttime lighting.
