# Final Day–Night Architecture

Authoritative current day→night snapshot: **22 October 2026**.

“Final” names the current showcase architecture, not a claim that its research problems are solved. This document describes the implemented, selected pipeline rather than an earlier proposal. Recent night→day experiments have **not** changed its generator weights, architecture, or default sky-repair path.

## 1. What the current model actually is

The current demo uses **LumiCycle student + sky repair — experimental**: a locally implemented, teacher-distilled RGB translation network followed by a locally trained sky/vegetation segmentation network and explicit source-guided sky compositing.

It is a **hybrid learned-and-analytic image translation pipeline**. It is not a diffusion model at inference, not a physically exact renderer, and not simply the original V2 checkpoint. External CycleGAN-Turbo outputs were used as training targets; their contribution must be acknowledged.

The supported public task is **day→night only**. Night→day was withdrawn because its outputs were unreliable. Separate experimental inverse training is not part of this release.

| Component | Current selection |
|---|---|
| Local Gradio model name | `LumiCycle distilled student` |
| Remote model ID | `distilled_best` |
| RGB generator | EMA `G_day_night`, cumulative global step **30,600** |
| Generator file | `E:\LumiCycleCache\experiments\turbo_vgg_target_control\best.pt` |
| Sky/vegetation network | Frozen `SkyStudent`, step **7,800** |
| Matte file | `runs/sky_student_regions_v2/best.pt` |
| Integrated repair version | `2026-10-22-v1` |
| Default demo maximum edge | **512 pixels**, preserving source aspect ratio |
| Active learned parameter count | **12,545,231**: generator 11,447,541 + matte 1,097,690 |

Checkpoint paths can be overridden by environment configuration. The table identifies the selected local files, not every possible override or a guarantee that a remote worker is currently online.

Generator SHA-256:
`1277b23bda9c13bc4797a19f44fa164000060308b5270a2b436c9b6617ce2185`

Matte SHA-256:
`afb481a22ca789880805398e6a00798be4d89450a0c657e10191c7c454032c83`

Global step 30,600 includes earlier continuation stages with different objectives. It must not be described as 30,600 steps of one unchanged training recipe. The selected checkpoint is not the matched experiment's final step 30,850.

## 2. End-to-end data flow

```text
Uploaded daytime photograph
  └─ EXIF orientation correction → RGB → aspect-preserving resize
     → reflection padding to a multiple of 4 → normalize to [-1, 1]
        ├─ EMA ResNet RGB student → raw night prediction ───────────┐
        └─ Original source → SkyStudent at 384×216                  │
           → sky and vegetation probabilities                     │
           → connected sky support + source-guided soft matte      │
           → local color-hole recovery / foreground unmixing       │
                                                                  ↓
             Explicit dim night sky + thin-edge foreground recovery
               → combine with raw night prediction
               → remove padding → clamp → image export
```

The two branches use the same source photograph. Sky segmentation is guided by the **daytime input**, not by an already corrupted generated sky. The integrated compositor does not update either network's weights.

## 3. RGB student architecture

Source: `daynight/models/networks.py`, principally `ResnetGenerator`, `ResnetBlock`, `CBAM`, and `ResizeConv2d`.

For a padded input of shape `B×3×H×W`, the selected network is:

| Stage | Operation | Output shape |
|---|---|---|
| Stem | Reflection pad 3; 7×7 convolution; InstanceNorm; ReLU | `B×64×H×W` |
| Downsample 1 | 3×3 convolution, stride 2; InstanceNorm; ReLU | `B×128×H/2×W/2` |
| Downsample 2 | 3×3 convolution, stride 2; InstanceNorm; ReLU | `B×256×H/4×W/4` |
| Bottleneck | Nine attention-equipped residual blocks | `B×256×H/4×W/4` |
| Upsample 1 | Nearest-neighbor 2× resize; reflected 3×3 convolution; InstanceNorm; ReLU | `B×128×H/2×W/2` |
| Upsample 2 | Same resize/convolution pattern | `B×64×H×W` |
| RGB head | Reflection pad 3; 7×7 convolution; Tanh | `B×3×H×W` |

Each residual block computes:

```text
f(x) = Conv3×3 → InstanceNorm → ReLU → Conv3×3 → InstanceNorm
output = x + CBAM(f(x))
```

Both convolutions use reflected border context. The residual connection helps preserve intermediate features; it does not mathematically guarantee unchanged objects or pixels.

### Attention

CBAM first applies channel attention: spatial average and maximum pooling feed a shared 1×1 convolutional MLP, whose summed outputs pass through a sigmoid. At 256 channels, its reduction-16 hidden width is 16. Channel weights multiply the residual features.

Spatial attention then concatenates channel-average and channel-maximum maps, applies a 7×7 convolution and sigmoid, and multiplies the features again. These are learned feature gates, **not** object masks or proof of understanding scene lighting.

### Important exclusions

- **InstanceNorm is still present.** This is not a “no-normalization VGG16 generator.”
- The selected decoder uses resize-convolution, not the original V2 transposed convolutions.
- There are no active full-resolution encoder→decoder U-Net skips in this RGB generator.
- The optional detail-skip adapter and ILN code in the repository are not active trained components of this selected checkpoint.
- It predicts RGB directly; it does not explicitly recover albedo, depth, normals, materials, sun position, or actual nighttime light locations.
- No discriminator, reverse generator, DINO network, VGG network, or diffusion teacher executes in the selected inference path.

## 4. SkyStudent and the integrated sky repair

Sources: `daynight/sky_repair.py` and `daynight/sky_refinement.py`.

### Learned matte network

SkyStudent is a small U-Net. Each feature block has two 3×3 convolutions, each followed by GroupNorm with four groups and SiLU. Encoder widths are **24→48→96**, with a **192-channel** middle block. Average pooling downsamples between levels. The decoder bilinearly upsamples, concatenates the corresponding encoder features, and reduces channels **96→48→24**. A 1×1 head produces two logits: sky and vegetation.

Inference resizes the source to **384×216** for this network, applies independent sigmoid probabilities, and bilinearly restores those maps to the source inference size. This internal fixed-size resize can distort non-road aspect ratios even though the final output preserves its aspect ratio. The network is frozen at inference.

### Why segmentation alone was insufficient

Thin wires and branches can occupy less than a matte-network pixel. A whole tree or cable bundle can be labeled foreground, including bright sky visible through its holes. Upsampling cannot reconstruct all missing details. Simple masked replacement can therefore leave white holes or hard patch boundaries.

### What the integrated repair does

1. **Connected sky support:** propagate from confident upper-frame sky on a small grid to suppress disconnected dashboard/windshield reflections. This introduces an explicit road-scene prior.
2. **Guided refinement:** use source luminance and local foreground/background color estimates to construct a continuous soft alpha rather than a hard segmentation cutout.
3. **Color-hole recovery:** above the observed sky horizon, recover pixels resembling nearby confident sky even if the semantic network labeled a cable bundle as foreground. Reliable nearby sky is required.
4. **Thin-edge recovery:** use source-derived foreground in a narrow sky neighborhood, including an 8-pixel max-pool radius and 1-pixel smoothing. Strong source-edge selection avoids blending bright generated wire halos back in.
5. **Explicit sky composition:** create a dim blue-gray field with a smooth vertical gradient and a small amount of retained source cloud texture. Approximate unmixing subtracts the old daylight sky contribution before introducing this new background.
6. **Bounded boundary exposure:** transfer locally estimated brightness to source foreground; the integrated thin-edge route caps its exposure ratio at **0.12**, versus **0.24** in the historical compositor route.
7. **Preserve the base elsewhere:** away from the repair support, retain the RGB student's output.

The sky color begins at RGB `[0.026, 0.035, 0.055]` in the compositor's `[0,1]` image convention, with an added vertical gradient. **These are image-space compositing values, not calibrated physical radiance.** No moon, stars, replacement geometry, or real light-source reconstruction is involved.

### Integration status

The repair is now invoked by default for the selected distilled-student option, locally and through the home worker. It was previously a separate experimental path; that earlier separation is no longer the current demo behavior.

This integration changed **inference**, not generator training. The 30,600-step checkpoint is unchanged. The current route does not perform the historical 16-phase generator averaging. Other comparison checkpoints do not automatically receive this repair.

`sky_repair=False` explicitly requests the raw student for ablation. Missing matte weights raise an error instead of silently presenting an unrepaired result. Metadata includes `sky_repair`, repair version, matte checkpoint/step, method, and sky coverage.

## 5. Training lineage and the selected final objective

### Earlier unpaired learning

Original LumiCycle/V2 used the CycleGAN framework: forward and reverse generators learn from separate day/night sets, with adversarial and reconstruction-related supervision. The repository includes cycle, identity, contrastive, semantic, edge, and regional illumination mechanisms across its historical recipes. Those original objectives must not be mistaken for the selected student's final training loop.

The BDD project manifests contain **5,000 training, 500 validation, and 1,000 test images per domain**, using strict daytime/night attributes and seed 42. Recorded dHash grouping was applied before splitting. Filename/exact-dHash audits are useful but do not prove the absence of every near-duplicate or related video frame.

### Teacher-distilled RGB student

The student was developed by continuing existing trained weights, not discarding all learning and starting again. The native-resolution pseudo-pair cache contains **2,048 training and 48 validation daytime sources**, with external CycleGAN-Turbo night targets at **512×288**. These targets are generated examples, **not paired ground-truth photographs of the same scene at night**.

The final matched no-VGG control continued from global step 29,850; its selected best checkpoint is local update 750, cumulative **30,600**. The matched experiment continued to 30,850, but later does not automatically mean better.

Final control settings: Adam, learning rate **1e-4**, betas **(0.5, 0.999)**, epsilon **1e-8**, zero weight decay, batch **4**, BF16, gradient clipping **3**, EMA decay **0.99**, native **512×288** training, paired random horizontal flips. Do not reuse the original GAN's batch/accumulation settings as this stage's settings.

### Active final control loss

Let `P` be the student's RGB prediction and `T` the generated teacher target, both in `[-1,1]`.

```text
R = mean absolute RGB error(P, T)
C = 0.5 × [L1(avgpool4(P), avgpool4(T))
           + L1(avgpool16(P), avgpool16(T))]
G = 0.5 × [L1(horizontal differences(P), horizontal differences(T))
           + L1(vertical differences(P), vertical differences(T))]

Fidelity = 10R + 2C + G
Total = Fidelity + 0.5 Highlight + 0.3 Footprint + 2 Sky
```

- **Highlight:** normalized RGB error weighted by bright teacher pixels. Brightness weight is `clamp((max_RGB((T+1)/2) - 0.55)/0.45, 0, 1)`.
- **Footprint:** normalized squared disagreement between predicted and teacher bright-pixel weights, discouraging oversized light regions.
- **Sky:** normalized RGB error to the **night teacher target** under a detached local sky matte; it is not a loss that preserves daytime sky colors.

There are **no active GAN, cycle, identity, PatchNCE, DINO, or VGG losses in this final control stage**. Historical loss fields inherited in checkpoint configuration do not override the executed distillation loop.

A separate matched ablation added frozen VGG feature supervision. Its measured effects were small and mixed; it did not establish a clear overall improvement, and the selected checkpoint is the **no-VGG control**. VGG as a loss network must not be confused with a VGG generator or removing generator InstanceNorm.

### External contribution

Official CycleGAN-Turbo supplied teacher targets, based on its externally pretrained diffusion system. The selected small student and matte execute locally without that teacher at inference. Teacher pretraining overlap with BDD is unknown, so this is not an uncontaminated comparison against an independent teacher.

Honest originality claim: local architecture implementation/adaptation, experimental training and loss choices, distillation integration, sky correction, safeguards, evaluation tooling, and deployment. Do not claim invention of CycleGAN, CBAM, U-Net, VGG, diffusion, or an entirely externally independent training process.

## 6. Runtime, interfaces, and reproduction

The local Gradio interface and remote frontend explicitly select the student with repair at 512 maximum edge. The remote frontend runs on Vercel; the home-PC worker performs inference using the local weights. Vercel does not supply the RTX 4070 Super or contain the trained model merely because it hosts the webpage. The PC and worker must be available for a remote request to finish.

The bare Python helper retains older defaults (`model_name="LumiCycle"`, `maximum_edge=768`). Therefore use an explicit model selection to reproduce the current demo:

```python
from PIL import Image
from daynight.inference import translate

output, metadata = translate(
    Image.open("input.jpg"),
    "day_to_night",
    model_name="LumiCycle distilled student",
    maximum_edge=512,
    sky_repair=True,
)
output.save("night_output.png")
print(metadata)
```

Preprocessing corrects EXIF orientation, converts to RGB, shrinks without stretching or enlarging, and reflection-pads to a multiple of four. CUDA inference uses BF16 for the neural branches; blending/unmixing uses float32. Output is cropped to the unpadded inference dimensions and converted to an image. It is not necessarily the original full resolution if resizing was required.

The selected student is deterministic in normal inference; the LumiRender intensity/wetness/seed controls should not be advertised as active stochastic lighting controls for this model. Offline local inference is possible once both checkpoints and dependencies are present. CPU execution is possible but has no GPU-speed guarantee.

The retained 22 October warm benchmark measured roughly **0.042 seconds median** and **0.045 seconds p95** over 15 calls on three validation images at 512 maximum edge on the RTX 4070 Super. It included local preprocessing/export and repair, but excluded cold generator loading and remote queue/network delay. It is not a latency guarantee for every image, resolution, or machine.

## 7. Previous approaches and what their outcomes mean

| Approach | Relationship to the current release | Observed shortcoming / interpretation |
|---|---|---|
| V1 at 13k / 40k | Historical unpaired translation checkpoints | Extra training did not reliably fix sky/detail artifacts; checkpoint age alone did not determine visual quality. |
| Original V2, selected 4.5k | Earlier preferred showcase and warm-start lineage | Convincing examples, but bright sky holes near branches/wires, retained daylight cues, and weak inverse outputs. |
| V2.1, 5.5k / 6k comparisons | Failed historical ablation | Visually weaker nighttime transformation. Over-constraining structure is a plausible contributor, not a proven sole cause. |
| LumiRender, 34,004 | Separate physics-guided architecture experiment, not this backbone | Observed outputs looked mainly darkened rather than convincingly relit. Planned physics modules/acceptance criteria do not prove successful learned decomposition. |
| Separate sky-repair candidate | Precursor to integrated correction | Improved sky treatment, but early attachment exposed hard cutouts/halos around cable bundles; integration required continuous hole recovery and boundary changes. |
| Raw distilled student | Current RGB network without its compositor | Faster/local teacher imitation, but residual bright sky/detail artifacts; no end-to-end guarantee of fine sky boundaries. |
| Student + repair | Current selected hybrid | Better controlled sky replacement, but masks and explicit compositor remain failure points. |
| New inverse experiments | Separate night→day development only | Not promoted to public inference; their step counts and decoder experiments do not change this day→night release. |

These observations are not a controlled proof that any whole architectural family cannot work. Training objectives, data quality, capacity, and optimization can all contribute.

## 8. Current architectural flaws and unsolved problems

1. **Fine cables/branches can still fail.** A coarse matte cannot perfectly resolve subpixel structures; local color recovery requires nearby reliable sky and may confuse bright surfaces with sky. The integrated fix reduces a failure mode, not every possible white gap.
2. **Sun and solar reflections are not reliably removed.** The system has no explicit sun identification, cast-shadow reasoning, surface geometry, or physical inverse-lighting solution.
3. **The sky is partly prescribed rather than generated.** A dim analytic gradient can look uniform or mismatched to the foreground. Source cloud texture is only weakly retained; true night cloud appearance is unknown.
4. **Boundary facades and thin foreground can become too dark.** The conservative exposure ceiling suppresses halos but trades away some plausible foreground brightness.
5. **Light placement and reflections are not physically enforced.** RGB learning can enlarge glare, invent lamps, distort text, or retain daytime reflection patterns. Sky repair does not solve these foreground errors.
6. **No hard geometry preservation guarantee.** Attention and gradient supervision encourage useful features but cannot prevent object warping or hallucinations.
7. **Road-domain bias.** BDD data, upper-frame sky connectivity, and fixed-aspect matte inference make arbitrary landscapes, portraits, indoor scenes, and unusual compositions less reliable.
8. **Teacher limitations propagate.** Generated targets can contain artifacts, and imitation can reproduce them. Matching the teacher is not the same as producing a physically correct night scene.
9. **Normalization may discard useful exposure statistics.** InstanceNorm is a design tradeoff, not a proven sole explanation of sky failure. A no-normalization replacement requires a controlled experiment rather than an assumption.
10. **Single-image nighttime state is ambiguous.** The input does not reveal which windows or lamps will be on, their powers, or the true night exposure. A convincing output is one plausible interpretation, not recovered ground truth.

## 9. Evidence, evaluation, and what not to claim

The current showcase contains **14 curated hybrid validation examples**, not a random test benchmark and not the earlier requested 20. The earlier raw showcase and rejected examples are historical artifacts, not proof of current quality on arbitrary uploads.

The selected checkpoint's recorded validation fidelity is approximately **0.50819**. This is a weighted teacher-matching score for that training/evaluation definition; it is **not 50.8% accuracy**, FID, or a realism percentage. Historical raw-control/VGG metrics do not measure the subsequently integrated hybrid by themselves.

A completed, independently verified final-test FID/KID, detector-retention, and blind-preference table has not been established for this release. Do not claim state of the art, universal scene support, perfect geometry, or that every planned acceptance criterion passed. BDD/license-restricted source images must not be redistributed merely because a showcase folder exists.

## 10. What changed most recently

- Integrated sky repair into the selected student's inference path, with continuous sky-color hole recovery and stronger source-derived thin-edge handling.
- Kept the selected generator at **30,600** and the matte at **7,800**; this was **not** another round of day→night training.
- Kept public inference day→night only, while experimental night→day work continued separately without replacing this release.
- This documentation update introduces **no** model, inference, training, deployment, or checkpoint change.

## 11. Code and evidence map

| Question | Inspect |
|---|---|
| RGB architecture and optional variants | `daynight/models/networks.py` |
| Selected checkpoint loading, EMA, preprocessing, routing | `daynight/inference.py` |
| Integrated repair and metadata | `daynight/sky_repair.py` |
| Matte network, guided filtering, hole recovery, composition | `daynight/sky_refinement.py` |
| Local demo defaults | `app.py` |
| Remote model-to-checkpoint mapping | `daynight/remote_worker.py` |
| Remote initial model/resolution | `remote-demo/components/demo.tsx` |
| Dated architecture history | `docs/CURRENT_MODEL_ARCHITECTURE.md` |
| Audited checkpoint metadata, counts, data manifests, benchmark | `artifacts/handovers/Day_Night_Project_Evidence_Handover_2026-10-22/` |

The evidence handover is a dated snapshot; newer inverse experiment files are not automatically part of it. Read source and actual checkpoint provenance before describing a repository option as a trained feature.

## 12. A concise explanation for the professor

“Our current day-to-night system uses a nine-block attention ResNet student with resize-convolution upsampling. We continued existing trained weights and distilled externally pretrained CycleGAN-Turbo outputs into this small local model. A separate locally trained U-Net estimates sky and vegetation, and explicit source-guided compositing repairs sky holes and fine boundaries. So our current inference system is hybrid, not a new diffusion backbone or a physically exact renderer. We evaluated earlier versions, retained their failures, and withdrew night-to-day from the demo because it was unreliable. Fine structures, direct sunlight, reflections, and out-of-domain scenes remain limitations.”
