# LumiCycle V2: Current Model Problems and Research Questions

Last reviewed: 21 October 2026

Purpose: a shareable problem brief for teammates or an external research assistant. This is a diagnosis, not a claim that the problems have been solved. Sections 13–16 provide concrete visual evidence, causal tests, and research constraints.

## 1. Scope of this diagnosis

This document analyzes the current showcase model, **LumiCycle V2 step 4,500**:

```text
Checkpoint: runs/lumicycle_v2_bdd100k/checkpoints/step_00004500.pt
SHA-256:   13767F7A1735B00A9F3A24568BC6DC10A89306CA4D0369C8B8A6C7CF3542E550
Training:  unpaired BDD100K road scenes, 256 x 256 random crops
Task:      separate day-to-night and night-to-day generators
```

The diagnosis is based on:

- all five deterministic day-to-night comparisons and all five deterministic night-to-day comparisons in `artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final`;
- the exact V2 configuration in `configs/lumicycle_v2.yaml`;
- the generator and discriminator implementation in `daynight/models/networks.py`;
- the losses in `daynight/losses.py` and their use in `daynight/trainer.py`;
- the recorded validation history and experiment lineage.

The visual observations are real. The explanations below are **engineering hypotheses supported by the implementation**, not experimentally proven causal claims. Each cause should be isolated in an ablation before it is treated as confirmed.

**Version distinction:** “current model” here means the retained V2 step-4,500 showcase baseline. A separate experimental sky-refinement candidate also exists in this workspace; section 15 records its remaining flaws. Its results must not be attributed to the original V2 generator, and it does not repair night-to-day.

## 2. Executive diagnosis

LumiCycle V2 does not perform physical relighting. It learns a deterministic mapping from the visual distribution of BDD100K daytime road images to the distribution of BDD100K night road images, and a separate inverse mapping. It is often able to produce a convincing overall night mood, but it does not explicitly understand sky, foliage, wires, lamps, materials, depth, glare, exposure, or missing scene radiance.

The main day-to-night problem is a conflict between two requirements:

1. the sky needs a large, smooth, low-frequency illumination and color change; and
2. branches, wires, poles, signs, and building edges in front of the sky need exact high-frequency preservation.

The current network compresses the image to one-quarter spatial resolution, has no full-resolution encoder-to-decoder skip path, and uses only heuristic sky supervision. Thin foreground structures and sky therefore become entangled. The generator may darken both, leave bright daytime-colored gaps, or produce red/cyan halos around the boundary.

The night-to-day direction has a more fundamental problem. A night photograph has clipped highlights, buried shadows, glare, noise, and information that the sensor never captured. An unpaired GAN cannot recover that missing information. The current model mostly increases luminance and shifts color statistics toward the daytime domain, which makes glare, haze, and sensor artifacts brighter rather than reconstructing a clean daylight scene.

## 3. Observed day-to-night failures

### 3.1 Incomplete sky conversion

**Observed behavior**

- Bright or cyan daytime sky remains visible through tree branches and around utility wires.
- Open sky may become darker while small holes between branches retain a daytime tone.
- In some images, the sky changes unevenly and develops dark patches rather than a coherent nighttime illumination field.

This is visible especially in day-to-night comparisons 1, 2, 3, and 5. Comparison 5 contains the clearest branch/sky boundary failure. Comparison 1 shows wires crossing a transformed sky with color fringing.

**Likely reasons**

1. **There is no semantic sky mask.** The regional illumination loss treats pixels that are both bright and high in the image as probable sky. That is only a heuristic. White buildings, signs, clouds, glare, and gaps around foliage can all be classified incorrectly.
2. **The regional loss is spatially coarse.** It pools luminance into only three horizontal bands and adds one upper-frame weighted mean. It can reward the correct average upper-frame brightness while still allowing many locally incorrect sky pixels.
3. **The discriminator is not source-conditioned.** It judges whether an output resembles some night image, not whether each output region is a plausible nighttime version of the corresponding input region.
4. **Training uses unpaired targets.** A random real night image provides target-domain statistics but no correct target sky for the input. The loss cannot say exactly which source pixels must become sky-colored.
5. **Random 256-pixel crops reduce global context.** A crop containing branches and a small patch of sky may not reveal whether the bright region is sky, a facade, a sign, or glare.
6. **Instance normalization removes per-image illumination statistics.** The network must reconstruct exposure and color information after repeatedly normalizing feature means and variances. This can encourage stylization instead of spatially faithful relighting.

### 3.2 Halos and red/cyan fringes around branches, wires, poles, and pipes

**Observed behavior**

- Thin dark objects against the sky develop colored outlines.
- Branches and wires become thicker, softer, or partially disappear.
- High-contrast boundaries show red, cyan, or orange bleeding.
- Small metal poles, traffic-light supports, signs, and railings may no longer have a clean one-pixel boundary.

**Likely reasons**

1. **Two stride-2 downsampling layers lose sub-pixel detail.** A 256 x 256 crop becomes a 64 x 64 bottleneck. A one- or two-pixel wire may be represented ambiguously or disappear before translation occurs.
2. **There are no long high-resolution skip connections.** The decoder cannot directly reuse the input's full-resolution edges. It must reconstruct them from compressed features.
3. **Transposed convolutions can produce uneven overlap.** The two `ConvTranspose2d` upsampling stages can introduce periodic texture, ringing, or checkerboard-like boundary artifacts.
4. **Sobel preservation is grayscale only.** It constrains luminance edge magnitude but cannot detect an edge that retains similar grayscale strength while changing to a wrong red/cyan color.
5. **DINOv2 supervision is global and resized to 224 pixels.** It protects broad semantic content, not individual branches, cables, characters, or tiny metal structures.
6. **The local PatchNCE implementation has no learned projection heads.** It encourages related generator features at matching locations. Whether learned projection heads would improve boundary correspondence is an untested hypothesis, not an established weakness by itself.
7. **Appearance and preservation gradients conflict.** The adversarial, color, and illumination objectives demand a large sky change. Cycle, identity, semantic, contrastive, and edge objectives demand preservation. At a thin foreground/sky boundary, both requirements act on nearly the same receptive field.
8. **Unpaired color-statistics matching can encourage tinting.** Matching the mean and standard deviation of a random night target can make a local region globally warmer, cooler, redder, or more cyan without respecting material boundaries.

### 3.3 The result sometimes looks like dark daytime or twilight

**Observed behavior**

- Some outputs are darker and warmer but do not contain a fully convincing night sky.
- Artificial light sources are inconsistent: a scene may become dark without gaining plausible lamp, window, or headlight behavior.
- Shadows and road surfaces often receive a global tone change rather than spatially varying illumination.

**Likely reasons**

1. The model has no explicit light-source, sky, shadow, depth, surface-normal, material, or reflection representation.
2. The adversarial objective only requires membership in the broad night-image distribution. A dark/tinted image can be an easier shortcut than correct scene-dependent relighting.
3. Cycle consistency rewards recoverability, not realism. Retaining daytime information makes reconstruction easier even when that information should change strongly.
4. Identity, semantic, edge, and contrastive losses can collectively discourage the large transformation required for a convincing night.
5. The BDD100K night domain itself includes varied exposure, twilight, artificial lighting, weather, windshield glare, and camera processing. “Night” is not one clean target style.
6. The generator is deterministic. It cannot represent several valid nighttime lighting configurations for the same daytime scene.

### 3.4 Global color casts and inconsistent local tone

**Observed behavior**

- Some images acquire a red/orange cast near buildings and traffic infrastructure.
- Cyan remains in sky holes or appears along edges.
- Objects that should keep approximately the same material color can shift differently depending on their surroundings.

**Likely reasons**

- The color loss matches whole-image statistics from an unrelated target image.
- Instance normalization discards feature statistics strongly related to color and illumination.
- A three-channel RGB output has no explicit separation between reflectance and illumination.
- The discriminator can accept a target-domain-looking tint even when it is locally attached to the wrong object.
- The Sobel loss sees grayscale structure and therefore does not penalize chromatic halos directly.

### 3.5 Periodic texture and reconstruction artifacts

**Observed behavior**

- Fine horizontal or grid-like texture is visible in some smooth skies, walls, and glare regions.
- Smooth regions can become locally noisy after translation.

**Likely reasons**

- Transposed-convolution overlap is a known architectural source of periodic artifacts.
- Training at 256 pixels and inferring at 512 x 288 or larger exposes texture patterns at a scale not directly optimized during training.
- Patch discriminators emphasize local texture realism and can reward high-frequency “night-like” texture even when a smooth region would be more correct.
- The camera characteristics of unrelated BDD100K images differ, so the model can mix scene translation with learned sensor/compression texture.

## 4. Why night-to-day is currently unusable

The five retained night-to-day examples consistently show the same pattern: the output is brighter, but it is not a recovered daylight photograph. Blue-gray haze spreads across the frame, saturated lamps remain or become larger, glare becomes more visible, and dark regions do not receive believable daylight texture.

### 4.1 The inverse problem is information-limited

A night sensor image may contain:

- clipped lamp and headlight pixels;
- shadows near zero with no recorded texture;
- motion blur and lens flare;
- shot noise and camera denoising artifacts;
- colored artificial illumination;
- reflections on the windshield or wet road.

No deterministic neural network can recover the exact daylight photons that were never captured. It can only synthesize a plausible estimate from learned priors. The current project should therefore not describe night-to-day as restoration of the true scene.

### 4.2 The training objective teaches distribution matching, not deglare or reconstruction

The night-to-day generator receives no aligned clean daytime target of the same scene. Its adversarial signal says “look like the daytime dataset,” while cycle loss says “retain enough information to map back to this night input.” Neither objective directly teaches removal of a particular flare, recovery of a particular shadow, or reconstruction of a saturated region.

### 4.3 The regional illumination heuristic is especially weak in this direction

For night-to-day, dark upper-frame source pixels are treated as likely regions to brighten. This cannot distinguish sky from a dark building, tree, tunnel, windshield border, or unlit object. A correct average upper-frame luminance can be achieved by adding a broad haze, which matches the observed results.

### 4.4 Random target statistics encourage a veil rather than spatially correct daylight

The color-statistics loss compares the generated image with an unrelated daytime image. Matching its mean and standard deviation can be achieved by lifting shadows and shifting the entire RGB distribution. This explains why outputs often look like a translucent blue/gray exposure layer rather than a relit scene.

### 4.5 Glare and saturation are amplified

The network has no explicit highlight mask, glare decomposition, point-spread-function model, RAW-domain exposure model, or inpainting stage. When global brightness is increased, the existing flare and saturated light blobs are increased as well. The model cannot infer clean structure beneath clipped highlights from unpaired supervision alone.

### 4.6 Symmetric CycleGAN assumptions do not fit the two directions equally

Day-to-night can often be approximated by removing exposure, shifting color, and adding darkness. Night-to-day requires inventing missing texture and suppressing camera artifacts. Treating the two directions with a nearly symmetric architecture and symmetric loss weights ignores this asymmetry.

### 4.7 The current evidence does not justify maintaining night-to-day as a showcase feature

Until a separate restoration-oriented method is trained and evaluated, night-to-day should be labeled experimental or removed from the main demonstration. Its current output is useful as evidence of a failed inverse mapping, not as a credible daytime reconstruction.

## 5. Architectural causes summarized

| Current design choice | Benefit | Current limitation |
|---|---|---|
| Two stride-2 encoder stages | Larger receptive field and efficient training | Thin structures can disappear at the 64 x 64 bottleneck |
| Nine ResNet blocks | Strong global transformation capacity | No explicit semantic or physical scene factorization |
| CBAM attention | Learns channel and spatial gates | Does not know sky, foliage, wires, lamps, glass, or road classes |
| InstanceNorm throughout generator | Stable batch-one style translation | Removes illumination/color statistics needed for faithful relighting |
| Transposed-convolution decoder | Learnable upsampling | Can create periodic, ringing, and boundary artifacts |
| No long high-resolution skips | Prevents trivial copying | Forces reconstruction of fine geometry from compressed features |
| Two PatchGAN scales | Local texture realism at two fields of view | Does not ensure pixel correspondence with the input |
| Global critic | Whole-frame target-domain pressure | One scalar cannot localize which region has incorrect lighting |
| Unconditional discriminators | Simple unpaired domain matching | Can reward any realistic night/day image, not faithfulness to this source |
| Cycle loss, weight 10 | Encourages scene recoverability | Can preserve source-domain appearance or hidden information |
| Identity loss, weight 2 | Reduces unnecessary changes | Can oppose the large illumination transformation |
| Simplified PatchNCE, weight 1 | Adds local content pressure | No learned projection heads; correspondence remains feature-dependent |
| DINOv2 loss, weight 0.75 | Broad semantic preservation | 224-pixel global representation misses fine boundaries |
| Grayscale Sobel loss, weight 2.5 | Preserves strong luminance edges | Misses chromatic fringes and can oppose legitimate new lighting edges |
| Color-statistics loss, weight 0.5 | Encourages target-domain color/exposure | Uses unrelated targets and can produce global casts |
| Regional illumination loss, weight 2 | Helps upper-frame/sky exposure | Three bands and brightness prior are not semantic segmentation |
| 256-pixel random crops | Fits the GPU and increases data variation | Loses large-scale context and does not train fine detail at demo resolution |
| Deterministic generator | Simple and fast | Cannot represent multiple plausible lighting outcomes |

## 6. Data and supervision problems

### 6.1 Road-domain bias

BDD100K is dominated by dashcam scenes: road at the bottom, sky near the top, and buildings, vegetation, vehicles, and signs near the horizon. The model can exploit this positional pattern. It has not earned a claim of general-purpose image relighting.

### 6.2 Unpaired ambiguity

The model never sees the correct night photograph corresponding to a training day photograph. It therefore cannot directly learn:

- which windows or lamps switch on;
- the true night sky color for that weather;
- which surfaces are wet or reflective;
- the correct shadow direction;
- how the same camera exposes the scene at both times.

### 6.3 Target-domain heterogeneity

Different night images have different cameras, exposure settings, weather, glare, traffic density, and artificial illumination. Matching this mixed distribution can produce an average style that is not physically coherent for any one input.

### 6.4 Lack of fine-detail labels

There is no explicit supervision for branch masks, wires, poles, glass, glare, sky boundaries, light sources, or reflection regions. General semantic and edge losses are expected to protect all of them indirectly, which is insufficient.

### 6.5 RGB images are a poor basis for true exposure recovery

BDD100K provides processed, tone-mapped RGB images rather than paired RAW captures. Gamma, white balance, clipping, denoising, sharpening, and compression have already changed the sensor signal. This especially limits night-to-day recovery.

## 7. Model-selection and evaluation problems

The selected step 4,500 checkpoint has the lowest recorded internal V2 score, but “best” does not mean best human-perceived night conversion.

The validation score combines cycle, edge, color, illumination, wavelet, and self-similarity terms. It has several limitations:

- it does not directly measure semantic sky completion;
- it does not measure colored boundary halos;
- it does not separately score branches, wires, glare, or saturated lights;
- it may prefer preservation even when a stronger illumination change would look more like night;
- it does not contain a paired ground truth for perceptual correctness;
- V1 and V2 scores are not necessarily comparable across changed formulas or settings;
- no complete retained CycleGAN baseline, FID/KID table, detector-retention study, or blind human test currently proves superiority.

This may explain why later steps can improve one component, such as cycle error, while looking worse overall.

## 8. Research questions to investigate

The following questions are intentionally phrased for literature and implementation research.

### 8.1 Sky and fine-boundary handling

1. Can an offline semantic segmentation teacher provide reliable sky, foliage, wire, pole, and building masks without being required at inference?
2. Would a dedicated low-frequency illumination branch plus a high-resolution reflectance/detail branch separate sky relighting from branch preservation?
3. Can boundary-aware supervision penalize chromatic halos without globally increasing edge loss and suppressing the night transformation?
4. Would a soft matting or alpha-boundary representation work better than a hard sky mask around leaves and wires?
5. Can a Laplacian/Gaussian pyramid be used only to separate coarse illumination from fine reflectance, avoiding V2.1's mistake of over-weighting all high-frequency preservation?

Suggested search terms:

```text
semantic guided day to night translation sky mask
image relighting intrinsic decomposition illumination reflectance
boundary aware image translation chromatic halo loss
thin structure preservation image-to-image translation wires branches
alpha matting sky replacement foliage boundary
Laplacian pyramid illumination reflectance relighting
```

### 8.2 Normalization and upsampling

1. How much of the color/exposure problem comes from generator InstanceNorm?
2. Can affine InstanceNorm, GroupNorm, LayerNorm, weight standardization, or no normalization preserve illumination statistics better?
3. Would nearest/bilinear resize followed by anti-aliased convolution remove periodic artifacts compared with transposed convolution?
4. Can partial high-resolution skips preserve geometry while preventing the network from copying the daytime sky unchanged?

Suggested search terms:

```text
instance normalization removes contrast illumination image translation
no normalization generator day night translation VGG16
anti-aliased upsampling checkerboard artifacts GAN resize convolution
selective skip connections relighting image translation
```

### 8.3 Source-conditioned realism

Important constraint: simply concatenating the source image with an unrelated real target is **not** a valid paired conditional-GAN solution. Random real pairs contain no correct scene correspondence. Any proposal here must explain where meaningful real conditions come from, or define a principled unpaired alternative.

1. Would a discriminator receiving `(source image, translated image)` penalize object movement and incorrect local correspondence better than a target-only discriminator?
2. Can semantic-region discriminators separately judge sky, road, vehicles, buildings, and vegetation?
3. Would a patch correspondence critic distinguish a realistic night patch placed on the wrong source object?

Suggested search terms:

```text
input conditioned discriminator unpaired image translation
semantic discriminator image-to-image translation region aware GAN
correspondence discriminator structure preserving translation
contrastive unpaired translation source conditioned critic
```

### 8.4 Better supervision

1. Can coarsely aligned day/night timelapse or repeated-route images supply confidence-weighted paired supervision?
2. Can optical flow, depth, and static-class masks exclude moving vehicles and occlusions from paired losses?
3. Would synthetic exposure and camera-pipeline pretraining teach controlled illumination before unpaired GAN refinement?
4. Can pseudo-paired data from a strong external model be used only as a labeled teacher experiment, with attribution and a locally implemented student?

Suggested search terms:

```text
aligned day night timelapse dataset image translation
coarse paired day night translation optical flow confidence
day to night neural ISP RAW paired dataset
synthetic low light camera pipeline Poisson Gaussian noise training
teacher student distillation day night image translation
```

### 8.5 Night-to-day as a separate restoration problem

1. Should night-to-day be separated from CycleGAN and treated as low-light enhancement plus deglare, denoising, highlight reconstruction, and color constancy?
2. What information can realistically be restored from sRGB, and what must be synthesized?
3. Would exposure-bracketed, paired RAW/sRGB, or burst data be necessary for credible restoration?
4. Can uncertainty or multiple stochastic outputs honestly represent missing shadow/highlight information?
5. Should the project focus only on day-to-night until a separately validated night-to-day method exists?

Suggested search terms:

```text
night to day translation low light restoration unpaired limitations
nighttime glare removal lens flare removal neural network
highlight reconstruction saturated image deep learning
RAW low light enhancement paired dataset SID LOL-v2
uncertainty stochastic inverse imaging missing information
```

## 9. Highest-priority controlled experiments

These are research directions, not yet validated solutions. They should be tested one at a time against the frozen V2 step-4,500 checkpoint.

1. **Normalization ablation:** compare the current InstanceNorm generator with no normalization, affine normalization, and GroupNorm while preserving as many trained weights as possible.
2. **Anti-aliased resize-convolution:** replace transposed convolution and measure periodic artifacts and boundary quality.
3. **Semantic sky conditioning:** use offline pseudo-labels and a soft sky/foreground boundary map. Apply the strong illumination objective only where confidence is high.
4. **Coarse/fine decomposition:** predict a smooth low-resolution illumination field and a bounded high-resolution residual. Do not allow the detail branch to control global exposure.
5. **Chromatic boundary loss:** measure color difference normal to sky/foreground boundaries, rather than increasing the global grayscale edge weight.
6. **Correspondence-aware realism:** investigate region critics or source-conditioned critics only after specifying valid supervision. Do not train a naive conditional discriminator using randomly paired real day/night images.
7. **Paired pilot:** add a small aligned or coarsely aligned subset with occlusion masking to determine whether direct supervision fixes the boundary ambiguity.
8. **Separate night-to-day study:** stop sharing the day-to-night objective design. Establish a restoration baseline before adding adversarial synthesis.

Each pilot should use the same fixed categories:

- open sky;
- dense branches against sky;
- thin wires and metal poles;
- buildings and signs near the skyline;
- vehicles and readable plates;
- glare and saturated lamps;
- wet or reflective roads;
- deep shadow;
- landscapes outside the road domain.

Record both the full image and magnified boundary crops. Stop a pilot early if it only changes global brightness, weakens night appearance, creates new colored fringes, moves objects, or worsens glare.

## 10. Measurements needed for a credible solution

No single score is sufficient. A future experiment should record:

- night/day domain confidence, calibrated on real held-out images;
- sky-region luminance and color distance using independently produced masks;
- boundary F1 and a chromatic-fringe metric around branch/sky and wire/sky edges;
- DINO or another semantic feature distance;
- detector consistency for vehicles, people, traffic lights, and signs;
- depth consistency;
- FID and KID on fixed sample counts, reported only as distribution measures;
- glare area and highlight-clipping changes for night-to-day;
- blind human ratings for realism, unchanged geometry, sky quality, fine-detail quality, and artifacts;
- latency and peak allocated/total GPU memory at the exact inference resolution.

The fixed V2 comparisons must remain unchanged during research. Otherwise it becomes easy to choose only examples that make a new method look better.

## 11. Claims we should not make

- “The attention module understands the sky.”
- “Cycle consistency guarantees that every object is preserved.”
- “Night-to-day recovers the real daylight scene.”
- “A higher resolution alone will fix branches and wires.”
- “More training steps must improve the model.”
- “The best internal score means the best visual result.”
- “The current model is general-purpose or state of the art.”

## 12. Concise problem statement for external research

> We have an unpaired CycleGAN-derived road-scene translator trained at 256 pixels. Its current day-to-night generator produces a convincing broad night style on some images, but it incompletely changes sky visible through foliage and creates red/cyan halos or loss of detail around branches, wires, poles, signs, and other thin structures. The generator downsamples twice to a 64 x 64 bottleneck, has no long high-resolution skips, uses InstanceNorm and transposed-convolution upsampling, and is supervised by target-only local/global critics, grayscale Sobel edges, global DINO features, simplified PatchNCE, unpaired color statistics, and a heuristic upper-frame illumination loss rather than semantic masks. We need a controlled improvement that strengthens spatially coherent sky relighting while preserving fine foreground boundaries, without merely increasing detail loss and weakening the night transformation as our V2.1 ablation did. The night-to-day generator is currently unusable: it mainly brightens images, amplifies glare, and creates blue-gray haze because unpaired sRGB training cannot recover clipped highlights or missing shadow detail. We need research on whether night-to-day should become a separate low-light restoration/deglare problem with paired or RAW supervision. Proposed methods must be evaluated against the frozen V2 step-4,500 checkpoint on fixed sky, branch, wire, glare, vehicle, landscape, and deep-shadow cases.

This paragraph can be supplied directly to another research system together with the architecture and configuration files.

## 13. Visual evidence inspected for this document

The following ten comparison images were opened and visually inspected. Each has the input on the left and V2 output on the right. These are inference examples, not an independently scored benchmark; five images per direction cannot establish a population-wide failure rate. Links are relative so the document remains usable when shared with its evidence folder.

Evidence root: `artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/`. Its `evidence.json` records checkpoint and input hashes, resolution, and inference setup. The comparisons are 512 × 288 per panel. At that resolution, a distant branch or cable may occupy only one or two pixels.

| Direction / example | Visible problem | What it demonstrates, without claiming causality |
|---|---|---|
| [Day→night 01: wires/intersection](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/day_to_night/01_01cce42a-cd2acad5_comparison.png) | Bright sky strips at the top; cyan/red fringing around wire/sky and roof boundaries; periodic sky texture. | Global darkening does not ensure coherent sky conversion or clean thin boundaries. |
| [Day→night 02: foliage/overpass](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/day_to_night/02_27f635b5-0893ff34_comparison.png) | Conspicuously bright building/overpass; sun-like mottled shadows remain on the road. | Dark sky can coexist with daylight illumination cues on the ground. |
| [Day→night 03: tree-lined street](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/day_to_night/03_281e05c2-d80ce328_comparison.png) | Bright sky holes beside foliage; luminous left facade; warm, patterned sky. | Broad night palette does not imply consistent local lighting. |
| [Day→night 04: windshield reflections](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/day_to_night/04_42f8d08e-f771524e_comparison.png) | Bright canopy gaps; reflected car/windshield overlay survives; very dark regions. | Translation does not separate scene geometry from camera reflections. |
| [Day→night 05: bare branches/SUV](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/day_to_night/05_798ed431-097c00d5_comparison.png) | Open sky darkens but tree gaps remain cyan/white; colored branch fringes and bright facade/window edges. | Clearest sky/foreground separation failure. |
| [Night→day 01: yellow taxi](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/night_to_day/01_01638325-504ab42d_comparison.png) | Blue haze; red brake-light glow and clipped rear bumper remain. | Brightening does not reconstruct daytime appearance or suppress artificial lights. |
| [Night→day 02: intersection](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/night_to_day/02_1accee17-60f12c98_comparison.png) | Broad cyan/white upper-frame patches; lamps stay on; gray road. | Brightening is misplaced rather than a believable daylight sky. |
| [Night→day 03: tilted street](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/night_to_day/03_8498af74-b8a12962_comparison.png) | Milky veil, green/cyan canopy/building patches; windshield streaks remain. | Camera artifacts are transformed or amplified rather than reliably removed. |
| [Night→day 04: bus](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/night_to_day/04_a2319601-c5ec1f7a_comparison.png) | Huge bright cyan patch above/left of bus; dark vague facades. | Global exposure correction can hide severe local relighting errors. |
| [Night→day 05: lamp glare](../artifacts/share/Day_Night_Finalized_Handover_2026-10-18/model_validation_evidence_final/night_to_day/05_b251b746-00138418_comparison.png) | Glare expands into washed-out upper frame; vehicle lighting remains nighttime. | Brighter night is not convincing day. |

Small poles, traffic-light supports, and overhead cables are present. **A specific close-up metal-pipe example was not supplied or isolated here.** Similar failure on pipes against sky is plausible but needs a dedicated example, not a fabricated observation.

## 14. Why a sky pixel beside a branch is not a simple sky pixel

Camera blur, antialiasing, resizing, and compression mix foreground and sky into boundary pixels. An approximate explanatory model in linear light is:

```text
observed_pixel = alpha * foreground + (1 - alpha) * daytime_sky
desired_pixel  = alpha * relit_foreground + (1 - alpha) * nighttime_sky
```

Here `alpha = 1` means solid branch/pipe and `alpha = 0` means pure sky. Partial coverage needs a fractional value. This is not the current V2 computation.

Three distinct problems follow:

1. **Classification:** identify tiny sky openings inside trees, not only open sky above them.
2. **Matting:** estimate partial foreground coverage. A binary mask cannot represent all mixed pixels.
3. **Color decontamination:** remove the old sky contribution before inserting darker sky. Copying the source boundary retains a daytime halo; aggressively darkening it destroys the branch.

V2 must learn these implicitly from unpaired images with no alpha target or corresponding clean nighttime boundary. This is a plausible explanation for the spatial pattern, not proof that arbitrary segmentation will fix it. Coarse masks can erase branches or leave the same holes.

### Tests needed to establish causes

| Hypothesis | Controlled test | Supporting evidence |
|---|---|---|
| Coarse illumination supervision misses sky holes | Annotate sky/mixed boundaries in a small validation subset with V2 frozen. | Errors concentrate in holes despite acceptable average sky brightness. |
| Sampling loses thin geometry | Compare registered crops across resolutions and small input pixel shifts. | Branch continuity or halos change sharply with scale/phase. |
| Decoder contributes grid texture | Adapt a resize-convolution decoder or compare phase-averaged predictions against frozen V2. | Periodic texture decreases without weakening night transformation. |
| Preservation losses resist relighting | Change one region-specific term at a time with fixed data/seed/schedule. | Sky improves without geometry regression. |
| Color matching contributes to haze | Warm-start otherwise identical pilots with/without that term. | Haze/casts improve across fixed cases, not one selected image. |
| Normalization discards useful cues | Controlled layer-specific adaptation with initialization checks. | Lighting and boundary ratings improve at matched geometry, not merely darkness. |
| Night→day needs dedicated supervision | Compare an appropriate paired restoration/deglare baseline. | Less glare and credible visible-region detail, rather than brightness alone. |

These tests are proposals, not completed experiments. A transposed-convolution layer exists, but that does not prove it causes every stripe. InstanceNorm exists but is not automatically the main culprit. No loss-weight ranking is justified without gradient measurements or ablations.

## 15. Separate experimental sky candidate: remaining flaws

`configs/sky_candidate.yaml` points to a V2-derived RGB generator plus a separate sky/vegetation matte predictor. Inference adds phase averaging, image-guided matte refinement, and an analytic sky compositor. See `daynight/sky_refinement.py`, `daynight/sky_pipeline.py`, and [the experiment record](SKY_REFINEMENT_RESEARCH_AND_EXPERIMENTS.md).

The [five-scene candidate comparison gallery](../outputs/sky_pilot/region_candidate_test_audit/comparison_gallery.jpg) was also inspected. Many previously bright branch gaps are darker. This is **not evidence that general relighting is solved**:

- Facades, windows, road markings, and vehicle highlights can remain overbright or artificially colored.
- The foliage/overpass example still shows daylight-like road shadows under dark sky.
- The windshield-reflection example becomes very dark without convincing spatially motivated artificial lighting.
- A prescribed sky color/gradient is not a learned estimate of the scene's true nighttime atmosphere.
- Matte and foreground-color estimation errors can create new boundaries; a single image does not uniquely determine the unmixed colors.
- Night→day remains the old V2 mapping, unchanged by this candidate.

These images have been inspected and must not later be presented as untouched blind test data for decisions influenced by this inspection. Reserve additional unseen images for final evaluation. Research should compare against both frozen V2 and the separately labeled hybrid, rather than propose sky compositing as though it had never been tried.

## 16. What to ask an external researcher

Share sections 1, 12–15, the linked images, and model/configuration files. Request:

1. Ranked diagnoses separating observations, code facts, and hypotheses.
2. Primary papers and maintained implementations, explaining specifically which visible defect each mechanism addresses.
3. A first experiment reusing V2 weights: compatible layers, changed layers, initialization, and safeguards against repeating V2.1's weak night appearance.
4. Separate approaches for boundary matting, scene-wide shadows/illumination, and night→day restoration—not one brightness fix.
5. Exact losses and supervision, avoiding daytime-sky preservation and bogus paired targets made from unrelated images.
6. Memory estimates for a 12 GB RTX 4070 Super, storage needs, and external-weight requirements at training versus inference.
7. Controlled ablations and rejection criteria. Darkness alone or grayscale-edge preservation must not count as success.

Preserve existing checkpoints and dataset splits. Describe architecture adaptation as warm-start training, not exact optimizer resumption. Do not promise recovery of unknowable light states or missing texture. This documentation update changes no weights and starts no training.
