# LumiCycle: Structure-Aware Unpaired Day–Night Image Translation

## Abstract

Day–night appearance changes can reduce the reliability of computer-vision systems used in driving and surveillance. Collecting perfectly aligned photographs of the same dynamic scene during day and night is expensive and often impossible. This project implements unpaired bidirectional image translation using BDD100K. It first reproduces CycleGAN as a controlled baseline and then introduces LumiCycle, which combines attention, multi-scale adversarial learning, patchwise contrastive learning, semantic feature consistency, and edge preservation. The project also improves experimental rigor by preventing train/test leakage, saving reproducible training state, and evaluating distribution similarity, structural fidelity, object retention, latency, and failure cases. A local Gradio application demonstrates Day → Night and Night → Day inference. Numerical results must be inserted from the final untouched test run; they are intentionally not fabricated in this document.

## 1. Motivation

Outdoor systems experience a large domain gap between daylight and night. At night, reduced illumination, sensor noise, glare, artificial light colors, and motion blur can hide or distort visual evidence. A translator can support research in data augmentation and domain adaptation, but a realistic output is not necessarily a truthful output. For safety-related imagery, the project therefore treats object and structure preservation as first-class requirements.

The selected IEEE GCCE 2025 paper compares CycleGAN and CycleGANAS using unpaired BDD100K day/night images. It reports better qualitative output from architecture search, especially in low-light scenes. The paper provides a useful starting point but leaves three opportunities:

1. Its architecture-search phase mixes training and test data, which weakens the validity of the final evaluation.
2. The comparison is primarily visual and does not establish semantic or downstream preservation.
3. Full neural architecture search is costly and unsuitable for a two-week project on a single 12 GB GPU.

LumiCycle preserves the paper’s unpaired, bidirectional objective while replacing test-set mixing with a strict split and replacing expensive NAS with targeted, explainable architectural and loss improvements.

## 2. Objectives

- Train Day → Night and Night → Day mappings without paired photographs.
- Reproduce a standard CycleGAN baseline from local source code.
- Improve visual realism while preserving lanes, vehicles, signs, silhouettes, and layout.
- Resume training safely across overnight sessions.
- Compare all local models using the same untouched test data.
- Provide a simple offline upload-and-translate application.
- Explain limitations and external dependencies honestly.

## 3. Dataset and Experimental Integrity

BDD100K provides road-scene imagery and attributes including time of day. The preparation tool recursively discovers official images and labels and accepts only the exact `daytime` and `night` attributes. It verifies image files, computes quality indicators, and creates balanced domain manifests.

The default split is:

| Split | Day | Night | Used for |
|---|---:|---:|---|
| Train | 5,000 | 5,000 | Gradient updates |
| Validation | 500 | 500 | Progress and checkpoint selection |
| Test | 1,000 | 1,000 | One final comparison |

A 64-bit difference hash groups exact and near-duplicate images before assignment. Every group belongs to one split only. This prevents visually repeated frames from inflating validation or test performance. Corrupt images are rejected. Blur and exposure flags are reported for review; the program does not silently curate the test set around favorable samples.

## 4. Baseline

The baseline follows CycleGAN:

- Two nine-block ResNet generators.
- Two 70×70 PatchGAN discriminators.
- Least-squares adversarial loss.
- L1 forward and backward cycle consistency.
- Identity loss to discourage unnecessary changes.
- Historical fake-image replay buffers.

For day image \(x\) and night image \(y\), the generators learn \(G_{DN}(x)\) and \(G_{ND}(y)\). Cycle reconstruction encourages:

\[
G_{ND}(G_{DN}(x)) \approx x, \qquad G_{DN}(G_{ND}(y)) \approx y.
\]

This is necessary because adversarial distribution matching alone does not force an output to retain the input scene.

## 5. Proposed LumiCycle Method

### 5.1 Attention generators

Each bottleneck residual block applies channel attention followed by spatial attention. Channel attention emphasizes useful feature types, while spatial attention emphasizes relevant image regions. This is intended to handle spatially non-uniform lighting such as headlights, sky, signs, and road reflections.

### 5.2 Multi-scale spectral discriminators

LumiCycle uses two PatchGAN scales. The higher-resolution discriminator focuses on local texture and light artifacts; the downsampled discriminator captures broader illumination. Spectral normalization limits discriminator sensitivity and reduces unstable adversarial updates.

### 5.3 Patch contrastive consistency

PatchNCE associates a translated feature at each sampled location with the corresponding source location and treats other locations as negatives. This discourages spatial rearrangement without requiring paired target images.

### 5.4 Semantic consistency

A frozen DINOv2-S encoder compares source and translated embeddings with cosine distance. The DINO weights are never updated. The feature prior penalizes large semantic changes while still allowing illumination to change.

### 5.5 Edge consistency

Sobel gradients are compared between source and translation. This provides a direct signal for road boundaries, object contours, poles, signs, and the skyline.

### 5.6 Complete objective

The default generator weights are:

| Term | Weight |
|---|---:|
| Adversarial | 1.0 |
| Cycle | 10.0 |
| Identity | 5.0 |
| PatchNCE | 1.0 |
| DINO semantic | 0.5 |
| Sobel edge | 2.0 |

These values are fixed in configuration files. Any changed experiment must preserve its resolved configuration beside the checkpoint.

## 6. Training and Plateau Control

Images are resized, randomly cropped to 256×256, and horizontally flipped. Training uses batch size one, two-step gradient accumulation, BF16 autocasting, gradient clipping, separate generator/discriminator learning rates, differentiable augmentation, replay buffers, and EMA generator weights.

Every checkpoint atomically records networks, EMA, optimizers, schedulers, replay buffers, random states, configuration identity, and monitoring state. `--max-hours` allows the PC to stop after an overnight window. `--resume auto` continues from the latest complete checkpoint.

TensorBoard records individual losses, discriminator accuracy, gradient norms, output variance, learning rates, VRAM, validation scores, and fixed translation grids. The trainer identifies common warning patterns:

- Non-finite losses or gradients.
- CUDA memory failure.
- Near-zero generated-image variance.
- Discriminator accuracy saturation.
- Four consecutive non-improving validations.

The failure state is written to `diagnostics`. Recoverable failures roll back to a good checkpoint. A plateau restores best weights and halves learning rates. These controls reduce common failure modes but cannot mathematically guarantee that every GAN run converges.

## 7. Evaluation Protocol

Unpaired data does not provide a pixel-aligned ground-truth target. Consequently, PSNR and SSIM are not primary metrics.

The final evaluation reports:

- **FID/KID:** similarity between generated and real target-domain distributions.
- **DINO distance:** semantic/structural difference between input and output.
- **Sobel distance:** edge preservation.
- **Detector retention:** fraction of frozen-detector objects matched after translation using class and IoU.
- **Latency and VRAM:** practical performance on the RTX 4070 Super.
- **Failure categories:** glare, deep shadow, small objects, signs, reflective roads, and out-of-domain photographs.

CycleGAN, the attention-only ablation, and LumiCycle are trained locally. CycleGAN-Turbo is an optional externally pretrained benchmark and must remain labeled as such.

## 8. Results

Populate this section only after running `scripts/evaluate_all.ps1`. Copy the completed table from `RESULTS_TEMPLATE.md`, add fixed comparison grids, and state the exact checkpoint steps.

The main claim is accepted only if LumiCycle improves target-domain FID/KID over the local CycleGAN and improves at least one structure/object measure without a meaningful object-retention regression. If the experiment does not satisfy this condition, report the negative result and analyze it rather than changing the test set or hiding failures.

## 9. Gradio Application

The application exposes image upload, webcam, clipboard input, direction selection, model selection, maximum resolution, output download, before/after comparison, and runtime metadata. LumiCycle is selected by default. The application loads EMA weights, corrects EXIF orientation, preserves aspect ratio, pads to valid dimensions, and gives actionable missing-checkpoint or memory messages.

All local-model inference works without internet after checkpoints and DINO caches are prepared. The optional Turbo choice is isolated so its absence cannot break LumiCycle.

## 10. Limitations and Ethics

- Translation is a plausible rendering, not recovered ground truth.
- Information absent from a dark input cannot be reconstructed reliably.
- Small objects or signal colors may be modified even when average metrics improve.
- BDD100K does not represent every country, camera, weather condition, or road type.
- Generated images must not be used directly for vehicle control, evidence, or surveillance decisions.
- BDD100K and external model licenses must be respected; dataset images are not redistributed.

## 11. LumiRender: Physics-Guided Successor Experiment

Later experiments showed that improving CycleGAN losses alone could not reliably produce plausible skies, tree-branch boundaries, artificial-light placement and road reflections. V2.1 was especially informative: its Laplacian/detail constraints preserved fine structure but weakened the required global lighting change. It is therefore reported as a failed ablation rather than hidden.

LumiRender restarts from random weights and reframes the task as image formation. A scene factorizer predicts reflectance, daylight illumination, depth, normals, roughness, wetness, semantic regions and emitter candidates. Eight anisotropic 2D Gaussian lights plus a smooth ambient/horizon field illuminate the inferred scene. A differentiable image-space renderer adds diffuse/specular terms, road/glass reflections, wet-road streaks and multi-scale bloom. Exposure, white balance, tone mapping, vignetting and Poisson–Gaussian noise form the camera stage. A high-pass correction is bounded to ±0.03 linear RGB.

Frozen Depth Anything V2 and Cityscapes Mask2Former models generate offline pseudo-labels; RAFT aligns coarse licensed pairs and creates forward/backward confidence masks. These teachers are absent from inference. The model trains in four stages totaling 28,000 optimizer steps. It is accepted only after the fixed difficult-scene suite, automatic metrics and blinded team review all pass. Until then, LumiCycle V2 remains the showcase default.

The 245-record reproducible abstract screen and 30-paper design matrix are supplied in `docs/LUMIRENDER_LITERATURE_SCREEN.csv` and `docs/LUMIRENDER_LITERATURE_REVIEW.md`. This architecture is a college research contribution, not a state-of-the-art claim and not recovery of the unknowable true night state.

## 12. Conclusion

LumiCycle turns the selected paper into a reproducible baseline, while LumiRender tests a more fundamental hypothesis: physically constrained factors and rendering should preserve scene geometry better than unconstrained RGB style translation. The implementation is complete, but the hypothesis must be accepted or rejected using the planned training runs and untouched evaluation—not asserted from architecture alone.

## References

1. I. Goodfellow et al., “Generative Adversarial Nets,” 2014.
2. J.-Y. Zhu et al., “Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks,” ICCV, 2017.
3. T. Park et al., “Contrastive Learning for Unpaired Image-to-Image Translation,” ECCV, 2020.
4. F. Yu et al., “BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning,” CVPR, 2020.
5. M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” 2023.
6. G. Parmar et al., “One-Step Image Translation with Text-to-Image Models,” 2024.
7. R. Hara and Q. Chen, “Unpaired Day-to-Night Image Translation Using Deep Generative Model,” IEEE GCCE, 2025.
8. M. S. Alam, P. Singh, and P. Bazilinskyy, “A Survey of Day-Night Illumination Domain Translation for Outdoor Vision,” 2026.
