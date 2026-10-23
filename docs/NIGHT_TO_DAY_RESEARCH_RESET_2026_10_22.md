# Night-to-day research reset: restoration is not relighting

Date: 22 October 2026. Status: research recommendation, NOT a new implemented or accepted model.

## Decision

Do not extend the latest noisy timelapse branches merely because their training loss falls. Preserve the existing trained 16k inverse as the comparison/warm-start parent. The next direction should separate degradation restoration from daylight translation, with reliable real supervision wherever available. Do not train a diffusion foundation model from scratch, substitute FLUX at inference, or alter the finished day-to-night demo.

This is a focused review of the primary sources below, not a claim to have exhaustively read hundreds of papers or reproduced them. Some directions were already known; the table distinguishes new actionable leads from previously tested approximations.

## What the actual implementation and outputs establish

The retained inverse is `ResnetGenerator` in `daynight/models/networks.py`: base width 64; two stride-2 downsampling layers; nine residual blocks with CBAM; InstanceNorm; two transpose-convolution upsampling layers; tanh RGB head. `daynight/experiment_night_to_day.py::inverse_from_payload` loads the existing EMA inverse weights. This is a feed-forward GAN-derived generator, not a diffusion model. The recent supervised continuation need not have an active discriminator: the pilot's adversarial loss defaults to zero.

Parent: `E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt`, global step 16,000. This is a conservative experimental parent, not an accepted night-to-day release.

The recent continuation objective is dominated by RGB imitation of 111 audited external CycleGAN-Turbo targets. Auxiliary experiments add real photographic pairs with partial masks. They do not establish genuine full-scene daytime ground truth for BDD sources.

Actual latest comparison inspected again during this research:
`outputs/night_to_day_experimental/20261022_timelapse31_rgb_vs_coarse1000/comparison_08.jpg`.
VAL84 retains strong rear-car illumination; VAL318 retains red lamp glow and dark upper foliage; VAL441 retains a huge washed-out flare; VAL358 has grainy foliage/sky and headlight-shaped road illumination. Both RGB and coarse timelapse branches fail to remove the recurring patterns. The preceding visual-review log records the complete 24-case reviews; this research pass did not independently reopen all 24.

Observed versus hypothesized causes:

| Observation | Plausible explanation requiring isolation |
|---|---|
| Dark areas become grainy when lifted | Input sensor/JPEG corruption may be amplified; newly synthesized texture may also contribute. Not all visible noise is necessarily sensor noise. |
| Night lamps and road light pools survive | Pseudo-targets retain them; source-preservation constraints can preserve illumination rather than geometry. |
| Sky holes and overhead mistakes persist | Ambiguous dark structure, insufficient scene reasoning, and imperfect supervision. |
| Lower paired-region loss does not improve BDD daylight realism | Nine static cameras, limited domain coverage, imperfect correspondence, and dominant pseudo-target anchors. |
| Some outputs have periodic speckles/striping | Decoder/normalization/optimization artifacts are candidates, not proven causes. |

InstanceNorm normalizes feature statistics; it is not correct to assert that it always creates noise. Likewise, transpose convolution is not proven to be our sole failure. A matched approximate resize-decoder continuation already failed to give a general improvement.

## Primary-method comparison

### 1. Retinexformer: illumination-guided corruption restoration

The method explicitly accounts for corruption hidden in darkness and amplified by lighting enhancement, and uses illumination-guided interactions. [ICCV 2023 paper](https://arxiv.org/abs/2303.06705), [official implementation](https://github.com/caiyuanhao1998/Retinexformer).

**Use here:** motivate a separately supervised restoration path, rather than asking one RGB translator to simultaneously denoise and invent daylight. **Limit:** low-light enhancement does not by itself remove nighttime illumination or produce daytime skies. An official checkpoint would be an attributed external restoration baseline, not our trained model. Replacing our backbone with it would not preserve our learned weights.

### 2. N2D-LPNet: separate low-frequency lighting from high-frequency detail

This night-to-day method uses a Laplacian-pyramid architecture and progressive guidance between frequencies. [CVPR Workshops 2023 primary paper](https://openaccess.thecvf.com/content/CVPR2023W/UG2/papers/Fan_Learning_To_See_in_Nighttime_Driving_Scenes_With_Inter-Frequency_Priors_CVPRW_2023_paper.pdf).

**Use here:** strongest architecture lead compatible with retaining our generator as a learned component: estimate broad daylight appearance separately from confidence-gated detail. **Limit:** simply copying source high-frequency bands would also copy noise, halos and headlight edges. Our existing average-pooling loss is NOT this architecture, and smaller-resolution inference was NOT a reproduction. No author checkpoint or complete runnable reproduction was verified in this review.

### 3. N2D3: preserve content differently in different degradation regions

N2D3 uses physics-informed degradation disentanglement and region-aware contrastive learning instead of treating every source feature as reliable content. [Primary paper](https://arxiv.org/html/2411.14504v1).

**Use here:** do not preserve a lamp halo as though it were a building boundary. Preserve observable structure while allowing unreliable illumination features to change. **Limit:** our thresholded glare downweighting already failed and is not N2D3. Brightness thresholds also select white cars/signs and miss broad road pools. No ready author weights were verified; do not describe this as a drop-in fix.

### 4. HVI / CIDNet: decouple chromatic information and intensity

HVI introduces a low-light-oriented color representation; CIDNet separates color and intensity processing. [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html), [official repository](https://github.com/Fediory/HVI-CIDNet).

**Use here:** alternative representation if our restoration branch creates colored speckles under large exposure changes. **Limit:** it is enhancement, not demonstrated replacement of nighttime lighting by daylight. Transforming our entire pretrained backbone into this representation is not a function-preserving warm start. Consider only as a separate controlled adapter experiment, not another simultaneous change.

### 5. NAFNet and Restormer: restoration controls, not daytime generators

NAFNet offers a relatively simple restoration architecture; Restormer provides efficient long-range restoration modeling. Both have official implementations and denoising checkpoints. [NAFNet paper](https://arxiv.org/abs/2204.04676), [NAFNet code](https://github.com/megvii-research/NAFNet), [Restormer paper](https://arxiv.org/abs/2111.09881), [Restormer code](https://github.com/swz30/Restormer).

**Use here:** reference baselines to test whether removing corruption before translation improves the retained generator. **Limit:** denoisers trained on other camera distributions can erase wires/texture, and neither resolves invisible scene content or turns off headlights. Do not quietly chain pretrained restoration into the demo and call the whole pipeline locally trained.

### 6. HiDT: content/style separation and high-resolution translation

HiDT is relevant to separating scene content from time-of-day appearance. Its official repository explicitly provides the inference implementation, not a complete training release. [Paper](https://arxiv.org/abs/2003.08791), [official repository](https://github.com/advimman/HiDT).

**Use here:** longer-term architectural reference for appearance conditioning. **Limit:** not a cheap continuation of our checkpoints, nor evidence of solving our severe night-driving glare. Lower priority than degradation handling and trustworthy targets.

### 7. What the user's VGG16/no-normalization example actually changes

The inspected UNIT configurations in `solesensei/day2night` set `vgg_w: 1`; the compared BDD/NEXET configurations change `dis.norm` from `none` to `in`. The trainer computes VGG feature loss using instance-normalized feature maps. Thus the repository does NOT support the inference that the screenshot proves removing every generator normalization layer will fix our inverse.

[No-normalization config](https://github.com/solesensei/day2night/blob/master/day2night/UNIT/configs/unit_day2night_bdd_nexet_vgg.yaml), [IN config](https://github.com/solesensei/day2night/blob/master/day2night/UNIT/configs/unit_day2night_bdd_nexet_vgg_in.yaml), [trainer](https://github.com/solesensei/day2night/blob/master/day2night/UNIT/trainer.py).

The compared configurations also differ in logging/display frequency; this is code inspection, not a forensic identification of the exact screenshot's training checkpoint. Those source files carry NVIDIA CC BY-NC-SA notices: inspect applicable licensing before reusing code. VGG is a loss network here, not the entire image generator.

### 8. DarkDriving: potentially more consequential than another loss

The ICRA 2026 paper reports 9,538 real day/night pairs collected with pose matching in a closed test field. The author repository lists dataset download links and a train/test split. This is a materially different supervision opportunity from our coarse Dark Zurich correspondences. [Paper](https://arxiv.org/abs/2603.18067), [author data page](https://github.com/DriveMindLab/DarkDriving-ICRA-2026).

Important limitations: reported centimeter-level pose alignment is not a guarantee of exact pixel correspondence; restricted test-field diversity may limit BDD transfer. Download availability, archive size and actual image quality have NOT been tested. The academic license restricts use and transfer, including derivatives; do not put downloaded data or example derivatives into public demo/GitHub artifacts without clarifying permission. [Dataset license](https://github.com/DriveMindLab/DarkDriving-ICRA-2026/blob/main/LICENSE). User's earlier dataset-source preference was Kaggle-only, so an author-hosted download needs a new source decision; no download was started.

## Recommended experiment sequence — proposed, not executed

### A. Isolate the noise failure before introducing a new network

Use the SAME retained 16k generator and fixed development inputs. Compare unchanged input against mild deterministic denoising and a labeled external restoration baseline only if its weights/source are approved. Also inspect repeatability and small, controlled input perturbations. Save native-resolution crops of flat sky, foliage, cable boundaries, road and saturated lamps. This separates input sensitivity from periodic generator artifacts without claiming causality from appearance alone.

Reject smoothing if it merely hides speckles by erasing boundaries, or makes relighting worse. A clean-looking night image still fails the task. This is a diagnostic, not a proposed final preprocessing filter.

### B. Obtain better supervision before expanding training

Audit whether DarkDriving can be lawfully and practically used, or obtain an approved alternative with genuinely aligned night/day coverage. Preserve author test split; reserve validation by route/sequence rather than adjacent random frames. Verify registration, scene diversity, lamps, saturation and black-region observability before treating pairs as targets. Keep our BDD test and 12-image lockbox untouched.

### C. One warm-start architecture change

Working hypothesis: separate corruption handling and scene-scale lighting from reliable spatial detail. Preserve all existing generator weights. Add a small residual restoration adapter initialized to identity with a zero output layer, then test it alone before a frequency-aware lighting branch. This is our proposed integration inspired by the papers, NOT a reproduction or claimed new state of the art.

The first learned adapter must have suitable clean/noisy training supervision; arbitrary blur targets or another unvalidated GAN teacher would not establish restoration quality. Freeze the parent first to isolate adapter behavior, then cautiously fine-tune if it passes. Any later global/low-frequency branch should begin with zero residual influence so checkpoint migration initially reproduces parent output. Do not blindly copy noisy source detail.

Avoid simultaneously adding HVI, global attention, a new decoder, new normalization and new adversarial objectives. That would make success or failure uninterpretable and risk losing learned behavior.

### D. Acceptance and compute gates

Compare against retained 16k and the saved matched 17k teacher-only control; if training resumes, run a new matched continuation control for the same new data and update budget. Use the same 24 development sources and review every output, including native crops. Require cleaner flat regions WITHOUT lost wires/branches, plausible daytime skies WITHOUT holes, reduced lamp/road illumination, preserved objects and convincing whole-image daylight. Record subjective ratings separately from teacher imitation and image statistics. Keep the existing 20/24 whole-image gate; do not lower it to obtain a success.

Only evaluate the reserved lockbox after selecting a candidate. Probe memory before longer training; no model-size or latency promises are justified before measurement. Never promote a failed branch simply because it runs quickly or loss decreases.

## State after this research

- No new training, weights download, architecture modification or deployment performed in this research pass.
- Existing experimental checkpoints preserved; 16k inverse remains the warm-start parent, not a quality-approved release.
- Public day-to-night weights, sky repair, model selection and remote worker untouched.
- Recommendation: restoration/translation separation plus better paired supervision; not diffusion-from-scratch and not another unconstrained image editor.

## Follow-up diagnostic executed (after the research pass)

The proposed frozen-generator filter/precision check is now executed, not pending:
`python -m daynight.audit_inverse_noise --output <new-evidence-directory>`.
See `NIGHT_TO_DAY_VISUAL_REVIEW.md` for review scope and all paths. All24 filter
comparisons were opened; mild bilateral/Gaussian preprocessing and post-filtering
do not fix the recurring lighting/sky failures. Synthetic1/255inputperturbations
produce mean output/inputRMSratios4.67(BF16) and4.60(FP32), while unchanged-input
repeats are identical in this diagnostic. Three additionalFP32nativeimages were
opened and remain visibly flawed. Neither blur nor precision is a complete fix;
the experiment does not establish that a particular layer creates the noise.
No training or promotion occurred.118tests and relevantRuffpass.
