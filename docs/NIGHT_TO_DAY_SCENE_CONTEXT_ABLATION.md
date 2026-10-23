# Experimental inverse: spatial scene-context ablation

22 October 2026. Implemented and trained, but NOT accepted for deployment.

## Hypothesis and implemented architecture

The original inverse has local convolutional spatial processing plus CBAM.
Global channel pooling is not the same operation as exchanging spatial tokens
across the whole scene. This experiment adds such exchange without replacing
the trained convolutional backbone. General motivation: long-range feature
aggregation in [Non-local Neural Networks](https://arxiv.org/abs/1711.07971).
Our pooled-attention implementation is NOT an exact reproduction of that paper
or proof of semantic understanding/nighttime relighting.

`daynight/inverse_scene_context.py` implements:

1. Existing base64, two-downsample, nine-CBAM-ResNet inverse encoder unchanged.
2. After its last residual block, pool the 256-channel bottleneck to an8x8grid.
3. Project to128channels and flatten to64spatialtokens.
4. Add a learned projection of normalized2Dcoordinates.
5. LayerNorm, four-head self-attention without dropout, residual connection.
6. LayerNorm,128→256→128GELUfeed-forward residual.
7. Project128→256, bilinearly upsample to original bottleneck resolution, and add
   to the bottleneck features.
8. Existing transpose-convolution decoder and tanhRGBhead unchanged.

The adapter's final projection weight AND bias start at zero. All previous
backbone tensors are copied exactly; no convolutional weights discarded and
no approximate decoder conversion. Local initializer seed2034 is isolated so
the main training RNG is not advanced by constructing the adapter. Training
uses the existing full backbone plus the new adapter, not a frozen-backbone-only
fit. This is a nondiffusion, single-forward image translator.

Measured parameter counts:11,447,541original +198,784adapter =11,646,325total.
The output residual is not geometrically constrained after training: attention
can still change appearance incorrectly. The8x8representation is coarse; it is
not a dense sky mask or guarantee of thin-detail preservation.

## Migration and loading evidence

- CPU tests verify exact initial tensor/output equality, unchanged main RNG,
  finite gradients and nonzero gradient to the zero-initialized projection.
- A nonzero test projection allows a far-corner input change to influence the
  opposite corner: the spatial-exchange path is genuinely connected.
- Loader roundtrip tests preserve all context weights and metadata.
- Unsupported resize decoder migration raises rather than silently converts.
- Actual full checkpoint migration verified every backbone tensor equal.
- Actual CUDA/BF16initial validation saved all24development images. Every saved
  pixel matches the prior no-context16kvalidation output (maximum difference0).
  That is an observed saved-image check, not a measurement of all possible inputs.

The separate experimental loader recognizes `inverse_architecture.scene_context`.
Public ModelManager, day-to-night checkpoints and app routing are not changed.
The new option is `--scene-context` in the private pilot. Globalstep continues
from16,000because the learned backbone is preserved; checkpoints record the
architecture addition explicitly. No reset to an alleged new from-scratch model.

## Controlled training

Same data and loss as the saved no-context LoLIauxiliary run:
111auditedreal-night/EXTERNALCycleGAN-Turbotargetprimarypairs, plus96audited
SYNTHETIC-low/REAL-dayLoLITRAINpairs. No new data, masks, foundation weights,
generative editor, noise augmentation, GAN or cycle term. All96prepared data
pairs were reviewed in the preceding run. New run did not reselect them.

Parent: `E:/LumiCycleCache/experiments/night2day_diverse111_transpose1000/adaptation_02000.pt`.
WarmstartEMA. Adam1e-4(.5,.999), BF16, batch2primary/1auxiliary, clip3, EMA.99;
LoLIauxiliary full-pixel RGBL1weight3. Independent data RNGs unchanged.1000updates
completed global17k; actual process exit0. Elapsed training loop124.468seconds
including interim validation; peak process allocated2.171995GiB. Prior control
ran with another training process concurrently, so timings are NOT a fair speed
comparison between architectures.

Checkpoint:
`E:/LumiCycleCache/experiments/night2day_context_LoLI96_from16000_v1/adaptation_01000.pt`.

The no-context matched control is the PREVIOUS SAME-DATA weight3run, not the
teacher-only weight0run:
`E:/LumiCycleCache/experiments/night2day_LoLI96_aux3_from16000_v1/adaptation_01000.pt`.

At500, context projection weightRMS=.00157880, EMA=.00134090: it is no longer
zero. At1000, an extra inference-only ablation bypasses this trained adapter
while retaining ALL OTHER weights of that same trained checkpoint. This is
different from the separately trained no-context control. No checkpoint bytes
are modified by bypassing the in-memory module.

## Measurements versus actual visual acceptance

| Model | Uniform development teacher fidelity | Mean LoLI TRAINphoto RGBL1 |
|---|---:|---:|
| Parent16k | 1.812819 | .172399 |
| No-context+500 | 1.781531 | .078141 |
| Context+500 | 1.760727 | .078305 |
| No-context+1000 | 1.747623 | .071747 |
| Context+1000 | 1.733452 | .070042 |

At1000, mean developmentRGBL1toEXTERNALreference: no-context.0723191,
contextON.0717364, contextOFF.0762608. Mean contextON/OFF RGB difference.0191594
in[0,1]. Thus the learned adapter affects outputs; it is not disconnected or a
complete no-op. Neither imitation nor this effect measures convincing daylight.
Disabling an adapted component is a useful dependence check, not a fair standalone
model retraining or proof that more context cannot help.

All24matched full-image comparisons opened at500and1000:

- `outputs/night_to_day_experimental/20261022_context_LoLI96_matched500`
- `outputs/night_to_day_experimental/20261022_context_LoLI96_matched1000`

The1000sheets include source, EXTERNALreference, separately trained no-context,
contextON and same-checkpoint contextOFF. Context changes exposure/colors and
slightly affects sky/foliage, but the failures persist:0/355/441glare,
142/214overhead confusion,8floatingdarkpatch,105skyhole,12sky/foliageartifacts,
78/86/358roadlightpools,202broadroadpatch,283mixednight/dayupper-scene,
84/284/318vehicle/lightdominance.71/426/50remain comparativelydaylike without
a convincing new overall improvement. Native contextON105/8/358at1000opened:
defects persist at native512resolution, not just in contact sheets.

No new LoLItraining-prediction panels were opened during this architecture trial;
the training statistic above is not a claim all96outputs were visually inspected.
All24BDDcomparison rows at each reviewed checkpoint were inspected. BDDtest,
12-source lockbox and LoLIheldout image pixels remain unused.

## Decision and limits

Not promoted. Keep16kconservativeparent and all ablation checkpoints. Adding this
small coarse context adapter under this supervision has NOT solved the requested
daylight transformation. The experiment does not rule out stronger contextual
architectures, better supervision, longer justified training or degradation-aware
learning. It does rule out declaring this bounded run a visual success because
its reference-matching loss is slightly lower.

Do not blindly extend this same recipe or replace the public default. Future
experiments should address supervision that retains nighttime illuminants and
unreliable source detail, not assume a larger attention block alone fixes it.
The existing N2D3 research notes distinguish degradation-aware contrastive
training from our already-failed brightness-mask approximation. A real unpaired
degradation-aware objective remains a different methodological test; it has not
been implemented by adding this adapter.

Verification:135Python tests passed; relevantRuffpassed. Four new tests cover
migration/RNG/identity, gradient/nonlocal connectivity, checkpoint loading and
unsupported conversion. No external model substitution, production mutation,
GitHub push or deployment. Final process inventory shows only the original
public remote-worker Python pair remains live. Goal remains unachieved.
