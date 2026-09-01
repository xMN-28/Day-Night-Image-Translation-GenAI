# Final Results Template

Do not enter estimated or remembered values. Copy values from each `metrics.json` after the untouched test evaluation.

| Model | Training source | D→N FID ↓ | N→D FID ↓ | D→N DINO ↓ | N→D DINO ↓ | Object retention ↑ | 512 px latency ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| CycleGAN | Local, from scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| Attention ablation | Local, from scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| LumiCycle | Local, from scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| CycleGAN-Turbo | External pretrained reference | TBD | TBD | TBD | TBD | TBD | TBD |

Record the exact checkpoint step, test sample count, resolution, GPU, PyTorch version, and command below the table.

## Required failure cases

Include at least one example of each:

- Strong headlight glare.
- Deep shadow with missing source detail.
- Small pedestrian or cyclist.
- Traffic signal/sign.
- Wet or reflective road.
- Non-driving image outside the training distribution.

