# 12-Minute Presentation Outline

## Team speaking plan

Replace Member A/B/C/D with names before presenting.

| Time | Speaker | Content |
|---:|---|---|
| 0:00–1:00 | Member A | Problem, motivation, and one-sentence result |
| 1:00–2:15 | Member A | Why paired day/night data is difficult; BDD100K |
| 2:15–3:30 | Member B | CycleGAN baseline and cycle consistency diagram |
| 3:30–5:15 | Member B | LumiCycle attention, PatchNCE, DINO and edge losses |
| 5:15–6:15 | Member C | Leakage-safe splits and overnight training safeguards |
| 6:15–7:30 | Member C | Metrics and ablation table |
| 7:30–10:00 | Member D | Live Gradio demonstration in both directions |
| 10:00–11:00 | Member A | Failure cases, ethics and limitations |
| 11:00–12:00 | All | Conclusion and transition to questions |

## Slide sequence

### Slide 1 — LumiCycle

- One strong day/night comparison image.
- “Unpaired, bidirectional, structure-aware translation.”
- Team names and department.

### Slide 2 — Why this matters

- Night domain gap: low illumination, glare, noise, lost detail.
- Paired collection is expensive and dynamic scenes do not align.
- Research use: augmentation and robustness experiments.

### Slide 3 — Starting paper

- CycleGAN vs CycleGANAS on BDD100K.
- Strength: unpaired bidirectional training.
- Gaps: test mixing, mostly qualitative evidence, expensive NAS.

### Slide 4 — CycleGAN in one picture

- Show the bidirectional diagram from `ARCHITECTURE.md`.
- Explain adversarial, cycle, and identity losses in plain language.

### Slide 5 — Our LumiCycle contribution

- Attention generators.
- Multi-scale spectral discriminators.
- PatchNCE + DINO semantics + Sobel edges.
- EMA output.

### Slide 6 — Scientific pipeline

- Strict time-of-day filter.
- Perceptual duplicate grouping.
- 5k/500/1k per domain split.
- Validation selects checkpoints; test is opened once.

### Slide 7 — Training on a 4070 Super

- 256 px, BF16, batch 1, accumulation 2.
- Full core graph memory probe result.
- Overnight `--max-hours` and exact resume.
- TensorBoard screenshot with labeled curves.

### Slide 8 — Results and ablation

- Fill from `RESULTS_TEMPLATE.md` after final evaluation.
- Compare CycleGAN, attention-only, LumiCycle, external Turbo.
- Highlight improvement and any metric that did not improve.

### Slide 9 — Visual comparisons

- Same fixed inputs for every model.
- Include one success and one failure.
- Zoom into a lane, pedestrian/sign, and light source.

### Slide 10 — Live demo

- Day → Night first, then Night → Day.
- Point out model label, resolution, and latency.
- Do not wait for an untested audience image as the first example.

### Slide 11 — Limitations and ethics

- Plausible rendering is not recovered truth.
- Hallucination risk.
- Dataset bias.
- Not for direct vehicle control.

### Slide 12 — Conclusion

- Reproducible baseline.
- Explainable enhanced model.
- Stronger evaluation and offline showcase.
- QR/link to repository if permitted.

## Demo narration

“This is our locally trained LumiCycle checkpoint, not the external Turbo reference. I upload a daytime road scene, choose Day to Night, and run one deterministic generator pass. The output changes illumination while the structural losses encourage lanes and objects to remain. The metadata shows the exact model, resolution, device, checkpoint and latency. Now we reverse the direction on a real night input. Night to day is harder because information lost in darkness cannot truly be recovered.”

