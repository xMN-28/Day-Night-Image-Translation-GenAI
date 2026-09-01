# LumiRender presentation and live-demo outline

## 10-slide outline

1. **Problem:** change illumination without moving the scene.
2. **Why V1/V2 plateaued:** RGB style matching can darken an image but does not know lights, materials or camera noise.
3. **Failed V2.1 ablation:** stronger detail preservation protected branches but suppressed the night transformation.
4. **Fundamental model:** linear image = reflectance × illumination + specular + reflection.
5. **Architecture:** factorizer → Gaussian light composer → renderer → camera → bounded correction.
6. **Gaussian clarification:** 2D lights, bloom and frequency separation; not 3D Gaussian splatting.
7. **Training evidence:** frozen offline teachers, licensed correspondences, four stages, strict split.
8. **Evaluation:** V1/V2/Turbo comparisons, nine difficult categories, blind review, FID/KID/structure/object metrics.
9. **Live demo:** same input with default, darker and wetter seeds; show metadata and reproducibility.
10. **Honest conclusion:** plausible seeded night, limitations, measured result, future general-purpose data.

## Suggested speaking assignment for four people

- Member 1: motivation, prior paper and baseline failure analysis.
- Member 2: image-formation fundamentals and scene factorizer.
- Member 3: Gaussian lights, reflections, camera model and training stages.
- Member 4: evaluation, demo, limitations, attribution and conclusion.

## Live-demo checklist

- Run `python -m daynight.preflight_lumirender` and save its output.
- Confirm `acceptance.json` says `passed: true`; otherwise demonstrate V2 and label LumiRender experimental.
- Start `python app.py` with Wi-Fi disabled.
- Test seed reproducibility on one road and one landscape.
- Keep screenshots for default, darker and wet outputs in the slide deck.
- Never describe Turbo or teacher models as team-created.
