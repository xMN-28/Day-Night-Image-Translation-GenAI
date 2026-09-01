# LumiCycle V3: General Outdoor Translation Plan

## Status

This is a future plan only. V3 must not begin until the V2 pilot and main run are evaluated and a winning V2 checkpoint is selected. V3 will continue from that winning generator checkpoint; it will not discard the training already completed.

## 1. Objective

Extend LumiCycle from a road-focused model into a **general outdoor day ↔ night translator** that works across:

- Roads and traffic scenes
- Cities, villages, and architecture
- Mountains, forests, fields, parks, and gardens
- Beaches, lakes, rivers, and snow scenes
- People, animals, and common outdoor objects
- Clear, cloudy, rainy, foggy, and sunset conditions

The honest scope is general outdoor photography. V3 will not claim guaranteed performance on illustrations, medical images, screenshots, satellite imagery, or every indoor scene.

## 2. Starting checkpoint

1. Compare V2 checkpoints against the original 13k model using fixed validation images and untouched test sets.
2. Select the winner using nighttime realism, sky conversion, structural preservation, object retention, and human review.
3. Initialize both V3 generators and their EMA weights from that winner.
4. Reset only components whose architecture or learned domain must change, such as broader-domain discriminators.
5. Keep every earlier checkpoint unchanged so regression testing remains possible.

## 3. Proposed datasets

| Source | Approximate V3 use | Role |
|---|---:|---|
| BDD100K | 5k day + 5k night | Retain road and vehicle performance |
| Transient Attributes | 3–4k per domain | Real outdoor locations across time and weather |
| SkyFinder | 4–6k per domain | Landscapes, repeated day/night scenes, and sky masks |
| ExDark | 5–7k night | People, animals, objects, indoor/outdoor low-light diversity |
| Places365 outdoor subset | 5–10k day | Broad landscapes and scene categories |
| COCO outdoor subset | Optional 3–5k | Common-object and person preservation |

CODaN should remain an external generalization test and must not be folded into training if its official test split is used for reporting.

Official references:

- [Transient Attributes paper](https://jrenzhile.com/publications/siggraph2014/TransientAttributes.pdf)
- [SkyFinder on Zenodo](https://zenodo.org/records/5884485)
- [ExDark official repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- [Places365 official project](https://github.com/CSAILVision/places365)
- [COCO official site](https://cocodataset.org/)
- [CODaN official repository](https://github.com/Attila94/CODaN)

Do not download the complete 55 TB AMOS archive or the complete Places365 collection. Select only the useful, license-compatible subset.

## 4. Licensing and data governance

- Accept and record each dataset’s official terms before downloading.
- Store source name, original identifier, license/terms reference, domain label, scene category, camera/location group, and quality flags in every manifest.
- Never upload source datasets to GitHub.
- Do not assume that a dataset’s code or annotation license transfers ownership of every photograph.
- Do not redistribute trained weights publicly until the source-dataset terms have been reviewed for that use.
- Cite every dataset in the report and presentation.

## 5. Filtering and labeling

Create a unified preparation command in the future:

```powershell
python -m daynight.prepare_v3_data --sources-config configs/v3_sources.yaml
```

The preparation pipeline should:

1. Correct EXIF orientation and reject corrupt files.
2. Remove exact and perceptual duplicates across all sources.
3. Classify images as strict day, strict night, or ambiguous twilight.
4. Keep twilight separate rather than forcing it into day or night.
5. Assign broad categories: road, city, architecture, nature, water, snow, people/animals, and difficult weather.
6. Flag extreme blur, clipping, darkness, watermarks, collages, and synthetic images.
7. Produce contact sheets for human review of every source/category combination.
8. Record sky masks when supplied by SkyFinder.

Automatic labeling may shortlist samples, but a reviewed subset is required. Bright indoor photographs and sunsets must not silently contaminate the day/night domains.

## 6. Leakage-resistant splits

Target approximately 20,000 day and 20,000 night training images, balanced by category.

| Split | Day | Night | Use |
|---|---:|---:|---|
| Train | 20,000 | 20,000 | Weight updates |
| Validation | 1,500 | 1,500 | Checkpoint selection and tuning |
| Internal test | 2,000 | 2,000 | Final project evaluation only |

Critical rules:

- Group webcam frames by camera/location before splitting.
- Group near-duplicate photographs before splitting.
- Keep photographers or source sequences together when identifiers permit.
- Hold out complete SkyFinder/Transient camera locations for testing.
- Preserve the existing BDD100K test split unchanged.
- Never use CODaN’s official test images for training or checkpoint selection.

## 7. Sampling strategy

Each training batch should be sampled by both domain and scene category instead of uniformly across files. Initial target mixture:

- 30–35% BDD100K road scenes
- 35–40% natural landscapes and outdoor environments
- 20–25% architecture, people, animals, and common objects
- 10% difficult weather, snow, fog, sunsets, and high-glare scenes

Retaining BDD100K replay prevents catastrophic forgetting. If road metrics regress, increase the road replay fraction rather than restarting V3.

## 8. Model and loss changes

Keep the winning V2 generator architecture initially so all learned generator weights transfer exactly.

Planned V3 additions:

- Source-aware balanced samplers
- Scene-category-conditioned discriminator sampling
- Sky-mask supervision when a real mask is available
- Whole-frame and multi-scale adversarial critics for broad scene realism
- Existing cycle, identity, PatchNCE, DINO semantic, edge, color, and regional illumination losses
- Lower identity pressure for day → night than night → day if experiments justify direction-specific weights
- Replay/distillation loss on fixed BDD100K examples to retain road behavior
- EMA generators for all validation and inference

Do not add a large new pretrained image generator and present its ability as LumiCycle’s. Frozen models used only for semantic measurement or filtering must be disclosed.

## 9. Curriculum

### Stage A — Data adaptation pilot

- Start from the winning V2 generators.
- Reset broader-domain discriminators.
- Train 2,000 steps at 256 px on a 50% BDD100K / 50% new-data mixture.
- Confirm that general scenes improve without road regression.

### Stage B — General outdoor training

- Continue the complete Stage A checkpoint, including optimizers and critics.
- Train approximately 15k–25k additional steps at 256 px.
- Validate every 500–1,000 steps using category-balanced metrics.
- Stop early if the combined validation score and fixed gallery stagnate.

### Stage C — Optional refinement

- Continue the winning Stage B generators at 384 px for up to 5k steps.
- Run only after all 256 px acceptance tests pass and 12 GB VRAM safety is confirmed.

## 10. Evaluation

Report metrics separately for roads, cities, landscapes, water, snow, people/animals, and difficult weather.

- FID and KID against target-domain images
- DINO semantic distance
- Edge consistency
- Sky-region exposure and color consistency
- Frozen detector consistency for people, animals, and vehicles
- LPIPS or perceptual distance for cycle reconstruction
- 512 px inference time and peak VRAM
- Fixed human-review gallery using identical inputs for V2 and V3
- CODaN external day/night generalization results

An aggregate score must never hide a large failure in one category. Publish category-level results and difficult failures.

## 11. Acceptance criteria

V3 is promoted only if:

- It clearly improves general outdoor night realism over V2.
- It improves sky conversion on held-out locations.
- It has no meaningful regression on the untouched BDD100K test split.
- It preserves people, animals, vehicles, buildings, and major scene geometry.
- It does not simply darken every image or add red/orange casts everywhere.
- Both directions remain functional in the offline Gradio app.
- Typical 512 px inference remains below two seconds on the RTX 4070 Super.

## 12. Expected resources

- Approximately 50–100 GB additional disk space depending on selected sources
- 12 GB VRAM remains sufficient because image resolution and batch size determine VRAM more than total dataset size
- Approximately one pilot session plus several overnight training sessions
- Additional time for manual dataset review and category-balanced evaluation

## 13. Deliverables

- `configs/v3_sources.yaml`
- `configs/lumicycle_v3.yaml`
- Multi-source preparation and auditing command
- Source/license manifest
- Category-balanced sampler
- V2-versus-V3 quantitative report
- Road/general/sky comparison galleries
- Updated Gradio model selector and documentation
- College report section explaining continued training, generalization, limitations, and dataset ethics

This plan intentionally preserves the project’s existing training investment while expanding it through controlled fine-tuning and measurable generalization.
