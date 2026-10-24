"""Compare exported inference pixels to originals before deleting training files."""
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from daynight.inference import ModelManager
from daynight.remote_worker import MODEL_CATALOG

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(6)
    path = ROOT / "checkpoints/showcase/manifest.json"
    manifest = json.loads(path.read_text())
    reports = []
    manager = ModelManager()
    for record in manifest["models"]:
        model_id = record["id"]
        if model_id == "sky_matte":
            continue
        spec = MODEL_CATALOG[model_id]
        inverse = model_id == "night_day_experimental"
        source = (ROOT / "outputs/night_to_day_showcase/night_inputs/01_val_0299.jpg" if inverse
                  else ROOT / "outputs/demo_showcase/daytime_inputs/02_val_0023.jpg")
        with Image.open(source) as image:
            image = image.convert("RGB")
            hashes = []
            for checkpoint in (record["source"], record["output"]):
                output, metadata = manager.translate_custom(image, spec.directions[0], spec.engine_name,
                    512, checkpoint_override=checkpoint, display_name=spec.label)
                assert metadata["global_step"] == record["step"]
                hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
            assert hashes[0] == hashes[1], f"Inference pixels changed for {model_id}"
        reports.append({"id": model_id, "identical_pixels": True, "step": record["step"], "rgb_sha256": hashes[0]})
        print(f"Identical inference: {model_id}", flush=True)
    (ROOT / "checkpoints/showcase/inference_equivalence.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
