import json
from pathlib import Path

from PIL import Image

from daynight.prepare_acdc_kaggle import write_acdc_nights


def test_registers_night_images_as_unpaired(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_path = root / "acdc" / "images" / "night" / "train" / "seq" / "frame.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)
    output = tmp_path / "acdc.jsonl"
    assert write_acdc_nights(root, output) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["paired"] is False
    assert row["source"] == "acdc"
    assert Path(row["night_path"]).name == "frame.png"
