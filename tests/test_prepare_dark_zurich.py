from pathlib import Path

from PIL import Image

from daynight.prepare_dark_zurich import write_pairs_csv


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def test_writes_validation_pair_csv(tmp_path: Path) -> None:
    root = tmp_path / "dark_zurich"
    lists = root / "lists_file_names"
    lists.mkdir(parents=True)
    night_key = "val/night/night_seq/night_frame_000001"
    day_key = "val_ref/day/day_seq/day_frame_000001_ref"
    (lists / "val_filenames.txt").write_text(night_key + "\n", encoding="utf-8")
    (lists / "val_ref_filenames.txt").write_text(day_key + "\n", encoding="utf-8")
    _touch_image(root / "rgb_anon" / f"{night_key}_rgb_anon.png")
    _touch_image(root / "rgb_anon" / f"{day_key}_rgb_anon.png")

    output = tmp_path / "pairs.csv"
    assert write_pairs_csv(root, output, "val") == 1
    text = output.read_text(encoding="utf-8")
    assert "day_frame_000001_ref_rgb_anon.png" in text
    assert "night_frame_000001_rgb_anon.png" in text
    assert ",val,1.0" in text
