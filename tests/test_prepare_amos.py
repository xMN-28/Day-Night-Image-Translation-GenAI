from pathlib import Path

from PIL import Image

from daynight.prepare_amos import write_amos_pairs


def test_pairs_night_with_same_camera_day(tmp_path: Path) -> None:
    root = tmp_path / "amos"
    image_dir = root / "DNIM" / "Image" / "camera"
    timestamps = root / "DNIM" / "time_stamp"
    image_dir.mkdir(parents=True)
    timestamps.mkdir(parents=True)
    day_name = "20260101_120000.jpg"
    night_name = "20260101_230000.jpg"
    for name in (day_name, night_name):
        Image.new("RGB", (8, 8)).save(image_dir / name)
    (timestamps / "camera.txt").write_text(
        f"{day_name} 20260101 12 00\n{night_name} 20260101 23 00\n", encoding="utf-8"
    )

    output = tmp_path / "pairs.csv"
    assert write_amos_pairs(root, output) == 1
    text = output.read_text(encoding="utf-8")
    assert day_name in text
    assert night_name in text
    assert ",train,0.5" in text
