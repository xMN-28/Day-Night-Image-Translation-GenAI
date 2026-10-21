from daynight.prepare_amos import _records, write_amos_pairs


def test_metadata_clock_overrides_filename_clock(tmp_path):
    images = tmp_path / "DNIM" / "Image" / "camera"
    stamps = tmp_path / "DNIM" / "time_stamp"
    images.mkdir(parents=True)
    stamps.mkdir()
    # Filenames look like night and day, respectively; metadata reverses them.
    (images / "20151101_212503.jpg").touch()
    (images / "20151101_072507.jpg").touch()
    metadata = stamps / "camera.txt"
    metadata.write_text("20151101_212503.jpg 20151101 14 27\n20151101_072507.jpg 20151101 00 27\n")
    rows = _records(images.parent, metadata)
    assert rows[0][0].isoformat() == "2015-11-01T14:27:00"
    assert rows[1][0].isoformat() == "2015-11-01T00:27:00"
    out = tmp_path / "pairs.csv"
    assert write_amos_pairs(tmp_path, out) == 1
    assert "20151101_212503.jpg" in out.read_text().splitlines()[1].split(",")[0]


def test_relative_date_fallback_retains_metadata_hour(tmp_path):
    images = tmp_path / "camera"
    images.mkdir()
    (images / "20151101_072507.jpg").touch()
    metadata = tmp_path / "camera.txt"
    metadata.write_text("20151101_072507.jpg 1 23 15\n")
    rows = _records(tmp_path, metadata)
    assert rows[0][0].isoformat() == "2015-11-01T23:15:00"
