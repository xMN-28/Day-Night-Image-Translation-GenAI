import pytest

from daynight.audit_carla_pairs import select_sequences, training_pairs


def test_exact_training_only_pairing():
    files = [{"name": f"data_carla/{domain}/recording00_1.png", "bytes": 1}
        for domain in ("day_train", "night_train", "day_test", "night_test")]
    rows = training_pairs(files)
    assert len(rows) == 1
    assert rows[0]["day"]["name"].split("/")[1] == "day_train"
    with pytest.raises(ValueError):
        training_pairs([{"name": "../day_train/recording00_1.png"}])


def test_recording_stratification_unique_and_deterministic():
    pairs = [{"sequence": f"recording{s:02d}", "name": f"recording{s:02d}_{i}.png"}
        for s in range(3) for i in range(1, 11)]
    selected = select_sequences(pairs, 12)
    assert len(selected) == len({r["name"] for r in selected}) == 12
    assert {r["sequence"] for r in selected} == {"recording00", "recording01", "recording02"}
    assert selected == select_sequences(pairs, 12)
