import pytest

from daynight.index_loli_training import near_any


def test_hash_match_threshold_including_high_unsigned_bit():
    flags, distances = near_any([0, 15, 31, 2 ** 63], [0], maximum_distance=4)
    assert flags == [True, True, False, True]
    assert distances == [0, 4, 5, 1]


def test_refuse_missing_protected_metadata():
    with pytest.raises(ValueError):
        near_any([0], [])
