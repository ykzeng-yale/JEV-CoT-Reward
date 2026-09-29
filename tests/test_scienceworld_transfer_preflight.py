import pytest

from scripts.preflight_scienceworld_transfer import sample_indices, validate_partition


@pytest.mark.parametrize(
    "maximum,train,dev,test",
    [
        (8, [0, 1, 2, 3], [4, 5], [6, 7]),
        (5, [0, 1], [2], [3, 4]),
    ],
)
def test_partition_validation_accepts_complete_disjoint_sets(maximum, train, dev, test):
    counts = validate_partition(maximum, train, dev, test)
    assert sum(counts.values()) == maximum


def test_partition_validation_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_partition(4, [0, 1], [1], [2, 3])


def test_partition_validation_rejects_duplicate_and_out_of_range_indices():
    with pytest.raises(ValueError, match="duplicate"):
        validate_partition(4, [0, 1, 1], [2], [3])
    with pytest.raises(ValueError, match="out-of-range"):
        validate_partition(4, [0, 1], [2], [4])


def test_replay_sampling_is_stable_unique_and_covers_ends():
    assert sample_indices([2, 3, 4, 5, 6]) == [2, 4, 6]
    assert sample_indices([9]) == [9]
    with pytest.raises(ValueError, match="no official test"):
        sample_indices([])
