import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prompt_split", ROOT / "scripts/analyze_scienceworld_conductivity_prompt_split.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_result():
    rows = []
    # Train: 3 singleton groups and one group of 297.
    for i in range(300):
        group = "a" if i >= 3 else f"train-{i}"
        rows.append({"split": "train", "variation_id": i, "observation_sha256": group.ljust(64, "0")})
    # Dev: 2 rows overlap train group a; one dev-only group has two rows.
    for i in range(300, 450):
        group = "a" if i < 302 else ("dev-only" if i < 304 else f"dev-{i}")
        rows.append({"split": "dev", "variation_id": i, "observation_sha256": group.ljust(64, "0")})
    return {
        "protocol": "scienceworld_conductivity_visible_index_v1",
        "split_counts": {"train": 300, "dev": 150, "test_loaded": 0},
        "test_ids_loaded": False,
        "answer_labels_or_values_read": False,
        "records": rows,
    }


def test_counts_shared_prompt_groups_and_perfect_dependence_sensitivity(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(synthetic_result()))
    report = MODULE.analyze(path)
    d = report["split_diagnostic"]
    assert d["cross_split_shared_exact_hash_groups"] == 1
    assert d["train_rows_in_cross_split_shared_groups"] == 297
    assert d["dev_rows_with_an_exact_train_input"] == 2
    assert d["dev_rows_without_an_exact_train_input"] == 148
    assert d["dev_distinct_hashes_without_an_exact_train_input"] == 147
    assert d["train"]["kish_effective_n_if_within_hash_outcomes_are_perfectly_dependent"] < 2


@pytest.mark.parametrize("change", [
    {"split_counts": {"train": 300, "dev": 150, "test_loaded": 1}},
    {"test_ids_loaded": True},
    {"answer_labels_or_values_read": True},
])
def test_refuses_test_or_label_exposure(tmp_path, change):
    payload = synthetic_result() | change
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        MODULE.analyze(path)
