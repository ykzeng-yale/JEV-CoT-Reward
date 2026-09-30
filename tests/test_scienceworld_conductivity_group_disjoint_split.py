import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "group_disjoint_split", ROOT / "scripts/build_scienceworld_conductivity_prompt_disjoint_split.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_result():
    rows = []
    # Shared train/dev digest has two dev members; dev-only digest has one row.
    for i in range(300):
        digest = ("a" if i in (0, 1) else f"train-{i}").ljust(64, "0")
        rows.append({"split": "train", "variation_id": i, "observation_sha256": digest})
    for i in range(300, 450):
        digest = ("a" if i < 302 else f"dev-{i}").ljust(64, "0")
        rows.append({"split": "dev", "variation_id": i, "observation_sha256": digest})
    return {
        "protocol": "scienceworld_conductivity_visible_index_v1",
        "split_counts": {"train": 300, "dev": 150, "test_loaded": 0},
        "test_ids_loaded": False,
        "answer_labels_or_values_read": False,
        "model_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "records": rows,
    }


def test_excludes_all_dev_members_of_cross_split_hash_groups_and_exports_no_labels(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(make_result()))
    report = MODULE.build(path)
    assert report["counts"]["retained_hash_disjoint_dev_rows"] == 148
    assert report["counts"]["excluded_dev_rows"] == 2
    assert report["counts"]["remaining_exact_hash_overlap"] == 0
    assert 300 not in report["retained_dev_ids"] and 301 not in report["retained_dev_ids"]
    assert "records" not in report
    assert report["source_label_values_read"] is False


@pytest.mark.parametrize("change", [
    {"test_ids_loaded": True},
    {"split_counts": {"train": 300, "dev": 150, "test_loaded": 1}},
    {"answer_labels_or_values_read": True},
    {"jev_calls": 1},
])
def test_refuses_test_or_outcome_access(tmp_path, change):
    payload = make_result() | change
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        MODULE.build(path)
