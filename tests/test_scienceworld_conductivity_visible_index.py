from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_scienceworld_conductivity_visible_index.py"
spec = importlib.util.spec_from_file_location("conductivity_visible_index", SCRIPT)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)
AUDITOR_SCRIPT = ROOT / "scripts/audit_scienceworld_conductivity_visible_index_result.py"
audit_spec = importlib.util.spec_from_file_location("conductivity_visible_index_auditor", AUDITOR_SCRIPT)
auditor = importlib.util.module_from_spec(audit_spec)
assert audit_spec.loader is not None
audit_spec.loader.exec_module(auditor)


def test_public_description_and_unique_part_decode_combination_index():
    rule = json.loads((ROOT / "configs/scienceworld_conductivity_visible_index_v1.json").read_text())["recovery_rule"]
    visible = (
        "Your task is to determine if unknown substance H is electrically conductive. "
        "If it is electrically conductive, place it in the yellow box. "
        "If it is electrically nonconductive, place it in the purple box. "
        "In the workshop you see an electric buzzer."
    )
    recovered, part_count, letter_visible = runner.recover_index(visible, rule)
    assert (recovered, part_count, letter_visible) == (174, 1, True)


def test_decoder_does_not_guess_when_visible_scene_has_multiple_components():
    rule = json.loads((ROOT / "configs/scienceworld_conductivity_visible_index_v1.json").read_text())["recovery_rule"]
    visible = "unknown substance B red box green box red light bulb electric motor"
    recovered, part_count, letter_visible = runner.recover_index(visible, rule)
    assert recovered is None
    assert part_count == 2
    assert letter_visible


def test_all_frozen_source_member_hashes_and_source_contracts_match(tmp_path):
    protocol_path = ROOT / "configs/scienceworld_conductivity_visible_index_v1.json"
    protocol = json.loads(protocol_path.read_text())
    source_root = tmp_path / "source"
    with tarfile.open(ROOT / protocol["source_archive"], "r:gz") as archive:
        for record in protocol["source_members"].values():
            data = archive.extractfile(record["path"]).read()
            relative = record["path"].removeprefix("scienceworld-source/")
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    hashes = runner.verify_source_contract(protocol, source_root)
    assert hashes == {name: row["sha256"] for name, row in protocol["source_members"].items()}


def test_source_hash_drift_fails_closed(tmp_path):
    protocol = json.loads((ROOT / "configs/scienceworld_conductivity_visible_index_v1.json").read_text())
    source_root = tmp_path / "source"
    record = protocol["source_members"]["conductivity_task"]
    target = source_root / record["path"].removeprefix("scienceworld-source/")
    target.parent.mkdir(parents=True)
    target.write_text("different source")
    with pytest.raises(ValueError, match="pinned source hash mismatch"):
        runner.verify_source_contract(protocol, source_root)


def _aggregate_result(protocol_path: Path):
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    rows = []
    for split, start, end in (("train", 0, 300), ("dev", 300, 450)):
        for variation in range(start, end):
            rows.append({
                "split": split,
                "variation_id": variation,
                "observation_sha256": f"{variation:064x}",
                "letter_visible": True,
                "candidate_part_count": 0,
                "source_index_decoded": False,
                "source_index_recovered": False,
                "variation_idx_in_observation": False,
            })
    return {
        "protocol": protocol["protocol"],
        "protocol_sha256": __import__("hashlib").sha256(protocol_bytes).hexdigest(),
        "source_member_sha256": {k: v["sha256"] for k, v in protocol["source_members"].items()},
        "split_counts": {"train": 300, "dev": 150, "test_loaded": 0},
        "row_count": 450,
        "train_index_recovery_count": 0,
        "dev_index_recovery_count": 0,
        "unique_observation_hash_count": 450,
        "decoder_applied_count": 0,
        "variation_idx_in_observation_count": 0,
        "records": rows,
        "raw_observations_written": False,
        "answer_labels_or_values_read": False,
        "score_reward_values_read": False,
        "test_ids_loaded": False,
        "model_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "gold_paths_requested": False,
    }


def test_independent_auditor_enforces_train_dev_only_and_aggregate_scope(tmp_path):
    protocol = ROOT / "configs/scienceworld_conductivity_visible_index_v1.json"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_aggregate_result(protocol)))
    report = auditor.audit(protocol, result_path)
    assert report["status"] == "NO_INDEX_RECOVERY_WITH_DECLARED_DECODER"
    assert report["test_rows"] == 0
    assert report["cross_split_shared_exact_observation_hashes"] == 0
    duplicate = json.loads(result_path.read_text())
    duplicate["records"][300]["observation_sha256"] = duplicate["records"][0]["observation_sha256"]
    duplicate["unique_observation_hash_count"] = 449
    result_path.write_text(json.dumps(duplicate))
    overlap_report = auditor.audit(protocol, result_path)
    assert overlap_report["cross_split_shared_exact_observation_hashes"] == 1
    assert overlap_report["cross_split_shared_rows"] == 2
    assert overlap_report["train_rows_with_cross_split_duplicate_input"] == 1
    assert overlap_report["dev_rows_with_train_seen_input"] == 1
    assert overlap_report["dev_unique_inputs_not_seen_in_train"] == 149
    result = json.loads(result_path.read_text())
    result["records"][-1]["split"] = "test"
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="non-train/dev record"):
        auditor.audit(protocol, result_path)
