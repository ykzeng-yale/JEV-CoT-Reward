import json
from pathlib import Path

import pytest

from scripts.audit_scienceworld_transfer_preflight import EXPECTED_REPLAYS, audit


def _fixture():
    protocol = json.loads((Path(__file__).parents[1] / "configs/scienceworld_transfer_preflight_v1.json").read_text())
    names = list(EXPECTED_REPLAYS)
    result = {
        "status": "PASS_NO_MODEL",
        "source_revision": protocol["source_revision"],
        "scienceworld_version": "1.3.0",
        "task_count": 30,
        "variation_count": 7207,
        "model_inference_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "raw_observations_written": False,
        "raw_action_strings_written": False,
        "task_score_or_reward_values_written": False,
        "gold_paths_requested": False,
        "split_integrity": "PASS: disjoint train/dev/test sets cover every supported variation",
        "partition_summary": {
            name: {"variation_count": variations, "train": 0, "dev": 0, "test": variations}
            for name, variations in protocol["task_variations"].items()
        },
        "replay_count": 9,
        "replay_records": [],
    }
    for task in names:
        for variation in (1, 2, 3):
            result["replay_records"].append({
                "task": task, "variation": variation,
                "action_sha256": "a" * 64, "legal_action_count": 1,
                "initial_state_sha256": "b" * 64, "post_action_state_sha256": "c" * 64,
                "initial_equal": True, "post_action_equal": True,
            })
    import hashlib
    protocol_bytes = json.dumps(protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    result["manifest_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    resources = {
        "state": "COMPLETED", "exit_code": "0:0", "account": "pi_fl426", "qos": "normal",
        "partition": "day", "cpu_count": 4, "memory_gib": 16, "gpu_count": 0,
        "elapsed_seconds": 120,
    }
    return result, protocol, resources


def test_valid_no_model_preflight_and_cpu_only_accounting_pass():
    result, protocol, resources = _fixture()
    audited = audit(result, protocol, resources)
    assert audited["audit_status"] == "PASS"
    assert audited["actual_gpu_hours"] == 0


@pytest.mark.parametrize("field,value", [
    ("model_inference_calls", 1), ("jev_calls", 1), ("network_calls", 1),
    ("raw_observations_written", True), ("gold_paths_requested", True),
])
def test_forbidden_model_calls_or_raw_content_are_rejected(field, value):
    result, protocol, resources = _fixture()
    result[field] = value
    with pytest.raises(ValueError):
        audit(result, protocol, resources)


def test_replay_mismatch_is_rejected():
    result, protocol, resources = _fixture()
    result["replay_records"][0]["post_action_equal"] = False
    with pytest.raises(ValueError, match="do not match"):
        audit(result, protocol, resources)


def test_wrong_slurm_gpu_shape_is_rejected():
    result, protocol, resources = _fixture()
    resources["gpu_count"] = 1
    with pytest.raises(ValueError, match="gpu_count"):
        audit(result, protocol, resources)


def test_out_of_range_variation_is_rejected():
    result, protocol, resources = _fixture()
    result["replay_records"][0]["variation"] = protocol["task_variations"][result["replay_records"][0]["task"]]
    with pytest.raises(ValueError, match="outside the official variation range"):
        audit(result, protocol, resources)


def test_protocol_mismatch_is_rejected():
    result, protocol, resources = _fixture()
    protocol["sample_rule"] = "changed"
    with pytest.raises(ValueError, match="different frozen protocol"):
        audit(result, protocol, resources)


def test_replay_plan_mismatch_is_rejected():
    result, protocol, resources = _fixture()
    protocol["replay_samples_per_task"] = 2
    with pytest.raises(ValueError, match="replay task/sample plan"):
        audit(result, protocol, resources)


def test_replay_count_mismatch_is_rejected():
    result, protocol, resources = _fixture()
    result["replay_count"] = 8
    with pytest.raises(ValueError, match="replay_count"):
        audit(result, protocol, resources)
