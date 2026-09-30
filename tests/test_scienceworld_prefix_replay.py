import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_scienceworld_prefix_replay import audit


PROTOCOL = json.loads((Path(__file__).parents[1] / "configs/scienceworld_prefix_replay_v1.json").read_text())


def _fixture():
    records = []
    for task, variations in PROTOCOL["heldout_variation_tasks"].items():
        for variation in variations:
            records.append({"task": task, "variation": variation, "step_count": 2, "steps": [
                {"step": i, "action_sha256": "a" * 64, "legal_action_count": 2, "state_sha256": "b" * 64, "matched": True, "terminated": False}
                for i in (1, 2)
            ]})
    result = {"status":"PASS_NO_MODEL", "protocol":PROTOCOL["protocol"], "protocol_sha256":hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(), "source_revision":PROTOCOL["source_revision"], "scienceworld_version":"1.3.0", "episode_count":len(records), "replay_records":records, "model_inference_calls":0, "jev_calls":0, "network_calls":0, "gold_paths_requested":False, "raw_observations_written":False, "raw_action_strings_written":False, "task_score_or_reward_values_written":False}
    resources={"state":"COMPLETED", "exit_code":"0:0", "account":"pi_fl426", "qos":"normal", "partition":"day", "cpu_count":4, "memory_gib":16, "gpu_count":0, "elapsed_seconds":120, "output_manifest_verified":True, "verified_output_files":199, "output_manifest_sha256":"c"*64}
    return result, resources


def test_complete_protocol_shaped_cpu_run_passes():
    result, resources = _fixture()
    report = audit(result, PROTOCOL, resources)
    assert report["episode_count"] == 9
    assert report["matched_steps"] == 18
    assert report["actual_gpu_hours"] == 0


@pytest.mark.parametrize("field,value", [("jev_calls",1),("model_inference_calls",1),("network_calls",1),("gold_paths_requested",True),("raw_action_strings_written",True)])
def test_forbidden_calls_or_outputs_fail(field,value):
    result, resources = _fixture(); result[field]=value
    with pytest.raises(ValueError): audit(result, PROTOCOL, resources)


def test_missing_episode_fails():
    result, resources = _fixture(); result["replay_records"].pop(); result["episode_count"]-=1
    with pytest.raises(ValueError, match="episode count"): audit(result, PROTOCOL, resources)


def test_mismatched_state_fails():
    result, resources = _fixture(); result["replay_records"][0]["steps"][1]["matched"]=False
    with pytest.raises(ValueError, match="equality"): audit(result, PROTOCOL, resources)


def test_gpu_allocation_fails():
    result, resources = _fixture(); resources["gpu_count"]=1
    with pytest.raises(ValueError, match="gpu_count"): audit(result, PROTOCOL, resources)


def test_unverified_output_manifest_fails():
    result, resources = _fixture(); resources["output_manifest_verified"] = False
    with pytest.raises(ValueError, match="checksum manifest"): audit(result, PROTOCOL, resources)
