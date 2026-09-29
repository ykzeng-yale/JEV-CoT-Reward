#!/usr/bin/env python3
"""Independent audit of the completed no-model ScienceWorld preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

HEX = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_REPLAYS = {
    "measure-melting-point-unknown-substance": 3,
    "test-conductivity-of-unknown-substances": 3,
    "mendelian-genetics-unknown-plant": 3,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit(result: dict, protocol: dict, resources: dict) -> dict:
    if protocol.get("heldout_variation_tasks") != list(EXPECTED_REPLAYS) or protocol.get("replay_samples_per_task") != 3:
        raise ValueError("held-out replay task/sample plan differs from this auditor")
    if result.get("status") != "PASS_NO_MODEL":
        raise ValueError("preflight did not report PASS_NO_MODEL")
    if result.get("source_revision") != protocol["source_revision"]:
        raise ValueError("source revision mismatch")
    if result.get("scienceworld_version") != protocol["expected_scienceworld_version"]:
        raise ValueError("runtime version mismatch")
    if result.get("task_count") != protocol["expected_task_count"]:
        raise ValueError("task count mismatch")
    if result.get("variation_count") != protocol["expected_variation_count"]:
        raise ValueError("variation count mismatch")
    if result.get("model_inference_calls") != 0 or result.get("jev_calls") != 0 or result.get("network_calls") != 0:
        raise ValueError("unexpected model, Jev, or network calls")
    for key in ("raw_observations_written", "raw_action_strings_written", "task_score_or_reward_values_written"):
        if result.get(key) is not False:
            raise ValueError(f"output privacy/claim guard failed: {key}")
    if result.get("gold_paths_requested") is not False:
        raise ValueError("gold-path generation must remain disabled")
    if result.get("manifest_sha256") != canonical_hash(protocol):
        raise ValueError("result was produced from a different frozen protocol")
    if result.get("split_integrity") != "PASS: disjoint train/dev/test sets cover every supported variation":
        raise ValueError("variation-split integrity did not pass")

    summary = result.get("partition_summary")
    expected_counts = protocol["task_variations"]
    if not isinstance(summary, dict) or set(summary) != set(expected_counts):
        raise ValueError("task inventory differs from the frozen official README table")
    for task, variations in expected_counts.items():
        row = summary[task]
        if row.get("variation_count") != variations:
            raise ValueError(f"variation count mismatch for {task}")
        counts = [row.get(name) for name in ("train", "dev", "test")]
        if any(not isinstance(value, int) or value < 0 for value in counts) or sum(counts) != variations:
            raise ValueError(f"invalid split counts for {task}")
        if counts[2] < 3:
            raise ValueError(f"held-out task has fewer than three test variations: {task}")

    records = result.get("replay_records")
    if not isinstance(records, list) or len(records) != sum(EXPECTED_REPLAYS.values()):
        raise ValueError("unexpected number of deterministic replay samples")
    if result.get("replay_count") != len(records):
        raise ValueError("replay_count field disagrees with replay records")
    counts: dict[str, int] = {name: 0 for name in EXPECTED_REPLAYS}
    seen: set[tuple[str, int]] = set()
    for row in records:
        if set(row) != {
            "task", "variation", "action_sha256", "legal_action_count",
            "initial_state_sha256", "post_action_state_sha256", "initial_equal", "post_action_equal",
        }:
            raise ValueError("replay record contains unexpected or raw fields")
        task = row["task"]
        if task not in EXPECTED_REPLAYS or not isinstance(row.get("variation"), int):
            raise ValueError("replay sample is not in the frozen held-out-task subset")
        if not 0 <= row["variation"] < expected_counts[task]:
            raise ValueError("replay variation is outside the official variation range")
        pair = (task, row["variation"])
        if pair in seen:
            raise ValueError("duplicate replay variation")
        seen.add(pair)
        counts[task] += 1
        if row.get("legal_action_count", 0) <= 0:
            raise ValueError("no legal action available at replay checkpoint")
        if row.get("initial_equal") is not True or row.get("post_action_equal") is not True:
            raise ValueError("independent replay states do not match")
        for key in ("action_sha256", "initial_state_sha256", "post_action_state_sha256"):
            if not isinstance(row.get(key), str) or not HEX.fullmatch(row[key]):
                raise ValueError(f"invalid digest field: {key}")
    if counts != EXPECTED_REPLAYS:
        raise ValueError("held-out replay task counts differ from protocol")

    if resources.get("state") != "COMPLETED" or str(resources.get("exit_code")) != "0:0":
        raise ValueError("Slurm preflight job did not exit successfully")
    for key, expected in (("account", "pi_fl426"), ("qos", "normal"), ("partition", "day")):
        if resources.get(key) != expected:
            raise ValueError(f"Slurm {key} differs from the frozen request")
    for key, expected in (("cpu_count", 4), ("memory_gib", 16), ("gpu_count", 0)):
        if int(resources.get(key, -1)) != expected:
            raise ValueError(f"Slurm {key} differs from the frozen request")
    elapsed = float(resources.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or not 0 < elapsed <= protocol["limits"]["wall_minutes"] * 60:
        raise ValueError("Slurm elapsed time is outside the frozen bound")

    return {
        "audit_status": "PASS",
        "claim_boundary": "Source-pinned no-model split and one-action deterministic-replay feasibility only; no intervention efficacy or OOD claim.",
        "task_count": result["task_count"],
        "variation_count": result["variation_count"],
        "replay_count": len(records),
        "replay_matches": len(records),
        "model_inference_calls": 0,
        "jev_calls": 0,
        "actual_gpu_hours": elapsed * int(resources["gpu_count"]) / 3600,
        "slurm_accounting": resources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    resources = json.loads(args.resources.read_text(encoding="utf-8"))
    audited = audit(result, protocol, resources)
    audited["result_sha256"] = sha256(args.result)
    audited["protocol_sha256"] = sha256(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audited, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: audited[key] for key in ("audit_status", "task_count", "variation_count", "replay_matches", "actual_gpu_hours")}, sort_keys=True))


if __name__ == "__main__":
    main()
