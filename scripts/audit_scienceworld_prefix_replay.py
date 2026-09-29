#!/usr/bin/env python3
"""Independent audit of no-model multi-step ScienceWorld prefix replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(result: dict, protocol: dict, resources: dict) -> dict:
    if result.get("status") != "PASS_NO_MODEL":
        raise ValueError("multi-step replay did not pass")
    if result.get("protocol") != protocol.get("protocol") or result.get("protocol_sha256") != canonical_hash(protocol):
        raise ValueError("frozen protocol identity mismatch")
    if result.get("source_revision") != protocol.get("source_revision") or result.get("scienceworld_version") != protocol.get("expected_scienceworld_version"):
        raise ValueError("simulator source/runtime mismatch")
    if any(result.get(key) != 0 for key in ("model_inference_calls", "jev_calls", "network_calls")):
        raise ValueError("model, Jev, or network call detected")
    for key in ("gold_paths_requested", "raw_observations_written", "raw_action_strings_written", "task_score_or_reward_values_written"):
        if result.get(key) is not False:
            raise ValueError(f"forbidden output/access flag: {key}")
    plan = {(task, variation) for task, values in protocol["heldout_variation_tasks"].items() for variation in values}
    records = result.get("replay_records")
    if not isinstance(records, list) or len(records) != len(plan) or result.get("episode_count") != len(plan):
        raise ValueError("episode count differs from frozen replay plan")
    seen = set()
    total_steps = 0
    for row in records:
        if set(row) != {"task", "variation", "step_count", "steps"}:
            raise ValueError("unexpected replay record fields")
        key = (row["task"], row["variation"])
        if key not in plan or key in seen:
            raise ValueError("replay episode not in frozen plan or duplicated")
        seen.add(key)
        steps = row["steps"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= protocol["max_steps_per_prefix"] or row["step_count"] != len(steps):
            raise ValueError("invalid or truncated replay step record")
        for index, step in enumerate(steps, 1):
            if set(step) != {"step", "action_sha256", "legal_action_count", "state_sha256", "matched", "terminated"}:
                raise ValueError("unexpected step fields")
            if step["step"] != index or step["matched"] is not True or not isinstance(step["terminated"], bool):
                raise ValueError("step index or environment equality check failed")
            if not isinstance(step["legal_action_count"], int) or step["legal_action_count"] <= 0:
                raise ValueError("empty legal-action set at recorded step")
            if not HEX.fullmatch(step["action_sha256"]) or not HEX.fullmatch(step["state_sha256"]):
                raise ValueError("malformed replay digest")
            if step["terminated"] and index != len(steps):
                raise ValueError("steps recorded after environment termination")
        total_steps += len(steps)
    if seen != plan:
        raise ValueError("one or more frozen episodes are missing")
    if resources.get("state") != "COMPLETED" or str(resources.get("exit_code")) != "0:0":
        raise ValueError("Slurm job not completed successfully")
    for key, expected in (("account", "pi_fl426"), ("qos", "normal"), ("partition", "day")):
        if resources.get(key) != expected:
            raise ValueError(f"unexpected Slurm {key}")
    for key, expected in (("cpu_count", 4), ("memory_gib", 16), ("gpu_count", 0)):
        if int(resources.get(key, -1)) != expected:
            raise ValueError(f"unexpected Slurm {key}")
    elapsed = float(resources.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or not 0 < elapsed <= protocol["limits"]["wall_minutes"] * 60:
        raise ValueError("Slurm runtime outside frozen wall-time budget")
    return {
        "audit_status": "PASS",
        "claim_boundary": "Multi-step deterministic replay feasibility under the pinned simulator only; not agent outcome, efficacy, or OOD evidence.",
        "episode_count": len(seen),
        "matched_steps": total_steps,
        "max_steps_per_prefix": protocol["max_steps_per_prefix"],
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
    audited = audit(json.loads(args.result.read_text()), json.loads(args.protocol.read_text()), json.loads(args.resources.read_text()))
    audited["result_sha256"] = file_hash(args.result)
    audited["protocol_sha256"] = file_hash(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audited, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: audited[k] for k in ("audit_status", "episode_count", "matched_steps", "actual_gpu_hours")}, sort_keys=True))


if __name__ == "__main__":
    main()
