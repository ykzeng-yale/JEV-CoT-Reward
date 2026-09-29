#!/usr/bin/env python3
"""No-model multi-step deterministic replay for frozen ScienceWorld prefixes."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def replay_task(env_a: Any, env_b: Any, task: str, variation: int, protocol: dict) -> dict:
    for env in (env_a, env_b):
        env.load(task, variationIdx=variation, simplificationStr="", generateGoldPath=False)
    reset_a, reset_b = env_a.reset(), env_b.reset()
    if canonical_hash(reset_a) != canonical_hash(reset_b):
        raise ValueError(f"initial states differ for {task}/{variation}")
    seed_text = f"{protocol['base_seed']}:{task}:{variation}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "big")
    rng = random.Random(seed)
    steps = []
    for index in range(protocol["max_steps_per_prefix"]):
        legal_a = sorted(env_a.get_valid_action_object_combinations())
        legal_b = sorted(env_b.get_valid_action_object_combinations())
        if legal_a != legal_b:
            raise ValueError(f"legal actions differ at {task}/{variation}/{index}")
        if not legal_a:
            break
        action = legal_a[rng.randrange(len(legal_a))]
        if action not in legal_b:
            raise ValueError(f"selected action is not legal in replay environment at {task}/{variation}/{index}")
        step_a, step_b = env_a.step(action), env_b.step(action)
        state_a, state_b = canonical_hash(step_a), canonical_hash(step_b)
        if state_a != state_b:
            raise ValueError(f"state digests differ at {task}/{variation}/{index}")
        steps.append({
            "step": index + 1,
            "action_sha256": canonical_hash(action),
            "legal_action_count": len(legal_a),
            "state_sha256": state_a,
            "matched": True,
            "terminated": bool(step_a[2]),
        })
        if step_a[2]:
            break
    return {"task": task, "variation": variation, "step_count": len(steps), "steps": steps}


def run(source_root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    source_root = source_root.resolve()
    version_file = source_root / "scienceworld" / "version.py"
    if not version_file.exists():
        from zipfile import ZipFile
        text = ZipFile(source_root / "scienceworld" / "scienceworld.jar").read("META-INF/MANIFEST.MF").decode()
        version = next(line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("Specification-Version:"))
        version_file.write_text(f"__version__ = {version!r}\n")
    import sys
    sys.path.insert(0, str(source_root))
    from scienceworld import ScienceWorldEnv, __version__
    if __version__ != protocol["expected_scienceworld_version"]:
        raise ValueError(f"ScienceWorld version mismatch: {__version__}")
    env_a, env_b = ScienceWorldEnv(), ScienceWorldEnv()
    records = []
    try:
        for task, variations in protocol["heldout_variation_tasks"].items():
            for variation in variations:
                records.append(replay_task(env_a, env_b, task, variation, protocol))
    finally:
        env_a.close()
        env_b.close()
    result = {
        "protocol": protocol["protocol"],
        "source_revision": protocol["source_revision"],
        "scienceworld_version": __version__,
        "python_version": platform.python_version(),
        "status": "PASS_NO_MODEL" if all(all(step["matched"] for step in row["steps"]) for row in records) else "FAIL",
        "episode_count": len(records),
        "max_steps_per_prefix": protocol["max_steps_per_prefix"],
        "replay_records": records,
        "model_inference_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "gold_paths_requested": False,
        "raw_observations_written": False,
        "raw_action_strings_written": False,
        "task_score_or_reward_values_written": False,
        "protocol_sha256": canonical_hash(protocol),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root, args.protocol, args.output)
    print(json.dumps({"status": result["status"], "episodes": result["episode_count"], "steps": sum(row["step_count"] for row in result["replay_records"]), "model_inference_calls": 0, "jev_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
