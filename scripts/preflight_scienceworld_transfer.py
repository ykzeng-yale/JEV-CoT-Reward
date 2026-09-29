#!/usr/bin/env python3
"""No-model portability, split-integrity, and deterministic-replay preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_partition(max_variations: int, train: list[int], dev: list[int], test: list[int]) -> dict:
    expected = set(range(max_variations))
    sets = {"train": set(train), "dev": set(dev), "test": set(test)}
    if any(not values <= expected for values in sets.values()):
        raise ValueError("variation split contains out-of-range indices")
    if sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"]:
        raise ValueError("variation split sets overlap")
    if set.union(*sets.values()) != expected:
        raise ValueError("variation split does not cover every supported index")
    if any(len(sets[name]) != len(values) for name, values in (("train", train), ("dev", dev), ("test", test))):
        raise ValueError("variation split contains duplicate indices")
    return {name: len(sets[name]) for name in ("train", "dev", "test")}


def sample_indices(test: list[int]) -> list[int]:
    if not test:
        raise ValueError("task has no official test variations")
    return list(dict.fromkeys((test[0], test[len(test) // 2], test[-1])))


def _step_digest(step: tuple) -> str:
    # Include return-value equality without persisting raw observations, scores, or rewards.
    return canonical_hash(step)


def run(manifest_path: Path, source_root: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = source_root.resolve()
    version_file = source_root / "scienceworld" / "version.py"
    if not version_file.exists():
        jar = source_root / "scienceworld" / "scienceworld.jar"
        from zipfile import ZipFile

        text = ZipFile(jar).read("META-INF/MANIFEST.MF").decode("utf-8")
        version = next(line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("Specification-Version:"))
        version_file.write_text(f"__version__ = {version!r}\n", encoding="utf-8")

    import sys

    sys.path.insert(0, str(source_root))
    from scienceworld import ScienceWorldEnv, __version__

    if __version__ != manifest["expected_scienceworld_version"]:
        raise ValueError(f"ScienceWorld version mismatch: {__version__}")

    env = ScienceWorldEnv()
    partitions: dict[str, dict] = {}
    try:
        task_names = env.get_task_names()
        if len(task_names) != manifest["expected_task_count"] or len(set(task_names)) != len(task_names):
            raise ValueError("unexpected task count or duplicate task names")
        if set(task_names) != set(manifest["task_variations"]):
            raise ValueError("runtime task names differ from the independently frozen README inventory")
        total_variations = 0
        for name in task_names:
            env.load(name, variationIdx=0, simplificationStr="", generateGoldPath=False)
            maximum = int(env.get_max_variations(name))
            if maximum != int(manifest["task_variations"][name]):
                raise ValueError(f"variation count differs from the frozen README inventory: {name}")
            split_counts = validate_partition(
                maximum,
                list(env.get_variations_train()),
                list(env.get_variations_dev()),
                list(env.get_variations_test()),
            )
            total_variations += maximum
            partitions[name] = {"variation_count": maximum, **split_counts}
        if total_variations != manifest["expected_variation_count"]:
            raise ValueError(f"unexpected supported variation total: {total_variations}")
    finally:
        env.close()

    replay_a = ScienceWorldEnv()
    replay_b = ScienceWorldEnv()
    replay_records = []
    try:
        for task_name in manifest["heldout_variation_tasks"]:
            if task_name not in partitions:
                raise ValueError(f"unknown replay task: {task_name}")
            replay_a.load(task_name, variationIdx=0, simplificationStr="", generateGoldPath=False)
            test_variations = list(replay_a.get_variations_test())
            for variation in sample_indices(test_variations)[: manifest["replay_samples_per_task"]]:
                replay_a.load(task_name, variationIdx=variation, simplificationStr="", generateGoldPath=False)
                replay_b.load(task_name, variationIdx=variation, simplificationStr="", generateGoldPath=False)
                reset_a = replay_a.reset()
                reset_b = replay_b.reset()
                actions_a = sorted(replay_a.get_valid_action_object_combinations())
                actions_b = sorted(replay_b.get_valid_action_object_combinations())
                if not actions_a or actions_a != actions_b:
                    raise ValueError(f"legal-action mismatch or empty action set: {task_name}/{variation}")
                action = actions_a[0]
                step_a = replay_a.step(action)
                step_b = replay_b.step(action)
                initial_equal = _step_digest(reset_a) == _step_digest(reset_b)
                post_action_equal = _step_digest(step_a) == _step_digest(step_b)
                if not initial_equal or not post_action_equal:
                    raise ValueError(f"independent reset/replay mismatch: {task_name}/{variation}")
                replay_records.append({
                    "task": task_name,
                    "variation": int(variation),
                    "action_sha256": canonical_hash(action),
                    "legal_action_count": len(actions_a),
                    "initial_state_sha256": _step_digest(reset_a),
                    "post_action_state_sha256": _step_digest(step_a),
                    "initial_equal": True,
                    "post_action_equal": True,
                })
    finally:
        replay_a.close()
        replay_b.close()

    result = {
        "protocol": manifest["protocol"],
        "status": "PASS_NO_MODEL",
        "source_revision": manifest["source_revision"],
        "scienceworld_version": __version__,
        "python_version": platform.python_version(),
        "task_count": len(partitions),
        "variation_count": sum(row["variation_count"] for row in partitions.values()),
        "split_integrity": "PASS: disjoint train/dev/test sets cover every supported variation",
        "partition_summary": partitions,
        "replay_count": len(replay_records),
        "replay_records": replay_records,
        "model_inference_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "raw_observations_written": False,
        "raw_action_strings_written": False,
        "task_score_or_reward_values_written": False,
        "gold_paths_requested": False,
        "manifest_sha256": canonical_hash(manifest),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.source_root, args.output)
    print(json.dumps({k: result[k] for k in (
        "status", "scienceworld_version", "task_count", "variation_count", "replay_count",
        "model_inference_calls", "jev_calls", "network_calls",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
