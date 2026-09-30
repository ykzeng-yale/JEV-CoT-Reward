#!/usr/bin/env python3
"""Repeat one public first action; emit hashes only, never endpoint fields."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

from jev_control.scienceworld_conductivity_action_audit import _full_view_hash


def h(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def view_parts(env, observation):
    description = env.taskdescription()
    actions = sorted(set(env.get_valid_action_object_combinations()))
    canonical_actions = json.dumps(actions, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "full_view_sha256": _full_view_hash(env, observation),
        "task_description_sha256": h(description.encode("utf-8")),
        "observation_sha256": h(observation.encode("utf-8")),
        "legal_actions_sha256": h(canonical_actions),
        "legal_action_count": len(actions),
    }


def main(protocol_path, result_path, source_root, output_path, variation, policy, repetitions):
    protocol = json.loads(protocol_path.read_bytes())
    result = json.loads(result_path.read_bytes())
    row = next(item for item in result["episodes"]
               if item["variation_id"] == variation and item["policy"] == policy)
    recorded_initial = row["initial_view_sha256"]
    recorded_first_post = row["trace"][0]["view_sha256"]
    action = row["trace"][0]["action"]
    archive = source_root / "scienceworld" / "scienceworld.jar"
    version_file = source_root / "scienceworld" / "version.py"
    if not version_file.exists():
        manifest = ZipFile(archive).read("META-INF/MANIFEST.MF").decode()
        version = next(line.split(": ", 1)[1] for line in manifest.splitlines()
                       if line.startswith("Specification-Version:"))
        version_file.write_text(f"__version__ = {version!r}\n")
    sys.path.insert(0, str(source_root))
    from scienceworld import ScienceWorldEnv, __version__
    if __version__ != protocol["expected_scienceworld_version"]:
        raise ValueError("pinned runtime mismatch")
    env = ScienceWorldEnv(envStepLimit=100)
    runs = []
    try:
        for repetition in range(repetitions):
            env.load(protocol["task"], variationIdx=variation,
                     simplificationStr="", generateGoldPath=False)
            observation, _ = env.reset()
            initial = view_parts(env, observation)
            legal = env.get_valid_action_object_combinations()
            if action not in legal:
                runs.append({"repetition": repetition, "action_legal": False,
                             "initial": initial, "post": None})
                continue
            observation, _, _, _ = env.step(action)
            runs.append({"repetition": repetition, "action_legal": True,
                         "initial": initial, "post": view_parts(env, observation)})
    finally:
        env.close()
    summary = {
        "variation_id": variation, "policy": policy, "repetitions": repetitions,
        "action": action, "recorded_initial_full_view_sha256": recorded_initial,
        "recorded_first_post_view_sha256": recorded_first_post,
        "runs": runs,
        "initial_all_match_record": all(r["initial"]["full_view_sha256"] == recorded_initial for r in runs),
        "post_all_match_record": all(r["post"] is not None and r["post"]["full_view_sha256"] == recorded_first_post for r in runs),
        "post_run_hashes_stable": len({r["post"]["full_view_sha256"] for r in runs if r["post"]}) <= 1,
        "outcome_fields_read": False,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("variation_id", "policy", "repetitions", "action", "initial_all_match_record", "post_all_match_record", "post_run_hashes_stable", "outcome_fields_read")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol", "result", "source-root", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--variation", type=int, required=True)
    p.add_argument("--policy", choices=["measurement", "masked_measurement", "random_measurement", "continue_prior"], required=True)
    p.add_argument("--repetitions", type=int, choices=[3], default=3)
    args = p.parse_args()
    main(args.protocol, args.result, args.source_root, args.output,
         args.variation, args.policy, args.repetitions)
