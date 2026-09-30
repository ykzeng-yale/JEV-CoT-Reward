#!/usr/bin/env python3
"""Train/dev-only audit of label consistency for identical initial observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_source(protocol: dict, source_root: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    for name, record in protocol["source_members"].items():
        relative = record["path"].removeprefix("scienceworld-source/")
        digest = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"pinned source hash mismatch: {name}")
        members[name] = digest
    return members


def source_label(env) -> bool:
    """Read the named source task modifier, never score/reward or goal progress."""
    interface = env.server.agentInterface().get()
    modifiers = list(interface.task().taskModifiers())
    # Task.taskModifiers also contains TaskObject and other TaskModifier
    # subclasses. Only TaskValueBool exposes key/value; calling key() on the
    # first arbitrary modifier fails through Py4J before reaching the label.
    values = []
    for mod in modifiers:
        if str(mod.getClass().getSimpleName()) != "TaskValueBool":
            continue
        if str(mod.key()) == "unknownIsConductive":
            values.append(bool(mod.value()))
    if len(values) != 1:
        raise ValueError(f"expected one source label modifier, found {len(values)}")
    return values[0]


def run(protocol_path: Path, source_root: Path, output_path: Path) -> dict:
    raw_protocol = protocol_path.read_bytes()
    protocol = json.loads(raw_protocol)
    member_hashes = verify_source(protocol, source_root)
    archive = source_root / "scienceworld" / "scienceworld.jar"
    version_file = source_root / "scienceworld" / "version.py"
    if not version_file.exists():
        manifest = ZipFile(archive).read("META-INF/MANIFEST.MF").decode("utf-8")
        version = next(line.split(": ", 1)[1] for line in manifest.splitlines() if line.startswith("Specification-Version:"))
        version_file.write_text(f"__version__ = {version!r}\n", encoding="utf-8")
    sys.path.insert(0, str(source_root))
    from scienceworld import ScienceWorldEnv, __version__

    if __version__ != protocol["expected_scienceworld_version"]:
        raise ValueError(f"ScienceWorld version mismatch: {__version__}")
    env = ScienceWorldEnv()
    rows: list[dict] = []
    seen_labels: dict[str, bool] = {}
    try:
        task = protocol["task"]
        env.load(task, variationIdx=0, simplificationStr="", generateGoldPath=False)
        train = list(env.get_variations_train())
        dev = list(env.get_variations_dev())
        expected = {
            split: list(range(spec["start_inclusive"], spec["end_exclusive"]))
            for split, spec in protocol["splits"].items() if split in ("train", "dev")
        }
        if train != expected["train"] or dev != expected["dev"] or set(train) & set(dev):
            raise ValueError("official train/dev split mismatch")
        for split, variations in (("train", train), ("dev", dev)):
            for variation in variations:
                env.load(task, variationIdx=variation, simplificationStr="", generateGoldPath=False)
                observation, info = env.reset()
                if not isinstance(observation, str) or not isinstance(info, dict):
                    raise TypeError("unexpected reset return types")
                observation_hash = sha(observation)
                label = source_label(env)
                if observation_hash in seen_labels:
                    matches = label == seen_labels[observation_hash]
                else:
                    matches = True
                    seen_labels[observation_hash] = label
                rows.append({
                    "split": split,
                    "variation_id": variation,
                    "observation_sha256": observation_hash,
                    "label_matches_lowest_id_same_observation": matches,
                })
    finally:
        env.close()
    if len(rows) != 450:
        raise ValueError(f"expected 450 development records, got {len(rows)}")
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": hashlib.sha256(raw_protocol).hexdigest(),
        "upstream_result_sha256": protocol["upstream_result_sha256"],
        "scienceworld_version": __version__,
        "source_member_sha256": member_hashes,
        "split_counts": {"train": len(train), "dev": len(dev), "test_loaded": 0},
        "records": rows,
        "raw_observations_written": False,
        "answer_label_values_written": False,
        "source_defined_label_values_read": True,
        "score_reward_values_read": False,
        "test_ids_loaded": False,
        "model_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "gold_paths_requested": False,
        "interpretation_limit": "This source-validity audit measures exact initial-input label consistency only on development variations. It is not agent accuracy, action efficacy, or a test-set estimate.",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) > protocol["limits"]["max_result_bytes"]:
        raise ValueError("result exceeds frozen output-size limit")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.source_root, args.output)
    print(json.dumps({k: result[k] for k in ("protocol", "split_counts", "raw_observations_written", "answer_label_values_written", "source_defined_label_values_read", "model_calls", "jev_calls")}, sort_keys=True))


if __name__ == "__main__":
    main()
