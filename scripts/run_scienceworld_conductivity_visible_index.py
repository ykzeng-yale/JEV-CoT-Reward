#!/usr/bin/env python3
"""Aggregate-only train/dev audit of visible task-combination index recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recover_index(text: str, rule: dict) -> tuple[int | None, int, bool]:
    """Decode only the public task-combination dimensions, never its label."""
    letter_order = rule["letter_order"]
    letters = set(re.findall(r"unknown substance ([A-Z])", text, re.IGNORECASE))
    if len(letters) != 1:
        return None, 0, False
    letter = next(iter(letters)).upper()
    if letter not in letter_order:
        return None, 0, True
    boxes = rule["answer_box_pairs"]
    box_hits = []
    lower = text.lower()
    for i, pair in enumerate(boxes):
        # The task description identifies which box encodes each label.
        if all(name in lower for name in pair):
            box_hits.append(i)
    if len(box_hits) != 1:
        return None, 0, True
    parts = [name for name in rule["part_names"] if name in lower]
    if len(parts) != 1:
        return None, len(parts), True
    part_index = rule["part_names"].index(parts[0])
    return letter_order.index(letter) * 25 + part_index * 5 + box_hits[0], 1, True


def verify_source_contract(protocol: dict, source_root: Path) -> dict[str, str]:
    sources = {}
    for name, record in protocol["source_members"].items():
        relative = record["path"].removeprefix("scienceworld-source/")
        data = (source_root / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"pinned source hash mismatch: {name}")
        sources[name] = data.decode("utf-8")
    checks = {
        "conductivity_task": (
            "val combinations = for {", "m <- unknownSubstancesSorted", "j <- partToPower", "n <- answerBoxes",
            'new TaskValueBool(key = "unknownIsConductive"',
            'description = "Your task is to determine if " + unknownSubstanceName.get',
            'val validLetters = Array("B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")',
            'val answerBoxColors = Array("red", "green", "blue", "orange", "yellow", "purple")',
            'val lightColors = Array("red", "green", "blue")',
        ),
        "unknown_substances": ("val randDouble = Random.nextFloat()", "if (randDouble < 0.50)"),
        "python_interface": ("Random.setSeed(variationIdx)", "taskMaker = new TaskMaker1()"),
        "answer_box": ('this.name = colourName + " box"',),
        "light_bulb": ('this.name = (color + " light bulb").trim',),
        "electric_motor": ('this.name = "electric motor"',),
        "electric_buzzer": ('this.name = "electric buzzer"',),
    }
    for name, required in checks.items():
        if any(token not in sources[name] for token in required):
            raise ValueError(f"pinned source no longer matches combination contract: {name}")
    return {name: protocol["source_members"][name]["sha256"] for name in sources}


def run(protocol_path: Path, source_root: Path, output_path: Path) -> dict:
    raw_protocol = protocol_path.read_bytes()
    protocol = json.loads(raw_protocol)
    member_hashes = verify_source_contract(protocol, source_root)
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
    records = []
    try:
        task = protocol["task"]
        env.load(task, variationIdx=0, simplificationStr="", generateGoldPath=False)
        train = list(env.get_variations_train())
        dev = list(env.get_variations_dev())
        expected_train = list(range(protocol["splits"]["train"]["start_inclusive"], protocol["splits"]["train"]["end_exclusive"]))
        expected_dev = list(range(protocol["splits"]["dev"]["start_inclusive"], protocol["splits"]["dev"]["end_exclusive"]))
        if train != expected_train or dev != expected_dev or set(train) & set(dev):
            raise ValueError(f"official train/dev split mismatch ({len(train)}, {len(dev)})")
        for split, variations in (("train", train), ("dev", dev)):
            for variation in variations:
                env.load(task, variationIdx=variation, simplificationStr="", generateGoldPath=False)
                observation, info = env.reset()
                if not isinstance(observation, str) or not isinstance(info, dict):
                    raise TypeError("unexpected reset return types")
                recovered, candidates, letter_visible = recover_index(observation, protocol["recovery_rule"])
                records.append({
                    "split": split,
                    "variation_id": variation,
                    "observation_sha256": sha(observation),
                    "letter_visible": letter_visible,
                    "candidate_part_count": candidates,
                    "source_index_decoded": recovered is not None,
                    "source_index_recovered": recovered == variation if recovered is not None else False,
                    "variation_idx_in_observation": "variationIdx" in observation,
                })
    finally:
        env.close()
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": hashlib.sha256(raw_protocol).hexdigest(),
        "scienceworld_version": __version__,
        "source_member_sha256": member_hashes,
        "split_counts": {"train": len(train), "dev": len(dev), "test_loaded": 0},
        "row_count": len(records),
        "train_index_recovery_count": sum(r["source_index_recovered"] for r in records if r["split"] == "train"),
        "dev_index_recovery_count": sum(r["source_index_recovered"] for r in records if r["split"] == "dev"),
        "unique_observation_hash_count": len({r["observation_sha256"] for r in records}),
        "variation_idx_in_observation_count": sum(r["variation_idx_in_observation"] for r in records),
        "records": records,
        "raw_observations_written": False,
        "answer_labels_or_values_read": False,
        "score_reward_values_read": False,
        "test_ids_loaded": False,
        "model_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "gold_paths_requested": False,
        "interpretation_limit": "The index decoder uses only the frozen public combination order and observation text; a recovered index makes the seeded label source-reproducible but does not measure model accuracy or action efficacy. A failed decoder is limited to the specified features and does not establish absence of every possible semantic leak.",
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.protocol, args.source_root, args.output)
    print(json.dumps({k: result[k] for k in ("protocol", "row_count", "split_counts", "train_index_recovery_count", "dev_index_recovery_count", "model_calls", "jev_calls")}, sort_keys=True))


if __name__ == "__main__":
    main()
