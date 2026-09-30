#!/usr/bin/env python3
"""Independently audit coverage, exact-input group consistency, and exclusions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


def _verify_source(protocol: dict, source_root: Path) -> None:
    expected = {name: v["sha256"] for name, v in protocol["source_members"].items()}
    actual = {}
    for name, record in protocol["source_members"].items():
        relative = record["path"].removeprefix("scienceworld-source/")
        actual[name] = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("pinned source member hash mismatch")


def _independently_replay(protocol: dict, source_root: Path) -> list[dict]:
    version_file = source_root / "scienceworld" / "version.py"
    if not version_file.exists():
        manifest = ZipFile(source_root / "scienceworld" / "scienceworld.jar").read("META-INF/MANIFEST.MF").decode("utf-8")
        version = next(line.split(": ", 1)[1] for line in manifest.splitlines() if line.startswith("Specification-Version:"))
        version_file.write_text(f"__version__ = {version!r}\n", encoding="utf-8")
    import sys
    sys.path.insert(0, str(source_root))
    from scienceworld import ScienceWorldEnv, __version__
    if __version__ != protocol["expected_scienceworld_version"]:
        raise ValueError("pinned ScienceWorld version mismatch")
    env = ScienceWorldEnv()
    rows = []
    labels_by_hash: dict[str, bool] = {}
    try:
        task = protocol["task"]
        for split in ("train", "dev"):
            span = protocol["splits"][split]
            for variation in range(span["start_inclusive"], span["end_exclusive"]):
                env.load(task, variationIdx=variation, simplificationStr="", generateGoldPath=False)
                observation, info = env.reset()
                if not isinstance(observation, str) or not isinstance(info, dict):
                    raise TypeError("unexpected reset output")
                digest = hashlib.sha256(observation.encode("utf-8")).hexdigest()
                interface = env.server.agentInterface().get()
                modifiers = list(interface.task().taskModifiers())
                values = []
                for modifier in modifiers:
                    if str(modifier.getClass().getSimpleName()) != "TaskValueBool":
                        continue
                    if str(modifier.key()) == protocol["label_key"]:
                        values.append(bool(modifier.value()))
                if len(values) != 1:
                    raise ValueError("source label modifier cardinality mismatch")
                label = values[0]
                match = digest not in labels_by_hash or label == labels_by_hash[digest]
                labels_by_hash.setdefault(digest, label)
                rows.append((split, variation, digest, match))
    finally:
        env.close()
    return rows


def audit(protocol_path: Path, result_path: Path, source_root: Path | None = None) -> dict:
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    result = json.loads(result_path.read_text())
    if result.get("protocol") != protocol["protocol"]:
        raise ValueError("protocol mismatch")
    if result.get("protocol_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("exact protocol byte digest mismatch")
    if result.get("upstream_result_sha256") != protocol["upstream_result_sha256"]:
        raise ValueError("upstream result digest mismatch")
    expected_hashes = {name: v["sha256"] for name, v in protocol["source_members"].items()}
    if result.get("source_member_sha256") != expected_hashes:
        raise ValueError("source member contract mismatch")
    rows = result.get("records")
    if not isinstance(rows, list) or len(rows) != 450:
        raise ValueError("expected exactly 450 development rows")
    expected = {
        split: set(range(spec["start_inclusive"], spec["end_exclusive"]))
        for split, spec in protocol["splits"].items() if split in ("train", "dev")
    }
    ids = {split: [] for split in expected}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        split = row.get("split")
        if split not in expected:
            raise ValueError("non-development split present")
        ids[split].append(row.get("variation_id"))
        digest = row.get("observation_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid observation digest")
        if not isinstance(row.get("label_matches_lowest_id_same_observation"), bool):
            raise ValueError("missing label-consistency comparison")
        groups[digest].append(row)
    if any(set(ids[s]) != expected[s] for s in expected):
        raise ValueError("train/dev variation coverage mismatch")
    for group in groups.values():
        reference = min(group, key=lambda r: r["variation_id"])
        if reference["label_matches_lowest_id_same_observation"] is not True:
            raise ValueError("reference row must match itself")
    shared = [g for g in groups.values() if {r["split"] for r in g} == {"train", "dev"}]
    conflicted = [g for g in groups.values() if any(not r["label_matches_lowest_id_same_observation"] for r in g)]
    shared_conflicted = [g for g in shared if any(not r["label_matches_lowest_id_same_observation"] for r in g)]
    if result.get("split_counts") != {"train": 300, "dev": 150, "test_loaded": 0}:
        raise ValueError("split counts violate frozen scope")
    if result.get("raw_observations_written") is not False or result.get("answer_label_values_written") is not False:
        raise ValueError("raw observation or answer label exported")
    if result.get("source_defined_label_values_read") is not True:
        raise ValueError("required source label read not recorded")
    if result.get("test_ids_loaded") is not False or result.get("gold_paths_requested") is not False:
        raise ValueError("test/gold-path scope exclusion failed")
    if any(result.get(k) != 0 for k in ("score_reward_values_read", "model_calls", "jev_calls", "network_calls")):
        raise ValueError("forbidden evaluation/model/network access recorded")
    if source_root is not None:
        _verify_source(protocol, source_root)
        replay = sorted(_independently_replay(protocol, source_root))
        observed = sorted((r["split"], r["variation_id"], r["observation_sha256"], r["label_matches_lowest_id_same_observation"]) for r in rows)
        if replay != observed:
            raise ValueError("independent environment replay disagrees with row-level observation/label comparisons")
    return {
        "protocol": protocol["protocol"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "independent_audit": "PASS",
        "train_rows": 300,
        "dev_rows": 150,
        "test_rows": 0,
        "unique_initial_observation_groups": len(groups),
        "cross_split_shared_exact_input_groups": len(shared),
        "cross_split_shared_rows": sum(len(g) for g in shared),
        "label_conflict_groups": len(conflicted),
        "cross_split_label_conflict_groups": len(shared_conflicted),
        "cross_split_label_agreement_groups": len(shared) - len(shared_conflicted),
        "source_defined_label_values_written": False,
        "model_calls": 0,
        "jev_calls": 0,
        "interpretation_limit": "Group comparisons characterize label identifiability under exact initial-observation equality on train/dev only; they do not estimate agent performance or intervention value.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    a = p.parse_args()
    if a.audit_output.exists():
        p.error("refusing to overwrite audit output")
    report = audit(a.protocol, a.result, a.source_root)
    a.audit_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
