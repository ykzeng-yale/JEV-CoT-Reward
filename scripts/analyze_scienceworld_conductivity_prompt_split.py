#!/usr/bin/env python3
"""Quantify initial-observation overlap and cluster-size sensitivity by split."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def analyze(result_path: Path) -> dict:
    raw = result_path.read_bytes()
    result = json.loads(raw)
    if result.get("protocol") != "scienceworld_conductivity_visible_index_v1":
        raise ValueError("unexpected upstream protocol")
    if result.get("split_counts", {}).get("test_loaded") != 0 or result.get("test_ids_loaded") is not False:
        raise ValueError("test rows must remain unopened")
    if result.get("answer_labels_or_values_read") is not False:
        raise ValueError("upstream must not contain answer-label values")
    rows = result.get("records")
    if not isinstance(rows, list) or len(rows) != 450:
        raise ValueError("expected exactly 450 train/dev records")

    split_rows: dict[str, list[dict]] = {"train": [], "dev": []}
    ids: dict[str, list[int]] = {"train": [], "dev": []}
    for row in rows:
        split = row.get("split")
        if split not in split_rows:
            raise ValueError("non-train/dev row present")
        digest = row.get("observation_sha256")
        variation = row.get("variation_id")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid observation hash")
        if not isinstance(variation, int):
            raise ValueError("invalid variation id")
        split_rows[split].append(row)
        ids[split].append(variation)
    if set(ids["train"]) != set(range(300)) or set(ids["dev"]) != set(range(300, 450)):
        raise ValueError("train/dev variation coverage mismatch")

    groups = {
        split: collections.Counter(row["observation_sha256"] for row in values)
        for split, values in split_rows.items()
    }
    shared = set(groups["train"]) & set(groups["dev"])
    dev_train_seen_rows = sum(groups["dev"][digest] for digest in shared)

    def summarize(counts: collections.Counter) -> dict:
        size_frequency = collections.Counter(counts.values())
        n_rows = sum(counts.values())
        sum_squared_cluster_sizes = sum(size * size * frequency for size, frequency in size_frequency.items())
        return {
            "rows": n_rows,
            "unique_initial_observation_hashes": len(counts),
            "cluster_size_frequency": {str(size): size_frequency[size] for size in sorted(size_frequency)},
            "kish_effective_n_if_within_hash_outcomes_are_perfectly_dependent": round(
                n_rows * n_rows / sum_squared_cluster_sizes, 3
            ),
        }

    return {
        "protocol": result["protocol"],
        "upstream_result_sha256": hashlib.sha256(raw).hexdigest(),
        "split_diagnostic": {
            "train": summarize(groups["train"]),
            "dev": summarize(groups["dev"]),
            "cross_split_shared_exact_hash_groups": len(shared),
            "train_rows_in_cross_split_shared_groups": sum(groups["train"][h] for h in shared),
            "dev_rows_with_an_exact_train_input": dev_train_seen_rows,
            "dev_rows_without_an_exact_train_input": len(split_rows["dev"]) - dev_train_seen_rows,
            "dev_distinct_hashes_without_an_exact_train_input": len(set(groups["dev"]) - set(groups["train"])),
        },
        "interpretation_limit": (
            "The perfect-dependence Kish effective-n value is a cluster-size sensitivity, not an estimate of "
            "outcome ICC, model generalization, power, label consistency, or action efficacy. Exact input overlap "
            "invalidates prompt-level train/dev independence for the overlapping groups; label outcomes were not read."
        ),
        "source_defined_label_values_read": False,
        "test_rows_read": 0,
        "model_calls": 0,
        "jev_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    report = analyze(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["split_diagnostic"], sort_keys=True))


if __name__ == "__main__":
    main()
