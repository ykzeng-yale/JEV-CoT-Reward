#!/usr/bin/env python3
"""Build a label-free, exact-observation-group-disjoint train/dev manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def build(result_path: Path) -> dict:
    raw = result_path.read_bytes()
    result = json.loads(raw)
    if result.get("protocol") != "scienceworld_conductivity_visible_index_v1":
        raise ValueError("unexpected input protocol")
    if result.get("test_ids_loaded") is not False or result.get("split_counts", {}).get("test_loaded") != 0:
        raise ValueError("test data must remain unopened")
    if result.get("answer_labels_or_values_read") is not False:
        raise ValueError("split construction must use the label-free index result")
    if any(result.get(key) != 0 for key in ("model_calls", "jev_calls", "network_calls")):
        raise ValueError("unexpected model, Jev, or network access")
    rows = result.get("records")
    if not isinstance(rows, list) or len(rows) != 450:
        raise ValueError("expected 450 train/dev rows")

    by_hash: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"train": [], "dev": []})
    for row in rows:
        # Deliberately read only these three fields; label comparisons/outcomes are not inputs.
        split, variation, digest = row.get("split"), row.get("variation_id"), row.get("observation_sha256")
        if split not in ("train", "dev") or not isinstance(variation, int):
            raise ValueError("invalid split or variation id")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid observation digest")
        by_hash[digest][split].append(variation)
    if sorted(v for group in by_hash.values() for v in group["train"]) != list(range(300)):
        raise ValueError("training coverage mismatch")
    if sorted(v for group in by_hash.values() for v in group["dev"]) != list(range(300, 450)):
        raise ValueError("development coverage mismatch")

    train_hashes = {digest for digest, group in by_hash.items() if group["train"]}
    kept_hashes = sorted(
        digest for digest, group in by_hash.items() if group["dev"] and digest not in train_hashes
    )
    excluded_hashes = sorted(
        digest for digest, group in by_hash.items() if group["dev"] and digest in train_hashes
    )
    retained_ids = sorted(v for digest in kept_hashes for v in by_hash[digest]["dev"])
    excluded_ids = sorted(v for digest in excluded_hashes for v in by_hash[digest]["dev"])
    if set(kept_hashes) & train_hashes:
        raise AssertionError("prompt group crossed derived train/dev split")

    train_group_sizes = Counter(len(group["train"]) for group in by_hash.values() if group["train"])
    return {
        "protocol": "scienceworld_conductivity_prompt_group_disjoint_split_v1",
        "upstream_protocol": result["protocol"],
        "upstream_result_sha256": hashlib.sha256(raw).hexdigest(),
        "source_row_hash_field": "observation_sha256",
        "split_policy": "Keep official train IDs; retain only official dev rows whose exact initial-observation hash is absent from train. Exclude every dev member of a shared hash group.",
        "official_train_ids": list(range(300)),
        "retained_dev_ids": retained_ids,
        "excluded_dev_ids_with_train_hash_match": excluded_ids,
        "retained_dev_observation_hashes": kept_hashes,
        "excluded_dev_observation_hashes": excluded_hashes,
        "counts": {
            "official_train_rows": 300,
            "official_train_unique_observation_groups": len(train_hashes),
            "official_dev_rows": 150,
            "official_dev_unique_observation_groups": len({h for h, g in by_hash.items() if g["dev"]}),
            "retained_hash_disjoint_dev_rows": len(retained_ids),
            "retained_hash_disjoint_dev_groups": len(kept_hashes),
            "excluded_dev_rows": len(excluded_ids),
            "excluded_shared_dev_groups": len(excluded_hashes),
            "remaining_exact_hash_overlap": 0,
            "test_rows_read": 0,
        },
        "train_group_size_frequency": {str(size): train_group_sizes[size] for size in sorted(train_group_sizes)},
        "interpretation_limit": "Derived hash-disjoint development manifest only. It is not an efficacy analysis; 30 retained dev groups are too few for small-effect claims, the 300-row training set itself contains conflicting targets for some duplicate observations, and the final split was not loaded.",
        "source_label_values_read": False,
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
    report = build(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
