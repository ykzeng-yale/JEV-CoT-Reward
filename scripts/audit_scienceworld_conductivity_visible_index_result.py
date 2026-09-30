#!/usr/bin/env python3
"""Independently validate aggregate-only conductivity index-recovery output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(protocol_path: Path, result_path: Path) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    result = json.loads(result_path.read_text())
    if result.get("protocol") != protocol["protocol"]:
        raise ValueError("protocol mismatch")
    if result.get("protocol_sha256") != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError("exact protocol-byte digest mismatch")
    expected_hashes = {name: record["sha256"] for name, record in protocol["source_members"].items()}
    if result.get("source_member_sha256") != expected_hashes:
        raise ValueError("source member hashes differ from frozen protocol")
    expected = {
        split: list(range(spec["start_inclusive"], spec["end_exclusive"]))
        for split, spec in protocol["splits"].items() if split in ("train", "dev")
    }
    rows = result.get("records")
    if not isinstance(rows, list) or len(rows) != 450:
        raise ValueError("expected exactly 450 train/dev-only records")
    ids = {split: [] for split in expected}
    for row in rows:
        split = row.get("split")
        if split not in expected:
            raise ValueError("non-train/dev record present")
        ids[split].append(row.get("variation_id"))
        if row.get("observation_sha256") is None or len(row["observation_sha256"]) != 64:
            raise ValueError("invalid observation hash")
        if row.get("variation_idx_in_observation"):
            raise ValueError("variationIdx field name appeared in controller input")
        if not isinstance(row.get("source_index_recovered"), bool):
            raise ValueError("missing aggregate-safe decoder flag")
    for split in expected:
        if sorted(ids[split]) != expected[split]:
            raise ValueError(f"{split} coverage mismatch")
    if result.get("split_counts") != {"train": 300, "dev": 150, "test_loaded": 0}:
        raise ValueError("split counts violate frozen scope")
    if result.get("row_count") != 450:
        raise ValueError("row count mismatch")
    if result.get("unique_observation_hash_count") != len({r["observation_sha256"] for r in rows}):
        raise ValueError("unique observation-hash aggregate mismatch")
    train = sum(r["source_index_recovered"] for r in rows if r["split"] == "train")
    dev = sum(r["source_index_recovered"] for r in rows if r["split"] == "dev")
    decoded = sum(r["source_index_decoded"] for r in rows)
    if decoded != sum(bool(r.get("letter_visible")) and r.get("candidate_part_count") == 1 for r in rows):
        raise ValueError("decoder-applied count disagrees with visible-feature flags")
    if result.get("variation_idx_in_observation_count") != sum(r["variation_idx_in_observation"] for r in rows):
        raise ValueError("visible variation-index field aggregate mismatch")
    if (train, dev) != (result.get("train_index_recovery_count"), result.get("dev_index_recovery_count")):
        raise ValueError("recovery aggregates disagree with rows")
    zero_claims = ("raw_observations_written", "answer_labels_or_values_read", "score_reward_values_read", "test_ids_loaded", "gold_paths_requested")
    if any(result.get(key) is not False for key in zero_claims):
        raise ValueError("scope exclusion flag not false")
    if result.get("model_calls") != 0 or result.get("jev_calls") != 0 or result.get("network_calls") != 0:
        raise ValueError("forbidden model, Jev, or network call recorded")
    hash_splits: dict[str, set[str]] = {}
    hash_rows: dict[str, int] = {}
    rows_by_split_hash: dict[str, dict[str, int]] = {split: {} for split in expected}
    split_hashes = {split: set() for split in expected}
    for row in rows:
        digest = row["observation_sha256"]
        hash_splits.setdefault(digest, set()).add(row["split"])
        hash_rows[digest] = hash_rows.get(digest, 0) + 1
        split_hashes[row["split"]].add(digest)
        rows_by_split_hash[row["split"]][digest] = rows_by_split_hash[row["split"]].get(digest, 0) + 1
    shared_hashes = [digest for digest, splits in hash_splits.items() if splits == {"train", "dev"}]
    train_shared_rows = sum(rows_by_split_hash["train"].get(digest, 0) for digest in shared_hashes)
    dev_shared_rows = sum(rows_by_split_hash["dev"].get(digest, 0) for digest in shared_hashes)
    return {
        "protocol": protocol["protocol"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "status": "INDEX_RECOVERY_OBSERVED" if decoded else "NO_INDEX_RECOVERY_WITH_DECLARED_DECODER",
        "independent_audit": "PASS",
        "train_rows": len(ids["train"]),
        "dev_rows": len(ids["dev"]),
        "test_rows": 0,
        "unique_train_observation_hashes": len(split_hashes["train"]),
        "unique_dev_observation_hashes": len(split_hashes["dev"]),
        "cross_split_shared_exact_observation_hashes": len(shared_hashes),
        "cross_split_shared_rows": sum(hash_rows[digest] for digest in shared_hashes),
        "train_rows_with_cross_split_duplicate_input": train_shared_rows,
        "dev_rows_with_train_seen_input": dev_shared_rows,
        "dev_unique_inputs_not_seen_in_train": len(split_hashes["dev"]) - len(shared_hashes),
        "train_index_recovery_count": train,
        "dev_index_recovery_count": dev,
        "decoder_applied_count": decoded,
        "model_calls": 0,
        "jev_calls": 0,
        "interpretation_limit": "A nonzero recovery rate indicates only that this fixed public feature decoder recovered combination indices; it is not model accuracy or action efficacy. Failure of this decoder cannot rule out other semantic reconstruction routes.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    a = p.parse_args()
    if a.audit_output.exists():
        p.error("refusing to overwrite audit output")
    result = audit(a.protocol, a.result)
    a.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
