#!/usr/bin/env python3
"""Aggregate-only provenance screen for the public ARMAP ScienceWorld corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


IDENTIFIER_FIELDS = {"variationidx", "variation_idx", "variation_id", "variationid"}
ALLOWED_OUTPUT_KEYS = {
    "audit",
    "candidate_term_occurrences",
    "candidate_term_record_counts",
    "dataset",
    "explicit_heldout_variation_field_records",
    "identifier_key_names",
    "record_count",
    "record_schema_keys",
    "rows_with_explicit_heldout_variation_ids",
}


def _walk(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key), value
            if isinstance(value, (dict, list)):
                yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            if isinstance(value, (dict, list)):
                yield from _walk(value)
            else:
                yield "", value


def audit_bytes(payload: bytes, protocol: dict[str, Any]) -> dict[str, Any]:
    dataset = protocol["dataset"]
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != dataset["bytes"] or digest != dataset["sha256"]:
        raise ValueError("input does not match the frozen dataset identity")

    records = json.loads(payload)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("expected the frozen top-level list-of-records schema")

    terms = tuple(protocol["scan"]["candidate_terms"])
    term_occurrences = Counter({term: 0 for term in terms})
    term_records = Counter({term: 0 for term in terms})
    identifier_names: set[str] = set()
    explicit_heldout_field_records = 0
    rows_with_explicit_ids = 0
    lower, upper = protocol["candidate"]["heldout_variation_range"]

    for row in records:
        entries = list(_walk(row))
        keys = [key.lower() for key, _ in entries if key]
        identifier_names.update(
            key
            for key in keys
            if any(part in key for part in protocol["scan"]["identifier_key_fragments"])
        )
        explicit_ids: set[int] = set()
        for key, value in entries:
            normalized_key = key.lower()
            if normalized_key not in IDENTIFIER_FIELDS:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                numeric_value = value
            elif isinstance(value, str) and re.fullmatch(r"\d+", value):
                numeric_value = int(value)
            else:
                continue
            if lower <= numeric_value <= upper:
                explicit_ids.add(numeric_value)
        explicit_heldout_field_records += int(any(key in IDENTIFIER_FIELDS for key in keys))
        rows_with_explicit_ids += int(bool(explicit_ids))

        text = "\n".join(value for _, value in entries if isinstance(value, str)).casefold()
        for term in terms:
            occurrences = text.count(term.casefold())
            term_occurrences[term] += occurrences
            term_records[term] += int(occurrences > 0)

    schema_keys = sorted({key for row in records for key in row})
    return {
        "audit": "aggregate_only_training_corpus_provenance_screen",
        "candidate_term_occurrences": dict(term_occurrences),
        "candidate_term_record_counts": dict(term_records),
        "dataset": {
            "repo": dataset["repo"],
            "revision": dataset["revision"],
            "file": dataset["file"],
            "bytes": len(payload),
            "sha256": digest,
        },
        "explicit_heldout_variation_field_records": explicit_heldout_field_records,
        "identifier_key_names": sorted(identifier_names),
        "record_count": len(records),
        "record_schema_keys": schema_keys,
        "rows_with_explicit_heldout_variation_ids": rows_with_explicit_ids,
    }


def audit_file(input_path: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    return audit_bytes(input_path.read_bytes(), protocol)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/armap_sciworld_train_overlap_v1.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_file(args.input, args.protocol)
    if set(result) != ALLOWED_OUTPUT_KEYS:
        raise AssertionError("audit result schema drift")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "record_count": result["record_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
