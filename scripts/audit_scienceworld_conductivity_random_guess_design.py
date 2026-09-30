#!/usr/bin/env python3
"""Audit the frozen random-guess allocation using development inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re


def audit(protocol: dict, census: dict) -> dict:
    seed = protocol["random_seed"]
    if "random_measurement" not in protocol["policies"]:
        raise ValueError("protocol has no frozen random_measurement baseline")
    dev = [row for row in census["records"] if row.get("split") == "dev"]
    expected = list(range(protocol["splits"]["dev"]["start_inclusive"],
                          protocol["splits"]["dev"]["end_exclusive"]))
    if [row.get("variation_id") for row in dev] != expected:
        raise ValueError("development census coverage/order mismatch")
    grouped: dict[str, list[int]] = defaultdict(list)
    full_hashes = []
    for row in dev:
        h = row.get("full_sha256")
        group = row.get("target_group")
        if not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h):
            raise ValueError("invalid full visible-input hash")
        if not isinstance(group, str) or not re.fullmatch(r"unknown substance [A-Z]", group):
            raise ValueError("invalid public task-group label")
        bit = int(hashlib.sha256(f"{h}:{seed}".encode()).hexdigest(), 16) % 2
        grouped[group].append(bit)
        full_hashes.append(h)
    group_summary = {
        group: {"rows": len(bits), "guess_ones": sum(bits), "guess_zeros": len(bits)-sum(bits),
                "distinct_guess_bits": len(set(bits))}
        for group, bits in sorted(grouped.items())
    }
    total_ones = sum(sum(bits) for bits in grouped.values())
    return {
        "audit": "frozen_random_guess_assignment_only",
        "random_seed": seed,
        "development_rows": len(dev),
        "distinct_full_visible_inputs": len(set(full_hashes)),
        "duplicate_full_input_rows": len(full_hashes)-len(set(full_hashes)),
        "guess_ones": total_ones,
        "guess_zeros": len(dev)-total_ones,
        "groups": group_summary,
        "checks": {"coverage_exact": True,
                   "no_group_collapsed_to_one_bit": all(x["distinct_guess_bits"] == 2 for x in group_summary.values()),
                   "uses_variation_id": False,
                   "uses_endpoint_or_outcome_fields": False,
                   "reads_census_label_flags_or_calibration": False},
        "interpretation": "Deterministic seeded hash allocation over the full visible-input hash; balance is descriptive for this fixed panel, not proof of independent population sampling or a Jev comparison.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    summary=audit(json.loads(args.protocol.read_bytes()), json.loads(args.census.read_bytes()))
    args.output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,sort_keys=True))

if __name__=="__main__":
    main()
