#!/usr/bin/env python3
"""Verify frozen InterWhen/Game24 task indices and solvability without retaining witnesses."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/interwhen_kstable_game24_v1/tasks.json"


def is_solvable(numbers: list[int]) -> bool:
    @lru_cache(None)
    def search(values: tuple[Fraction, ...]) -> bool:
        if len(values) == 1:
            return values[0] == 24
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                a, b = values[i], values[j]
                rest = [values[k] for k in range(len(values)) if k not in (i, j)]
                results = {a + b, a * b, a - b, b - a}
                if b:
                    results.add(a / b)
                if a:
                    results.add(b / a)
                for value in results:
                    if search(tuple(sorted(rest + [value]))):
                        return True
        return False

    return search(tuple(sorted(Fraction(n) for n in numbers)))


def audit(path: Path = TASKS) -> dict:
    payload = json.loads(path.read_text())
    tasks = payload["tasks"]
    count = len(tasks)
    total = payload["total_rows"]
    expected_indices = [int(i * (total - 1) / (count - 1)) for i in range(count)]
    observed_indices = [row["row_idx"] for row in tasks]
    if observed_indices != expected_indices:
        raise ValueError("Task rows do not match the frozen evenly-spaced upstream sample")
    if any(set(row) != {"row_idx", "numbers"} for row in tasks):
        raise ValueError("Frozen task rows contain unexpected fields, including possible label leakage")
    if any(len(row["numbers"]) != 4 or any(type(n) is not int for n in row["numbers"]) for row in tasks):
        raise ValueError("Malformed four-number task")
    solvable = sum(is_solvable(row["numbers"]) for row in tasks)
    if solvable != count:
        raise ValueError(f"Only {solvable}/{count} sampled tasks are exactly solvable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"status": "passed", "task_count": count, "unique_row_count": len(set(observed_indices)),
            "exactly_solvable_count": solvable, "witnesses_retained": False,
            "solutions_or_difficulty_fields_present": False,
            "dataset_revision": payload["dataset_revision"], "tasks_sha256": digest}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, default=TASKS)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.tasks)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
