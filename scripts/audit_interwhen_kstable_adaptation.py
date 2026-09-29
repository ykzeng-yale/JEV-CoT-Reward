#!/usr/bin/env python3
"""Independent complete-record audit and paired descriptive analysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_interwhen_kstable_adaptation import KStableDetector, check_game24, extract_final_boxed, sha256_file


CONFIG = ROOT / "configs/interwhen_kstable_game24_v1.json"
TASKS = ROOT / "data/interwhen_kstable_game24_v1/tasks.json"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [ctr - half, ctr + half]


def bootstrap(values, seed=68121, reps=10000):
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(reps):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return [draws[int(.025 * reps)], draws[min(reps - 1, int(.975 * reps))]]


def audit(run_dir: Path):
    config = json.loads(CONFIG.read_text())
    tasks_meta = json.loads(TASKS.read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = read_jsonl(run_dir / "episodes.jsonl")
    call_rows = read_jsonl(run_dir / "calls.jsonl")
    if manifest.get("status") != "complete" or summary.get("status") != "complete":
        raise ValueError("Run is not complete; preserve partial records and do not infer outcomes")
    if sha256_file(TASKS) != config["dataset"]["tasks_sha256"]:
        raise ValueError("Frozen task hash mismatch")
    if manifest.get("task_file_sha256") != config["dataset"]["tasks_sha256"]:
        raise ValueError("Run used a different task file")
    taskset_audit = manifest.get("taskset_audit", {})
    if (taskset_audit.get("status") != "passed" or taskset_audit.get("task_count") != len(tasks_meta["tasks"])
            or taskset_audit.get("exactly_solvable_count") != len(tasks_meta["tasks"])
            or taskset_audit.get("tasks_sha256") != config["dataset"]["tasks_sha256"]):
        raise ValueError("Independent pre-run task-set audit missing or inconsistent")
    if manifest.get("model_revision") != config["model"]["revision"]:
        raise ValueError("Model revision mismatch")
    identity = manifest.get("model_identity", {})
    if identity.get("snapshot_revision") != config["model"]["revision"] or not identity.get("weight_revision_sha256"):
        raise ValueError("Pinned model snapshot or weight digest is missing/mismatched")
    if manifest.get("cuda") != "12.8":
        raise ValueError(f"Unexpected CUDA runtime: {manifest.get('cuda')}")
    expected = {t["row_idx"]: t for t in tasks_meta["tasks"]}
    arms = set(config["policies"])
    by_key = {}
    for row in rows:
        idx, policy = row["dataset_row_idx"], row["policy"]
        key = (idx, policy)
        if key in by_key:
            raise ValueError(f"Duplicate episode {key}")
        if idx not in expected or policy not in arms or row["numbers"] != expected[idx]["numbers"]:
            raise ValueError(f"Unexpected task or arm in {key}")
        if row["seed"] != 42 + idx:
            raise ValueError(f"Seed mismatch for {key}")
        expression = row.get("final_expression")
        reparsed = extract_final_boxed(row["trace_and_answer"])
        if reparsed != expression:
            raise ValueError(f"Final answer extraction does not reproduce for {key}")
        checked = check_game24(expression, row["numbers"]) if expression else None
        if row["exact_outcome"] != bool(checked and checked.valid):
            raise ValueError(f"Terminal outcome does not match exact evaluator for {key}")
        if row.get("early_stop_triggered"):
            if policy != "interwhen_kstable_k2" or row.get("initial_finish_reason") != "checkpoint":
                raise ValueError(f"Invalid early-stop record for {key}")
            detector = KStableDetector(row["numbers"], k=2)
            if not detector.stable(row["monitor_trace_prefix"]):
                raise ValueError(f"Recorded monitor trigger not reproduced for {key}")
            if detector.trigger_equation != row.get("trigger_equation") or detector.trigger_line_count != row.get("trigger_line_count"):
                raise ValueError(f"Monitor trigger evidence mismatch for {key}")
            if row["injected_control_tokens"] != len(row.get("injected_close_token_ids", [])) or row["injected_control_tokens"] <= 0:
                raise ValueError(f"Injected control token accounting mismatch for {key}")
        elif row["injected_control_tokens"] != 0:
            raise ValueError(f"Unexpected injected control token in {key}")
        expected_calls = 2 if row["early_stop_triggered"] else 1
        if row["calls"] != len(row["call_records"]) or row["calls"] != expected_calls:
            raise ValueError(f"Call count mismatch for {key}")
        if sum(call["generated_tokens"] for call in row["call_records"]) != row["generated_tokens"]:
            raise ValueError(f"Generated-token ledger mismatch for {key}")
        if sum(call["prompt_tokens"] for call in row["call_records"]) != row["prompt_tokens_processed"]:
            raise ValueError(f"Prompt-token ledger mismatch for {key}")
        if not math.isclose(sum(call["elapsed_seconds"] for call in row["call_records"]),
                            row["model_service_seconds"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Service-time ledger mismatch for {key}")
        for field in ("generated_tokens", "prompt_tokens_processed", "model_service_seconds"):
            if row[field] < 0 or not math.isfinite(row[field]):
                raise ValueError(f"Invalid {field} for {key}")
        by_key[key] = row
    expected_keys = {(idx, p) for idx in expected for p in arms}
    if set(by_key) != expected_keys:
        raise ValueError(f"Incomplete episode matrix: got {len(by_key)}, expected {len(expected_keys)}")
    if summary.get("planned_episodes") != len(expected_keys) or summary.get("completed_episodes") != len(rows):
        raise ValueError("Summary counts do not match the frozen episode matrix")
    if manifest.get("completed_episodes") != len(rows):
        raise ValueError("Manifest completion count mismatch")
    expected_call_rows = []
    for row in rows:
        for call in row["call_records"]:
            expected_call_rows.append((row["dataset_row_idx"], row["policy"], call["generated_tokens"],
                                       call["prompt_tokens"], call["elapsed_seconds"], call["finish_reason"]))
    observed_call_rows = [(x["dataset_row_idx"], x["policy"], x["generated_tokens"], x["prompt_tokens"],
                           x["elapsed_seconds"], x["finish_reason"]) for x in call_rows]
    if sorted(expected_call_rows) != sorted(observed_call_rows):
        raise ValueError("Call ledger does not match episode-level call records")
    diffs, token_diffs, time_diffs = [], [], []
    correct = {p: 0 for p in arms}
    timeouts = {p: 0 for p in arms}
    triggers = 0
    for idx in expected:
        cont = by_key[(idx, "continue")]
        mon = by_key[(idx, "interwhen_kstable_k2")]
        correct["continue"] += int(cont["exact_outcome"])
        correct["interwhen_kstable_k2"] += int(mon["exact_outcome"])
        for policy in arms:
            timeouts[policy] += sum(call["finish_reason"] == "timeout"
                                    for call in by_key[(idx, policy)]["call_records"])
        diffs.append(int(mon["exact_outcome"]) - int(cont["exact_outcome"]))
        token_diffs.append(mon["generated_tokens"] - cont["generated_tokens"])
        time_diffs.append(mon["model_service_seconds"] - cont["model_service_seconds"])
        triggers += int(mon["early_stop_triggered"])
    n = len(expected)
    report = {
        "status": "audit_passed", "n_tasks": n, "n_episodes": len(rows), "n_calls": sum(x["calls"] for x in rows),
        "paired_discordant_tasks": sum(d != 0 for d in diffs),
        "policies": {p: {"correct": correct[p], "n": n, "accuracy": correct[p] / n,
                         "wilson_95": wilson(correct[p], n),
                         "generation_timeouts": timeouts[p],
                         "generated_tokens_mean": statistics.mean(by_key[(i, p)]["generated_tokens"] for i in expected),
                         "prompt_tokens_processed_total": sum(by_key[(i, p)]["prompt_tokens_processed"] for i in expected),
                         "model_service_seconds_total": sum(by_key[(i, p)]["model_service_seconds"] for i in expected)} for p in arms},
        "interwhen_kstable_k2": {"natural_triggers": triggers, "trigger_rate": triggers / n,
                                  "mean_generated_token_difference_vs_continue": statistics.mean(token_diffs),
                                  "bootstrap_95_token_difference": bootstrap(token_diffs),
                                  "mean_service_second_difference_vs_continue": statistics.mean(time_diffs),
                                  "paired_accuracy_difference": statistics.mean(diffs),
                                  "paired_accuracy_bootstrap_95": bootstrap(diffs),
                                  "paired_discordance_counts": {"monitor_wins": sum(d == 1 for d in diffs),
                                                                 "monitor_losses": sum(d == -1 for d in diffs),
                                                                 "ties": sum(d == 0 for d in diffs)}},
        "interpretation_limit": "Public training split and one seed; describes this baseline adaptation only, not InterWhen paper replication, Jev value, OOD transfer, or confirmatory efficacy.",
    }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = audit(a.run_dir)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
