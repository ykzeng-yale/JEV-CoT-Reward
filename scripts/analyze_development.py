#!/usr/bin/env python3
"""Offline integrity audit and descriptive fixed-control development comparison.

No model is loaded. The cached tokenizer is read locally to reconstruct exact
call prefixes, including injected finalization tokens. Audit-time tokenizer
hashes do not establish its launch-time identity. Partial output is explicitly
provisional and never includes the completed-sample bootstrap comparison.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_control.tasks import make_task, verify

CONDITIONS = ("baseline", "sham")
PLAN_FIELDS = ("protocol", "problems", "repeats", "task_seed", "sampling_seed",
               "budget", "checkpoint_target", "checkpoint_cap", "final_reserve",
               "temperature", "top_p", "quantization")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prefix_hash(ids):
    return digest(json.dumps(ids, separators=(",", ":")).encode())


def fail(message):
    raise ValueError(message)


def require(condition, message):
    if not condition:
        fail(message)


def read_json(path):
    try:
        return json.loads(path.read_text(), parse_constant=lambda _: fail("Nonfinite JSON number"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: missing or malformed JSON") from exc


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda _: fail("Nonfinite JSON number"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no}: malformed JSON") from exc
        require(isinstance(row, dict), f"{path.name}:{line_no}: expected object")
        rows.append(row)
    return rows


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: expected integer >= {minimum}")
    return value


def numeric(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{name}: expected finite nonnegative number")
    return value


def token_ids(value, name):
    require(isinstance(value, list) and all(type(t) is int and t >= 0 for t in value),
            f"{name}: expected nonnegative token IDs")
    return value


def literal(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)
            require(isinstance(value, str), f"Frozen {name} must be a string literal")
            return value
    fail(f"Frozen source lacks {name}")


def public_source_omissions(run, manifest):
    """Validate an explicit public release without relaxing private-run audits.

    Original launch hashes remain in manifest.json. Only the declared unused
    Jev network adapter may be absent; no executed local helper is substitutable.
    Source/input bytes and the separate release inventory are checked before an
    omission is honored. Hash inventories provide consistency, not signatures.
    """
    release_path = run / "release_manifest.json"
    if not release_path.exists():
        return {}, {"mode": "complete_launch_inventory", "omitted_original_sources": []}
    release = read_json(release_path)
    require(release.get("format_version") == "public-local-development-v1" and
            release.get("protocol") == manifest.get("protocol") == "development-v1-online-boundary",
            "Unsupported public development release format/protocol")
    require(isinstance(manifest.get("original_run_id"), str) and
            release.get("original_run_id") == manifest["original_run_id"],
            "Public release run identity mismatch")
    hashes = release.get("released_files_sha256")
    require(isinstance(hashes, dict), "Missing public release file hashes")
    required_files = {"manifest.json", "tasks.jsonl", "outcomes.jsonl", "summary.json", "source_manifest.json"}
    require(required_files <= hashes.keys(), "Public release lacks required input/source-manifest hashes")
    for name, expected in hashes.items():
        require(isinstance(name, str) and re.fullmatch(r"(?:[a-z_]+\.jsonl?|source/[A-Za-z_]+\.py)", name),
                "Unexpected public release file path")
        require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "Invalid public file SHA-256")
        path = run / name
        require(not path.is_symlink() and path.is_file() and digest(path.read_bytes()) == expected,
                f"Public release file hash mismatch: {name}")
    sources = read_json(run / "source_manifest.json")
    require(sources.get("protocol") == manifest["protocol"], "Public source manifest protocol mismatch")
    original = manifest.get("source_sha256", {})
    omitted = {}
    records = sources.get("omitted_original_sources")
    require(isinstance(records, list), "Public source omission declaration is missing")
    for item in records:
        require(isinstance(item, dict), "Malformed public source omission")
        name = item.get("original_repository_path")
        require(name == "src/jev_control/jev.py", "Public release may omit only the unused Jev adapter")
        require(name not in omitted and name in original and item.get("original_declared_sha256") == original[name],
                "Public source omission conflicts with original launch inventory")
        require(not (run / "source" / Path(name).name).exists(), "Declared omitted public source is unexpectedly present")
        omitted[name] = original[name]
    entries = sources.get("files")
    require(isinstance(entries, list), "Missing public source provenance entries")
    captured = set()
    for item in entries:
        require(isinstance(item, dict), "Malformed public source provenance entry")
        name = item.get("original_repository_path")
        require(isinstance(name, str) and name in original and name not in captured and name not in omitted,
                "Unexpected, duplicate, or omitted public source entry")
        captured.add(name)
        filename = f"source/{Path(name).name}"
        require(item.get("status") == "captured_development_launch_source" and
                item.get("released_path") == filename and item.get("source_run_id") == manifest.get("original_run_id"),
                "Public source must be this run's captured launch file")
        require(item.get("sha256") == item.get("original_declared_sha256") == original[name] == hashes.get(filename),
                "Public source hash mappings do not match original launch inventory")
    require(captured | omitted.keys() == original.keys(), "Undeclared missing source in public release")
    return omitted, {
        "mode": "public_release_with_explicit_unused_adapter_omission",
        "released_file_hashes_verified": True,
        "release_manifest_sha256": digest(release_path.read_bytes()),
        "source_manifest_sha256": digest((run / "source_manifest.json").read_bytes()),
        "omitted_original_sources": [{"original_repository_path": name, "original_declared_sha256": sha,
                                      "content_verified": False, "reason": "Unused network adapter intentionally not released"}
                                     for name, sha in omitted.items()],
        "note": "Original launch hash inventory is preserved separately from content verified in this public release.",
    }


def audit_sources(run, manifest):
    hashes = manifest.get("source_sha256")
    require(isinstance(hashes, dict) and hashes, "Missing frozen source hashes")
    required = {"scripts/run_development.py", "scripts/run_screen.py",
                "src/jev_control/tasks.py", "src/jev_control/mlx_backend.py"}
    require(required <= hashes.keys(), "Missing required frozen source hashes")
    omitted, source_validation = public_source_omissions(run, manifest)
    names = set()
    for name, expected in hashes.items():
        require(isinstance(name, str) and re.fullmatch(r"(?:scripts|src/jev_control)/[A-Za-z_]+\.py", name),
                "Unexpected source inventory key")
        require(Path(name).name not in names, "Colliding source snapshot filenames")
        names.add(Path(name).name)
        require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "Invalid source SHA-256")
        if name in omitted:
            continue
        path = run / "source" / Path(name).name
        require(path.is_file() and digest(path.read_bytes()) == expected, f"Frozen source hash mismatch: {Path(name).name}")
    current_tasks_hash = digest((ROOT / "src/jev_control/tasks.py").read_bytes())
    require(current_tasks_hash == hashes["src/jev_control/tasks.py"],
            "Task generator/verifier source differs from frozen run; explicit review required")
    final = literal((run / "source/run_screen.py").read_text(), "FINAL")
    suffix = literal((run / "source/run_development.py").read_text(), "PROMPT_SUFFIX")
    return hashes, final, suffix, source_validation


class LocalTokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer
        path = Path(directory).expanduser() / "tokenizer.json"
        require(path.is_file(), "A cached local tokenizer.json is required")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.audit_sha256 = digest(path.read_bytes())

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def marker(text):
    return bool(re.search(r"\d\s*[+*/−-]\s*\d|(?:distance|cost|dist|total)\s*[:=(]|\d\s*=\s*\d", text, re.I))


def audit_episode(row, item, plan, tokenizer, final):
    """Reconstruct the actual protocol without executing any saved source."""
    context = f"{row['problem_id']}:{row['condition']}:{row['repeat']}"
    calls = row.get("calls")
    require(isinstance(calls, list) and calls, f"{context}: missing generation calls")
    condition = row["condition"]
    prompt = item["prompt_ids"]
    tokens, call_index, spent = [], 0, 0
    reasoning_budget = plan["budget"] - plan["final_reserve"]

    def take(cap, seed, checkpoint_allowed=False):
        nonlocal call_index, spent
        require(call_index < len(calls), f"{context}: missing required successive call")
        call = calls[call_index]
        require(isinstance(call, dict), f"{context}: malformed call")
        ids = token_ids(call.get("token_ids"), f"{context}: call token_ids")
        n = integer(call.get("generated_tokens"), "call generated_tokens")
        require(n == len(ids), f"{context}: generated_tokens differs from raw token IDs")
        require(n <= cap, f"{context}: call token cap exceeded")
        expected_prefix = prompt + tokens
        require(call.get("prefix_sha256") == prefix_hash(expected_prefix), f"{context}: call prefix hash mismatch")
        require(integer(call.get("prompt_tokens"), "call prompt_tokens") == len(expected_prefix),
                f"{context}: call prompt count mismatch")
        require(type(call.get("seed")) is int and call["seed"] == seed, f"{context}: call seed mismatch")
        require(call.get("text") == tokenizer.decode(ids), f"{context}: call text/token decode mismatch")
        finish = call.get("finish_reason")
        require(finish in {"length", "stop", "timeout"} | ({"checkpoint"} if checkpoint_allowed else set()),
                f"{context}: invalid finish reason for call")
        require(finish != "length" or n == cap, f"{context}: length finish before requested cap")
        numeric(call.get("elapsed_seconds"), "call elapsed_seconds")
        tokens.extend(ids)
        spent += n
        call_index += 1
        return call

    initial = take(reasoning_budget if condition == "baseline" else plan["checkpoint_cap"],
                   row["seed"], condition == "sham")
    checkpoint = row.get("checkpoint")
    if condition == "baseline":
        require(checkpoint is None, f"{context}: baseline has checkpoint")
        expected_status = "not_requested"
    else:
        initial_ids = initial["token_ids"]
        first_boundary = None
        last = len(initial_ids) - (initial["finish_reason"] == "stop")
        for end in range(plan["checkpoint_target"], last + 1):
            text = tokenizer.decode(initial_ids[:end])
            if text.endswith("\n") and "FINAL:" not in text:
                first_boundary = end
                break
        if initial["finish_reason"] == "checkpoint":
            require(first_boundary == len(initial_ids), f"{context}: checkpoint is not first eligible online boundary")
            require(isinstance(checkpoint, dict), f"{context}: missing checkpoint record")
            saved_ids = token_ids(checkpoint.get("token_ids"), "checkpoint token_ids")
            saved_position = integer(checkpoint.get("position"), "checkpoint position")
            require(saved_ids == initial_ids and saved_position == len(initial_ids),
                    f"{context}: checkpoint token/position mismatch")
            require(checkpoint.get("prefix_sha256") == prefix_hash(prompt + initial_ids), f"{context}: checkpoint prefix hash mismatch")
            require(checkpoint.get("text") == initial["text"], f"{context}: checkpoint text mismatch")
            require(type(checkpoint.get("calculation_marker")) is bool and checkpoint["calculation_marker"] == marker(initial["text"]),
                    f"{context}: calculation marker mismatch")
            expected_status = "eligible"
        else:
            require(first_boundary is None, f"{context}: missed eligible online boundary")
            require(checkpoint is None, f"{context}: unexpected checkpoint record")
            expected_status = ("completed_early" if initial["finish_reason"] == "stop" else
                               "timeout" if initial["finish_reason"] == "timeout" else
                               "answer_phase_before_checkpoint" if "FINAL:" in initial["text"] else "no_boundary_before_cap")
        if initial["finish_reason"] not in {"stop", "timeout"} and reasoning_budget - spent:
            take(reasoning_budget - spent, row["seed"] + 1)
    require(row.get("checkpoint_status") == expected_status, f"{context}: checkpoint status mismatch")
    expected_overhead = []
    if calls[call_index - 1]["finish_reason"] not in {"stop", "timeout"} and plan["budget"] - spent:
        if "FINAL:" not in tokenizer.decode(tokens):
            injection = tokenizer.encode(final)
            tokens.extend(injection)
            expected_overhead.append({"kind": "finalization_instruction", "tokens": len(injection)})
        take(plan["budget"] - spent, row["seed"] + 2)
    require(call_index == len(calls), f"{context}: unexpected extra generation calls")
    require(row.get("overhead") == expected_overhead, f"{context}: FINAL instruction overhead mismatch")
    require(spent <= plan["budget"], f"{context}: episode token cap exceeded")
    require(row.get("text") == tokenizer.decode(tokens), f"{context}: episode text/token decode mismatch")
    resources = {"generated_tokens": sum(c["generated_tokens"] for c in calls),
                 "prompt_tokens_processed": sum(c["prompt_tokens"] for c in calls),
                 "elapsed_seconds": sum(c["elapsed_seconds"] for c in calls),
                 "inserted_instruction_tokens": sum(x["tokens"] for x in expected_overhead),
                 "generation_calls": len(calls)}
    for key in ("generated_tokens", "prompt_tokens_processed", "elapsed_seconds"):
        recorded = (numeric if key == "elapsed_seconds" else integer)(row.get(key), f"{context}: {key}")
        require(math.isclose(recorded, resources[key], rel_tol=1e-9, abs_tol=1e-9), f"{context}: {key} differs from call sum")
    recorded_outcome = row.get("outcome")
    require(isinstance(recorded_outcome, dict) and type(recorded_outcome.get("success")) is bool,
            f"{context}: malformed recorded outcome")
    require(set(recorded_outcome) <= {"success", "reason", "path_cost", "value"} and
            recorded_outcome.get("reason") in {"missing_final", "invalid_path_format", "checked_path", "checked_expression", "invalid_answer"},
            f"{context}: unexpected recorded outcome fields")
    if "path_cost" in recorded_outcome:
        numeric(recorded_outcome["path_cost"], "recorded path_cost")
    if "value" in recorded_outcome:
        require(isinstance(recorded_outcome["value"], str) and re.fullmatch(r"-?\d+(?:/\d+)?", recorded_outcome["value"]),
                f"{context}: malformed recorded arithmetic value")
    reverified = verify(item["task"], row["text"])
    return {"problem_id": row["problem_id"], "family": row["family"], "repeat": row["repeat"],
            "condition": condition, "seed": row["seed"], "success": reverified["success"],
            "recorded_outcome": row["outcome"], "reverified_outcome": reverified,
            "outcome_disagreement": row["outcome"] != reverified,
            "checkpoint_status": expected_status, "calculation_marker": bool(checkpoint and checkpoint["calculation_marker"]),
            "finish_reasons": [c["finish_reason"] for c in calls], "costs": resources}


def aggregate(rows):
    n = len(rows)
    return {"episodes": n, "independent_problems": len({r["problem_id"] for r in rows}),
            "successes": sum(r["success"] for r in rows),
            "success_rate": sum(r["success"] for r in rows) / n if n else None,
            "costs": {k: sum(r["costs"][k] for r in rows) for k in
                      ("generated_tokens", "prompt_tokens_processed", "elapsed_seconds", "inserted_instruction_tokens", "generation_calls")},
            "checkpoint_status": dict(Counter(r["checkpoint_status"] for r in rows)),
            "calculation_marker_checkpoints": sum(r["calculation_marker"] for r in rows),
            "timeout_episodes": sum("timeout" in r["finish_reasons"] for r in rows),
            "verification_reasons": dict(Counter(r["reverified_outcome"]["reason"] for r in rows))}


def quantile(values, q):
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    lower = int(at)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (at - lower)


def bootstrap(problems, draws, seed):
    rng = random.Random(seed)
    groups = [[p for p in problems if p["family"] == f] for f in sorted({p["family"] for p in problems})]
    samples = []
    for _ in range(draws):
        selected = [rng.choice(g) for g in groups for _ in g]
        samples.append(sum(p["sham_minus_baseline"] for p in selected) / len(selected))
    return {"estimand": "Mean per-original-problem sham success rate minus fixed baseline success rate",
            "independent_problems": len(problems),
            "difference": sum(p["sham_minus_baseline"] for p in problems) / len(problems),
            "descriptive_stratified_percentile_95": [quantile(samples, .025), quantile(samples, .975)],
            "strata": dict(Counter(p["family"] for p in problems)), "draws": draws, "seed": seed,
            "resampling_unit": "Original problem, retaining both conditions and all repeats; preserve family counts",
            "interpretation": "Descriptive small-development-sample interval; no equivalence claim, Jev effect, or confirmatory coverage guarantee."}


def analyze(run, *, partial=False, plan=None, tokenizer=None, tokenizer_path=None, draws=2000, seed=271828):
    run = Path(run)
    integer(draws, "bootstrap draws", 1)
    integer(seed, "bootstrap seed")
    plan = plan or read_json(ROOT / "configs/development_v1.json")
    manifest = read_json(run / "manifest.json")
    require(isinstance(manifest, dict), "Manifest must be an object")
    for key in PLAN_FIELDS:
        require(key in plan and manifest.get(key) == plan[key], f"Manifest differs from expected plan: {key}")
    for key in ("problems", "repeats", "budget", "checkpoint_target", "checkpoint_cap"):
        integer(plan[key], f"plan {key}", 1)
    for key in ("task_seed", "sampling_seed"):
        integer(plan[key], f"plan {key}")
    integer(plan["final_reserve"], "plan final_reserve")
    require(0 < plan["checkpoint_target"] <= plan["checkpoint_cap"] < plan["budget"] - plan["final_reserve"], "Invalid plan envelope")
    hashes, final, suffix, source_validation = audit_sources(run, manifest)
    tokenizer = tokenizer or LocalTokenizer(tokenizer_path or manifest.get("model", ""))
    require(tokenizer.encode(final), "FINAL instruction encoded to empty tokens")
    expected_tasks = []
    for index in range(plan["problems"]):
        task = make_task(index, plan["task_seed"])
        task["prompt"] += suffix
        expected_tasks.append(task)
    expected_ids = {t["id"]: i for i, t in enumerate(expected_tasks)}
    items = {}
    for item in read_jsonl(run / "tasks.jsonl"):
        require(isinstance(item.get("task"), dict), "Malformed task record")
        pid = item["task"].get("id")
        require(isinstance(pid, str) and pid in expected_ids, "Unexpected task identifier")
        require(pid not in items, f"Duplicate task: {pid}")
        require(item["task"] == expected_tasks[expected_ids[pid]], f"Task differs from frozen generation plan: {pid}")
        prompt = token_ids(item.get("prompt_ids"), "task prompt_ids")
        require(prompt, "Empty task prompt IDs")
        require(item["task"]["prompt"] in tokenizer.decode(prompt), f"Saved prompt IDs do not contain task prompt: {pid}")
        items[pid] = item
    raw_rows = read_jsonl(run / "outcomes.jsonl")
    lookup, audited = {}, []
    for row in raw_rows:
        pid, condition = row.get("problem_id"), row.get("condition")
        require(isinstance(pid, str) and pid in items, "Outcome references missing or unknown task")
        require(condition in CONDITIONS, "Unexpected condition")
        repeat = integer(row.get("repeat"), "outcome repeat")
        require(repeat < plan["repeats"], "Repeat outside expected plan")
        key = (pid, repeat, condition)
        require(key not in lookup, f"Duplicate episode: {pid}:{repeat}:{condition}")
        require(row.get("family") == items[pid]["task"]["family"], "Outcome/task family mismatch")
        expected_seed = plan["sampling_seed"] + expected_ids[pid] * 1000 + repeat * 10
        require(type(row.get("seed")) is int and row["seed"] == expected_seed, "Episode seed differs from expected plan")
        audited.append(audit_episode(row, items[pid], plan, tokenizer, final))
        lookup[key] = row
    expected_keys = {(pid, r, c) for pid in expected_ids for r in range(plan["repeats"]) for c in CONDITIONS}
    missing = sorted(expected_keys - lookup.keys())
    summary_exists = (run / "summary.json").is_file()
    complete = not missing and len(items) == plan["problems"] and summary_exists
    if summary_exists:
        summary = read_json(run / "summary.json")
        require(isinstance(summary, dict) and summary.get("episodes") == len(raw_rows), "Summary episode count mismatch")
    require(complete or partial, f"Incomplete expected plan: {len(raw_rows)}/{len(expected_keys)} episodes; completion summary={summary_exists}. Use --partial only for provisional diagnostics.")
    pairs = []
    for pid in expected_ids:
        for repeat in range(plan["repeats"]):
            base, sham = lookup.get((pid, repeat, "baseline")), lookup.get((pid, repeat, "sham"))
            if base is None or sham is None:
                continue
            left, right = base["calls"][0]["token_ids"], sham["calls"][0]["token_ids"]
            first_difference = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
            full_match = first_difference is None and len(left) >= len(right)
            pairs.append({"problem_id": pid, "repeat": repeat, "baseline_initial_tokens": len(left),
                          "sham_initial_tokens": len(right), "compared_tokens": min(len(left), len(right)),
                          "sham_initial_is_baseline_prefix": full_match, "first_disagreement_position_zero_based": first_difference,
                          "baseline_shorter_than_sham": len(left) < len(right),
                          "sham_checkpoint_status": sham["checkpoint_status"]})
    problems = []
    for task in expected_tasks:
        selected = [r for r in audited if r["problem_id"] == task["id"]]
        conditions = {c: aggregate([r for r in selected if r["condition"] == c]) for c in CONDITIONS}
        matched = all(conditions[c]["episodes"] == plan["repeats"] for c in CONDITIONS)
        problems.append({"problem_id": task["id"], "family": task["family"], "conditions": conditions,
                         "complete_paired_repeats": matched,
                         "sham_minus_baseline": conditions["sham"]["success_rate"] - conditions["baseline"]["success_rate"] if matched else None})
    conditions = {}
    for condition in CONDITIONS:
        selected = [r for r in audited if r["condition"] == condition]
        conditions[condition] = {**aggregate(selected),
            "families": {f: aggregate([r for r in selected if r["family"] == f]) for f in sorted({t["family"] for t in expected_tasks})},
            "failed_outcome_costs": aggregate([r for r in selected if not r["success"]])["costs"]}
    by_status = {s: aggregate([r for r in audited if r["condition"] == "sham" and r["checkpoint_status"] == s])
                 for s in sorted({r["checkpoint_status"] for r in audited if r["condition"] == "sham"})}
    return {"schema_version": 1, "status": "provisional_partial_diagnostics" if partial else "complete_development_descriptive",
            "scientific_final": False, "expected_plan_complete": complete, "partial_explicitly_requested": partial,
            "purpose": "Development framing and fixed baseline/sham comparison; no Jev-effect or equivalence claim",
            "counts": {"expected_problems": plan["problems"], "recorded_tasks": len(items),
                       "problems_with_outcomes": len({r["problem_id"] for r in audited}), "expected_episodes": len(expected_keys),
                       "observed_episodes": len(audited), "paired_problem_repeat_cells": len(pairs), "missing_episodes": len(missing)},
            "missing_episode_keys": [{"problem_id": p, "repeat": r, "condition": c} for p, r, c in missing],
            "conditions": conditions, "per_original_problem": problems,
            "fixed_control_comparison": bootstrap(problems, draws, seed) if complete and not partial else None,
            "eligibility_and_markers": {"sham_status_breakdown": by_status,
                                        "eligible_with_marker": aggregate([r for r in audited if r["condition"] == "sham" and r["checkpoint_status"] == "eligible" and r["calculation_marker"]]),
                                        "eligible_without_marker": aggregate([r for r in audited if r["condition"] == "sham" and r["checkpoint_status"] == "eligible" and not r["calculation_marker"]])},
            "prepause_matching": {"paired_cells": len(pairs), "full_sham_prefix_agreements": sum(p["sham_initial_is_baseline_prefix"] for p in pairs),
                                  "disagreements": [p for p in pairs if not p["sham_initial_is_baseline_prefix"]], "pairs": pairs,
                                  "interpretation": "Observed same-problem/seed prefix diagnostic only; fresh resume seeds and stochastic execution do not imply global bitwise equivalence."},
            "outcome_reverification": {"episodes": len(audited), "disagreements": [r for r in audited if r["outcome_disagreement"]],
                                       "analysis_uses": "Recomputed strict task outcomes; original records are never rewritten"},
            "accounting": {**aggregate(audited)["costs"], "includes": "All saved generation calls, including failed outcomes and timeouts; injected instructions count as prompt overhead, not generated tokens",
                           "wall_time_scope": "Sum of recorded generation-call service times, excluding model load and unsaved failures; not end-to-end elapsed time"},
            "integrity": {"call_caps_counts_seeds_prefixes_and_decoded_text_verified": True,
                          "final_instruction_and_successive_prefixes_verified": True, "completion_summary_present": summary_exists},
            "provenance": {"run_source_sha256": hashes, "analysis_source_sha256": digest(Path(__file__).read_bytes()),
                           "source_validation": source_validation,
                           "verifier_source_sha256": digest((ROOT / "src/jev_control/tasks.py").read_bytes()),
                           "input_sha256": {name: digest((run / name).read_bytes()) for name in ("manifest.json", "tasks.jsonl", "outcomes.jsonl", "summary.json") if (run / name).is_file()},
                           "audit_time_tokenizer_json_sha256": getattr(tokenizer, "audit_sha256", None),
                           "tokenizer_identity_note": "Audit reference only; the launch manifest did not independently hash tokenizer bytes.",
                           "expected_plan": {k: plan[k] for k in PLAN_FIELDS}},
            "limitations": ["Twelve original problems by default; repeated generations are not independent tasks.",
                            "The family-stratified bootstrap describes this fixed-control development comparison; it cannot establish equivalence or nominal small-sample coverage.",
                            "Calculation markers are surface syntax, not verified useful reasoning.",
                            "Saved-call costs include failed outcomes; work that crashed before a row was saved cannot be recovered from these files.",
                            "No policy learning, Jev calls, model inference or network calls are performed by this analysis."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--plan", type=Path, help="Explicit alternative expected plan; defaults to frozen development_v1")
    parser.add_argument("--tokenizer", type=Path, help="Cached tokenizer directory override; local files only")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "Refusing to overwrite an analysis result")
        result = analyze(args.run, partial=args.partial, plan=read_json(args.plan) if args.plan else None,
                         tokenizer_path=args.tokenizer, draws=args.draws, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "outcome_disagreements": len(result["outcome_reverification"]["disagreements"])}))


if __name__ == "__main__":
    main()
