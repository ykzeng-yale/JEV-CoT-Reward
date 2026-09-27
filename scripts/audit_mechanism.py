#!/usr/bin/env python3
"""Completed-run integrity gate for mechanism-v1, without inference or APIs.

Replay the *recorded call contract*, not model generation. Launch source files
are checked against their recorded hashes. Supported protocol ASTs identify the
semantics being audited independently of subsequently edited working sources.
Only the three hash-approved, standard-library task functions are evaluated.
No saved runner, analyzer, model code or arbitrary import is executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import heapq
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import sys

from analyze_development import LocalTokenizer, digest, integer, numeric, prefix_hash, read_json, read_jsonl, require, token_ids

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("continue", "repair", "branch")
PROTOCOL = "mechanism-v1-online-emitted-checkpoint"
# Function bodies as inspected for this protocol; comments and top docstrings
# do not change these identities. This deliberately does not gate on current
# file hashes or execute an older saved analyzer to interpret the evidence.
CONTRACTS = {
    "scripts/run_screen.py": {
        "rollout": "d98e9ce32040c599fde9a6483770d92c0b2dd7dbddd73e70421837c63e23f173",
        "retained_statistics": "1ef53c77b8be5866168e5519307deebc9884869bd2cb22ebabf1f0f48e663ed3"},
    "scripts/run_development.py": {
        "boundary_now": "66834df9671133dd0e4ecee6a1d50ecda51d95c6850a00c0dbcd75854a713fdd",
        "calculation_marker": "857dd3690ee1f3ee1aef2c39b1653471439fd518257bf4a6da01c0c3e08ee9a6"},
    "scripts/run_mechanism.py": {
        "schedule_for": "a65af79152287459d5bbb2a3206b6c324432b348ffb7b49ca1533c705c0f57de",
        "planned_tasks": "7c120e46208afb934cb15ebb767f3581ea9f148f1108ff193052befb9a3a7f83",
        "checkpoint_from_initial": "f0bae5764dda833415eda22f70a8b7208e9504037ed2a72deb8716b45a56f066"},
    "src/jev_control/mlx_backend.py": {
        "MLXBackend.choose_candidate": "9c32d51d65add8dd97927711694448714afca82c3ea01525e4ea00abd0f65d74"},
    "src/jev_control/tasks.py": {
        "make_task": "3a6a69eb1d76dc978732d0701941e3505b9bc4b16dd615204af2a8ae5e2f9343",
        "shortest_distance": "dbfa9754c3a2ac427f638f426c89b727aa252688b7b3092470635e7bd6fa7dc3",
        "verify": "a2cac559156f23d08cd43f7558307d7fcd776bb69e44487e239e48d1d39659f8"},
}


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name and not name.startswith("/"), "Invalid relative inventory path")
    path = PurePosixPath(name)
    require(all(x not in {".", "..", ""} for x in path.parts) and str(path) == name,
            "Unsafe relative inventory path")
    return path


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256 record")
    return value


def literal(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Frozen source lacks literal {name}")


def function(tree, name):
    body = tree.body
    for part in name.split("."):
        matches = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        require(len(matches) == 1, "Frozen source lacks a unique supported protocol function")
        node = matches[0]
        body = node.body
    if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    return node


def frozen_sources(run, manifest):
    hashes = manifest.get("source_sha256")
    require(isinstance(hashes, dict), "Missing launch source inventory")
    require(len(hashes) <= 128, "Oversized launch source inventory")
    required = set(CONTRACTS) | {"src/jev_control/features.py", "configs/mechanism_v1.json"}
    require(required <= hashes.keys(), "Incomplete launch source inventory")
    trees = {}
    for name, expected in hashes.items():
        safe_relative(name)
        require(name.startswith(("scripts/", "src/jev_control/", "configs/")) or name in {"pyproject.toml", "requirements.lock.txt"},
                "Unexpected launch source inventory location")
        path = run / "source" / name
        require(path.is_file() and digest(path.read_bytes()) == sha(expected), "Frozen source file/hash mismatch")
        if name in required and name.endswith(".py"):
            trees[name] = ast.parse(path.read_text())
    task_nodes = []
    for name, functions in CONTRACTS.items():
        for func, expected in functions.items():
            node = function(trees[name], func)
            # Python 3.12 adds empty type_params to otherwise identical ASTs.
            canonical = ast.dump(node, include_attributes=False).replace(", type_params=[]", "")
            require(digest(canonical.encode()) == expected,
                    f"Unsupported frozen protocol contract: {Path(name).name}:{func}")
            if name == "src/jev_control/tasks.py":
                task_nodes.append(node)
    # Each node was individually matched to inspected source; exclude all
    # imports and arbitrary module-level expressions from the saved file.
    namespace = {"ast": ast, "Counter": Counter, "Fraction": Fraction,
                 "heapq": heapq, "random": random, "re": re}
    module = ast.fix_missing_locations(ast.Module(body=task_nodes, type_ignores=[]))
    exec(compile(module, "<approved frozen task functions>", "exec"), namespace)
    constants = {
        "FINAL": literal(trees["scripts/run_screen.py"], "FINAL"),
        "REPAIR": literal(trees["scripts/run_screen.py"], "REPAIR"),
        "PROMPT_SUFFIX": literal(trees["scripts/run_development.py"], "PROMPT_SUFFIX"),
    }
    require(all(isinstance(v, str) for v in constants.values()), "Invalid frozen instruction literals")
    require(tuple(literal(trees["scripts/run_screen.py"], "ACTIONS")) == ACTIONS, "Frozen action order mismatch")
    require(literal(trees["scripts/run_mechanism.py"], "PROTOCOL") == PROTOCOL, "Frozen protocol mismatch")
    rubric = {"schema": literal(trees["src/jev_control/features.py"], "SCHEMA_VERSION"),
              "questions": literal(trees["src/jev_control/features.py"], "QUESTIONS")}
    config = read_json(run / "source/configs/mechanism_v1.json")
    return hashes, constants, namespace, rubric, config


def model_tokenizer(manifest, directory, tokenizer):
    identity = manifest.get("model_identity")
    require(isinstance(identity, dict), "Missing launch model identity")
    files, weights = identity.get("file_sha256"), identity.get("weight_files_sha256")
    require(isinstance(files, dict) and isinstance(weights, dict) and weights, "Missing launch asset hashes")
    require(len(files) <= 1024, "Oversized launch asset inventory")
    for name, value in files.items():
        safe_relative(name)
        sha(value)
    require(all(name in files and files[name] == value for name, value in weights.items()), "Weight/file hash inventories disagree")
    require(set(weights) == {name for name in files if PurePosixPath(name).suffix in {".safetensors", ".bin", ".npz"}},
            "Weight asset classification mismatch")
    require(identity.get("weight_revision_sha256") == digest(json.dumps(weights, sort_keys=True).encode()), "Weight revision digest mismatch")
    require("tokenizer.json" in files and "config.json" in files, "Missing pinned tokenizer/config assets")
    directory = Path(directory or identity.get("path", "")).expanduser()
    require(directory.is_dir(), "A local tokenizer asset directory is required")
    checked_metadata = {}
    for name, expected in files.items():
        if name in weights:
            continue
        path = directory / name
        require(path.is_file(), "Pinned tokenizer/model metadata file is missing")
        require(path.stat().st_size <= 64 * 1024 * 1024, "Unexpected oversized tokenizer metadata asset")
        require(digest(path.read_bytes()) == expected, "Tokenizer/model metadata differs from launch hash")
        checked_metadata[name] = expected
    require(read_json(directory / "config.json") == identity.get("config"), "Launch model configuration copy mismatch")
    quantization = manifest.get("actual_quantization_config")
    require(isinstance(quantization, dict) and quantization.get("bits") == 4
            and quantization.get("group_size") == 64 and quantization.get("mode", "affine") == "affine",
            "Unexpected actual model quantization")
    actual_config = manifest.get("actual_model_config")
    require(isinstance(actual_config, dict) and actual_config.get("quantization") == quantization,
            "Actual model/quantization records disagree")
    require(manifest.get("quantize_bits") in (0, 4), "Unknown quantization loading route")
    if manifest["quantize_bits"] == 0:
        require(identity["config"].get("quantization") == quantization, "Prequantized source/loaded quantization differ")
    tokenizer = tokenizer or LocalTokenizer(directory)
    revision = identity.get("snapshot_revision")
    return tokenizer, {"launch_weight_revision_sha256": identity["weight_revision_sha256"],
                       "launch_weight_file_count": len(weights), "weights_rehashed_during_audit": False,
                       "weight_identity_scope": "Internal consistency of launch-recorded weight digests; no fresh tensor-file hashing or inference replay",
                       "metadata_file_count_verified_against_launch": len(checked_metadata),
                       "tokenizer_json_sha256": files["tokenizer.json"],
                       "snapshot_revision": revision if isinstance(revision, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", revision) else None}


def scalar(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"{name}: expected finite number")
    return value


def equal_number(recorded, actual, label):
    numeric(recorded, label)
    require(math.isclose(recorded, actual, rel_tol=1e-8, abs_tol=1e-8), f"{label} accounting mismatch")


def generation(g, prefix, cap, seed, tokenizer, checkpoint=False):
    require(isinstance(g, dict), "Malformed generation")
    ids = token_ids(g.get("token_ids"), "generation token_ids")
    n = integer(g.get("generated_tokens"), "generated_tokens")
    require(n == len(ids) and n <= cap, "Raw generated token count/cap mismatch")
    require(g.get("prefix_sha256") == prefix_hash(prefix), "Generation prefix hash mismatch")
    require(integer(g.get("prompt_tokens"), "prompt_tokens") == len(prefix), "Generation prompt count mismatch")
    require(type(g.get("seed")) is int and g["seed"] == seed, "Generation seed mismatch")
    require(g.get("text") == tokenizer.decode(ids), "Generation text/token mismatch")
    finish = g.get("finish_reason")
    require(finish in {"stop", "length", "timeout"} | ({"checkpoint"} if checkpoint else set()), "Unexpected generation finish reason")
    require(finish != "length" or n == cap, "Length finish before requested token cap")
    numeric(g.get("elapsed_seconds"), "generation elapsed_seconds")
    numeric(g.get("peak_memory_gb"), "generation peak_memory_gb")
    for array_key, mean_key in (("token_logprobs", "mean_logprob"), ("token_entropies", "mean_entropy")):
        values = g.get(array_key)
        require(isinstance(values, list) and len(values) == n, "Missing/misaligned per-token statistics")
        for value in values:
            scalar(value, array_key)
        if n:
            require(math.isclose(scalar(g.get(mean_key), mean_key), sum(values) / n, rel_tol=1e-7, abs_tol=1e-7), "Generation mean statistic mismatch")
        else:
            require(g.get(mean_key) is None, "Empty generation has a nonempty mean statistic")
    return g


class Ledger:
    def __init__(self, run, tokenizer):
        self.starts = read_jsonl(run / "generation_started.jsonl")
        self.events = read_jsonl(run / "generation_events.jsonl")
        require(len(self.starts) == len(self.events), "Unmatched durable generation starts/events")
        self.cursor = 0
        self.tokenizer = tokenizer
        for sequence, (start, event) in enumerate(zip(self.starts, self.events)):
            require(type(start.get("sequence")) is int and start["sequence"] == sequence
                    and type(event.get("sequence")) is int and event["sequence"] == sequence,
                    "Duplicate, missing or reordered durable call sequence")
            require(event.get("status") == "complete", "Completed run has failed/unknown-work durable call")
            require({k: v for k, v in event.items() if k not in {"status", "generation"}} == start,
                    "Durable start/event identity mismatch")
            numeric(start.get("started_unix"), "call start time")
            if sequence:
                require(start["started_unix"] >= self.starts[sequence - 1]["started_unix"], "Durable start order is not chronological")

    def take(self, context, prefix, cap, seed, checkpoint=False, row_call=None):
        require(self.cursor < len(self.events), "Missing durable generation event")
        event = self.events[self.cursor]
        start = self.starts[self.cursor]
        expected_keys = set(context) | {"sequence", "started_unix", "prefix_ids", "max_tokens", "seed"}
        require(set(start) == expected_keys and all(start.get(k) == v for k, v in context.items()), "Durable call context/schedule mismatch")
        require(token_ids(start.get("prefix_ids"), "durable prefix_ids") == prefix, "Durable successive-call prefix mismatch")
        require(integer(start.get("max_tokens"), "durable cap", 1) == cap, "Durable requested token cap mismatch")
        require(type(start.get("seed")) is int and start["seed"] == seed, "Durable call seed mismatch")
        result = generation(event.get("generation"), prefix, cap, seed, self.tokenizer, checkpoint)
        if row_call is not None:
            require(result == row_call, "Outcome/checkpoint generation differs from durable event")
        self.cursor += 1
        return result


def online_boundary(g, target, tokenizer):
    ids = g["token_ids"]
    last = len(ids) - (g["finish_reason"] == "stop")
    found = None
    for end in range(target, last + 1):
        text = tokenizer.decode(ids[:end])
        if text.endswith("\n") and "FINAL:" not in text:
            found = end
            break
    if g["finish_reason"] == "checkpoint":
        require(found == len(ids), "Checkpoint is not the first eligible online stop")
    else:
        require(found is None, "Initial generation passed an eligible online boundary")


def checkpoint(record, item, initial, tokenizer):
    retained, prompt = initial["token_ids"], record.get("prompt_ids")
    require(record.get("retained_ids") == retained and record.get("discarded_prefix_tail_tokens") == 0,
            "Checkpoint trims, replaces or discards emitted tokens")
    require(record.get("task") == item["task"] and record.get("problem_id") == item["task"]["id"]
            and record.get("index") == item["index"], "Checkpoint/task join mismatch")
    expected_hash = digest(json.dumps({"prompt": prompt, "retained": retained}, sort_keys=True).encode())
    require(record.get("sha256") == expected_hash, "Checkpoint identity hash mismatch")
    text = tokenizer.decode(retained)
    parts = text.rstrip().rsplit("\n\n", 1)
    history, segment = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
    require(record.get("state") == {"task": item["task"]["prompt"], "history": history, "latest_segment": segment},
            "Checkpoint rubric state differs from retained-only segmentation")
    expected_marker = bool(re.search(r"\d\s*[+*/−-]\s*\d|(?:distance|cost|dist|total)\s*[:=(]|\d\s*=\s*\d", text, re.I))
    require(type(record.get("calculation_marker")) is bool and record["calculation_marker"] == expected_marker,
            "Checkpoint calculation marker mismatch")
    n = len(retained)
    stats = {"retained_stat_token_count": n, "retained_mean_logprob": sum(initial["token_logprobs"]) / n,
             "retained_mean_entropy": sum(initial["token_entropies"]) / n}
    require(record.get("retained_features") == stats, "Retained feature statistics mismatch")


def audit_rollout(row, item, cp, manifest, tokenizer, constants, ledger):
    scheduled = item["scheduled"]
    action, seed = scheduled["action"], scheduled["seed"]
    for key in ("index", "repeat", "seed", "shared_prefix_generated_tokens", "generated_tokens", "prompt_tokens_processed"):
        integer(row.get(key), f"outcome {key}")
    require(all(row.get(k) == v for k, v in {"problem_id": item["task"]["id"], "family": item["task"]["family"],
            "index": item["index"], "action": action, "repeat": scheduled["repeat"], "seed": seed,
            "checkpoint_sha256": cp["sha256"], "shared_prefix_generated_tokens": cp["initial"]["generated_tokens"]}.items()),
            "Outcome differs from frozen task/checkpoint/schedule")
    calls = row.get("calls")
    require(isinstance(calls, list), "Missing rollout call records")
    prompt, retained = cp["prompt_ids"], cp["retained_ids"]
    assistant, overhead, position, spent, discarded = list(retained), [], 0, 0, 0
    remaining = manifest["budget"] - cp["initial"]["generated_tokens"]
    reserve = manifest["final_reserve"]
    context = {"phase": "continuation", "problem_id": item["task"]["id"], "index": item["index"],
               "action": action, "repeat": scheduled["repeat"]}

    def take(cap, call_seed):
        nonlocal position, spent
        require(position < len(calls), "Missing successive outcome generation call")
        g = ledger.take(context, prompt + assistant, cap, call_seed, row_call=calls[position])
        position += 1
        spent += g["generated_tokens"]
        return g

    if action == "repair":
        cutoff = 0
        for i in range(len(retained) - 1, max(0, len(retained) - 128), -1):
            if tokenizer.decode(retained[:i]).endswith("\n\n"):
                cutoff = i
                break
        cutoff = max(cutoff, len(retained) - 128)
        injection = tokenizer.encode(constants["REPAIR"])
        assistant = retained[:cutoff] + injection
        overhead.append({"kind": "repair_instruction", "tokens": len(injection), "removed_tokens": len(retained) - cutoff})
    if action == "branch":
        cap = min(64, max(1, (remaining - reserve) // 3))
        candidates = [take(cap, seed + 100 + j) for j in range(2)]
        scores = [g["mean_logprob"] if g["mean_logprob"] is not None else -math.inf for g in candidates]
        winner = max(range(2), key=scores.__getitem__)
        assistant += candidates[winner]["token_ids"]
        discarded = candidates[1 - winner]["generated_tokens"]
        overhead.append({"kind": "branch_selection", "winner": winner, "selector": "mean_base_logprob"})
        done = candidates[winner]["finish_reason"] in {"stop", "timeout"}
    else:
        done = False
    allowance = remaining - spent - reserve
    if not done and allowance > 0:
        g = take(allowance, seed + 200)
        assistant += g["token_ids"]
        done = g["finish_reason"] in {"stop", "timeout"}
    if not done and remaining - spent > 0:
        if "FINAL:" not in tokenizer.decode(assistant):
            injection = tokenizer.encode(constants["FINAL"])
            assistant += injection
            overhead.append({"kind": "finalization_instruction", "tokens": len(injection)})
        g = take(remaining - spent, seed + 300)
        assistant += g["token_ids"]
    require(position == len(calls), "Unexpected extra rollout calls")
    require(row.get("overhead") == overhead, "Repair/branch/FINAL overhead or selector mismatch")
    require(row.get("text") == tokenizer.decode(assistant), "Rollout text does not follow selected token chain")
    require(spent <= remaining, "Rollout plus shared prefix exceeds emitted budget")
    for key, actual in (("generated_tokens", spent), ("prompt_tokens_processed", sum(g["prompt_tokens"] for g in calls)),
                        ("elapsed_seconds", sum(g["elapsed_seconds"] for g in calls))):
        equal_number(row.get(key), actual, key)
    return discarded


def indexed(records, label):
    require(isinstance(records, list), f"Malformed {label} records")
    result = {}
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get("problem_id"), str), f"Malformed {label} identity")
        require(record["problem_id"] not in result, f"Duplicate {label} problem")
        result[record["problem_id"]] = record
    return result


def audit(run, *, tokenizer_dir=None, tokenizer=None):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    require(isinstance(manifest, dict) and manifest.get("status") == "complete", "Only completed mechanism runs can pass this audit")
    require(manifest.get("protocol") == PROTOCOL, "Unsupported mechanism protocol")
    summary = read_json(run / "summary.json")
    require(isinstance(summary, dict) and summary.get("status") == "complete", "Missing completed summary")
    hashes, constants, tasks, rubric, config = frozen_sources(run, manifest)
    fixed = {"protocol": PROTOCOL, "seed": 491027, "repeats": 4, "budget": 1024,
             "checkpoint_target": 256, "checkpoint_cap": 384, "final_reserve": 96,
             "temperature": .7, "top_p": .9, "actions": list(ACTIONS)}
    require(all(config.get(k) == v and manifest.get(k) == v for k, v in fixed.items()), "Launch config/manifest differs from supported mechanism plan")
    for key in ("seed", "repeats", "budget", "checkpoint_target", "checkpoint_cap", "final_reserve"):
        integer(manifest.get(key), f"manifest {key}")
    require(manifest.get("task_seed") == manifest["seed"] and manifest.get("sampling_seed") == manifest["seed"], "Task/sampling seed mismatch")
    start, count = integer(manifest.get("start_index"), "start_index"), integer(manifest.get("problems"), "problems", 1)
    require(start + count <= 24, "Shard exceeds fixed enrollment range")
    require(manifest.get("prompt_suffix") == constants["PROMPT_SUFFIX"], "Prompt framing differs from frozen source")
    require(manifest.get("selector") == "mean_base_logprob", "Unknown branch selector")
    numeric(manifest.get("started_unix"), "run start time")
    require(numeric(manifest.get("finished_unix"), "run finish time") >= manifest["started_unix"], "Run time order mismatch")
    numeric(manifest.get("wall_elapsed_seconds"), "run wall elapsed seconds")
    numeric(summary.get("wall_elapsed_seconds"), "summary wall elapsed seconds")
    for name, key in (("schedule.json", "schedule_sha256"), ("rubric.json", "rubric_sha256")):
        require(digest((run / name).read_bytes()) == sha(manifest.get(key)), "Frozen schedule/rubric hash mismatch")
    require(read_json(run / "rubric.json") == rubric and manifest.get("schema") == rubric["schema"], "Frozen rubric/schema mismatch")
    schedule = read_json(run / "schedule.json")
    require(isinstance(schedule, list) and len(schedule) == count, "Missing enrolled schedule items")
    expected_schedule = []
    for index in range(start, start + count):
        task = tasks["make_task"](index, manifest["seed"])
        task["prompt"] += constants["PROMPT_SUFFIX"]
        arms = [{"action": a, "repeat": r, "seed": manifest["seed"] + index * 10000 + ACTIONS.index(a) * 1000 + r * 10}
                for a in ACTIONS for r in range(manifest["repeats"])]
        random.Random(manifest["seed"] + index).shuffle(arms)
        expected_schedule.append({"index": index, "task": task, "initial_seed": manifest["seed"] + index, "schedule": arms})
    require(schedule == expected_schedule, "Schedule/tasks/seeds do not match frozen enrollment")
    tokenizer, model_evidence = model_tokenizer(manifest, tokenizer_dir, tokenizer)
    ledger = Ledger(run, tokenizer)
    checkpoints = indexed(read_jsonl(run / "checkpoints.jsonl"), "checkpoint")
    attempts = indexed(read_jsonl(run / "checkpoint_attempts.jsonl"), "checkpoint attempt")
    skipped = indexed(read_json(run / "skipped.json"), "skip")
    pids = {item["task"]["id"] for item in schedule}
    require(not (checkpoints.keys() & skipped.keys()) and (checkpoints.keys() | skipped.keys()) == pids,
            "Missing, duplicate or replaced enrolled problem")
    require(attempts.keys() == checkpoints.keys(), "Durable checkpoint attempts do not match completed checkpoints")
    require(list(checkpoints) == list(attempts) == [item["task"]["id"] for item in schedule if item["task"]["id"] in checkpoints],
            "Checkpoint records do not follow enrolled problem order")
    require(list(skipped) == [item["task"]["id"] for item in schedule if item["task"]["id"] in skipped],
            "Skip records do not follow enrolled problem order")
    rows = read_jsonl(run / "outcomes.jsonl")
    row_index, disagreements, disagreement_count, discarded = 0, [], 0, 0

    def reverify(recorded, task, text, context):
        nonlocal disagreement_count
        require(isinstance(recorded, dict) and type(recorded.get("success")) is bool, "Malformed recorded terminal label")
        checked = tasks["verify"](task, text)
        if recorded != checked:
            disagreement_count += 1
            if len(disagreements) < 20:
                disagreements.append({**context, "recorded_success": recorded["success"],
                                      "reverified_success": checked["success"], "reverified_reason": checked["reason"]})

    for item in schedule:
        pid = item["task"]["id"]
        record = checkpoints.get(pid, skipped.get(pid))
        integer(record.get("index"), "initial record index")
        require(isinstance(record.get("initial"), dict), "Missing saved initial generation copy")
        prompt = token_ids(record.get("prompt_ids"), "saved initial prompt")
        require(prompt and item["task"]["prompt"] in tokenizer.decode(prompt), "Saved prompt does not contain the enrolled task")
        initial = ledger.take({"phase": "initial", "problem_id": pid, "index": item["index"]}, prompt,
                              manifest["checkpoint_cap"], item["initial_seed"], checkpoint=True, row_call=record.get("initial"))
        online_boundary(initial, manifest["checkpoint_target"], tokenizer)
        if pid in skipped:
            require(initial["finish_reason"] != "checkpoint", "Eligible checkpoint incorrectly skipped")
            reason = ("completed_before_checkpoint" if initial["finish_reason"] == "stop" else
                      "timeout_before_checkpoint" if initial["finish_reason"] == "timeout" else
                      "answer_phase_reached_before_checkpoint" if "FINAL:" in initial["text"] else "no_boundary_before_cap")
            require(record.get("task") == item["task"] and record.get("index") == item["index"] and record.get("reason") == reason,
                    "Skipped task/reason mismatch")
            reverify(record.get("initial_outcome"), item["task"], initial["text"], {"problem_id": pid, "phase": "skipped_initial"})
            continue
        require(initial["finish_reason"] == "checkpoint", "Checkpoint lacks actual online stop event")
        checkpoint(record, item, initial, tokenizer)
        require({k: v for k, v in record.items() if k != "jev"} == attempts[pid], "Checkpoint differs from pre-judgment durable attempt")
        require("jev_error" not in record, "Completed run contains failed checkpoint judgment")
        require(("jev" in record) == bool(manifest.get("jev")), "Optional judgment presence differs from manifest")
        for scheduled in item["schedule"]:
            require(row_index < len(rows), "Missing scheduled arm/repeat outcome")
            row = rows[row_index]
            discarded += audit_rollout(row, {**item, "scheduled": scheduled}, record, manifest, tokenizer, constants, ledger)
            reverify(row.get("outcome"), item["task"], row["text"], {"problem_id": pid, "action": scheduled["action"], "repeat": scheduled["repeat"]})
            row_index += 1
    require(row_index == len(rows), "Unexpected, duplicate or unscheduled outcomes")
    require(ledger.cursor == len(ledger.events), "Unassigned durable generation calls remain")
    generations = [event["generation"] for event in ledger.events]
    token_total = sum(g["generated_tokens"] for g in generations)
    elapsed = sum(g["elapsed_seconds"] for g in generations)
    initial_total = sum(r["initial"]["generated_tokens"] for r in list(checkpoints.values()) + list(skipped.values()))
    expected_summary = {"attempted_problems": count, "recorded_problems": count, "eligible_problems": len(checkpoints),
                        "skipped_problems": len(skipped), "episodes": len(rows), "planned_episodes": count * 12,
                        "continuation_generated_tokens": sum(r["generated_tokens"] for r in rows),
                        "continuation_seconds": sum(r["elapsed_seconds"] for r in rows),
                        "actual_shared_prefix_tokens": initial_total,
                        "durable_completed_call_generated_tokens": token_total,
                        "durable_completed_call_seconds": elapsed, "failed_calls_with_unknown_work": 0}
    for key, value in expected_summary.items():
        equal_number(summary.get(key), value, f"summary {key}")
    require(summary.get("generation_finish_reasons") == dict(Counter(g["finish_reason"] for g in generations)), "Summary finish reason mismatch")
    require(summary.get("skip_reasons") == dict(Counter(r["reason"] for r in skipped.values())), "Summary skip reason mismatch")
    require(summary.get("actions") == {a: {"n": sum(r["action"] == a for r in rows),
            "successes": sum(r["outcome"]["success"] for r in rows if r["action"] == a)} for a in ACTIONS}, "Summary original action labels/counts mismatch")
    require(token_total == initial_total + sum(r["generated_tokens"] for r in rows), "Shared-prefix plus rollout ledger mismatch")
    input_names = ("manifest.json", "summary.json", "schedule.json", "rubric.json", "checkpoint_attempts.jsonl",
                   "checkpoints.jsonl", "skipped.json", "outcomes.jsonl", "generation_started.jsonl", "generation_events.jsonl")
    return {"schema_version": 1, "status": "passed_integrity_audit" if not disagreement_count else "outcome_label_disagreements",
            "ready_for_statistical_analysis": not disagreement_count, "protocol": PROTOCOL,
            "counts": {"planned_problems": count, "eligible_problems": len(checkpoints), "skipped_problems": len(skipped),
                       "episodes": len(rows), "durable_calls": len(generations), "checkpoint_attempts": len(attempts)},
            "accounting": {"generated_tokens": token_total, "prompt_tokens_processed": sum(g["prompt_tokens"] for g in generations),
                           "elapsed_seconds": elapsed, "shared_initial_generated_tokens": initial_total,
                           "discarded_branch_generated_tokens": discarded, "unknown_work_calls": 0,
                           "wall_elapsed_seconds": summary["wall_elapsed_seconds"],
                           "inserted_instruction_tokens": sum(x.get("tokens", 0) for row in rows for x in row["overhead"])},
            "outcome_reverification": {"disagreement_count": disagreement_count, "examples": disagreements,
                                       "example_limit": 20, "original_records_modified": False},
            "checks": {"launch_source_and_config_hashes": True, "supported_frozen_semantic_contracts": True,
                       "frozen_tasks_schedule_seeds_and_rubric": True, "online_checkpoint_and_retained_state": True,
                       "durable_starts_events_and_saved_call_copies": True, "repair_branch_finalization_token_chains": True,
                       "all_emitted_budgets_and_saved_costs": True, "skip_and_completion_accounting": True},
            "model_tokenizer_identity": model_evidence,
            "provenance": {"launch_source_sha256": hashes, "auditor_sha256": digest(Path(__file__).read_bytes()),
                           "validator_utilities_sha256": digest((ROOT / "scripts/analyze_development.py").read_bytes()),
                           "input_sha256": {name: digest((run / name).read_bytes()) for name in input_names},
                           "supported_contract_ast_sha256": CONTRACTS},
            "limitations": ["This is a structural post-completion audit, not a replay of stochastic model computation or proof of model correctness.",
                            "Saved original prompt IDs anchor the chain; task text is checked inside their decoding, but chat-template reproduction is not asserted.",
                            "Launch-recorded tensor digests are checked internally; only local tokenizer/model metadata bytes are rehashed.",
                            "Optional judgment presence and frozen rubric are checked; no paid API or private provider billing ledger is accessed.",
                            "Repeated continuations remain clustered within original problems; this report performs no effect estimation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, help="Local directory containing launch-matching tokenizer/model metadata")
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "Refusing to overwrite an audit report")
        report = audit(args.run, tokenizer_dir=args.tokenizer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (ValueError, OSError, SyntaxError) as exc:
        parser.error(str(exc) if isinstance(exc, ValueError) else f"Malformed or missing audit input ({type(exc).__name__})")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "outcome_disagreements": report["outcome_reverification"]["disagreement_count"]}))
    if not report["ready_for_statistical_analysis"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
