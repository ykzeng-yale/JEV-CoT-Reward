#!/usr/bin/env python3
"""Analyze all-arm checkpoint replays without model loading or network calls.

Usage:
  .venv/bin/python scripts/analyze_screen.py runs/phase0 --output results/phase0_analysis.json
  .venv/bin/python scripts/analyze_screen.py runs/screen --learned --output results/screen_analysis.json

Cross-fitting evaluates one intervention followed by the logged continuation
policy, conditional on checkpoint eligibility. It is NOT sequential policy
evaluation. Future output text, reference answers, and per-arm outcomes are
never representation inputs. TF-IDF, imputation, scaling, and constant-action
selection are fitted within each training fold only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ACTIONS = ("continue", "repair", "branch")
SEMANTIC_KEYS = (
    "hypothesis", "contradiction", "unsupported", "repeated_failure",
    "localized_error", "testable", "locally_valid",
)
CHEAP_NAMES = (
    "prompt_tokens", "retained_tokens", "prefix_generated_tokens",
    "discarded_prefix_tail_tokens", "remaining_generation_tokens", "budget",
    "retained_mean_entropy", "retained_mean_logprob", "task_characters",
    "history_characters", "segment_characters", "history_paragraphs",
    "segment_words", "segment_unique_word_fraction",
)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: expected an object")
            rows.append(row)
    return rows


def number(value: Any, *, missing: float = math.nan) -> float:
    if value is None:
        return missing
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric field")
    return float(value)


def nonnegative(value: Any) -> float:
    result = number(value, missing=0.0)
    if result < 0:
        raise ValueError("Negative resource count")
    return result


def semantic_features(record: dict | None) -> np.ndarray:
    """Accept adapter records, answer dictionaries, or explicit features maps."""
    if not record or record.get("error") or record.get("status") in {"failed", "error"}:
        return np.full(len(SEMANTIC_KEYS), np.nan)
    source = record.get("response", record)
    if source is None:
        return np.full(len(SEMANTIC_KEYS), np.nan)
    if not isinstance(source, dict):
        raise ValueError("Typed feature response must be an object")
    answers = source.get("answers", source.get("features", source.get("probabilities", {}))) or {}
    if not isinstance(answers, dict):
        raise ValueError("Typed feature answers must be an object")
    values = []
    for key in SEMANTIC_KEYS:
        value = answers.get(key)
        if isinstance(value, dict):
            value = value.get("noul")
        value = number(value)
        if math.isfinite(value) and not 0 <= value <= 1:
            raise ValueError(f"Typed feature {key} is outside [0, 1]")
        values.append(value)
    return np.asarray(values, dtype=float)


def observable_features(checkpoint: dict, budget: int) -> tuple[str, np.ndarray]:
    state = checkpoint.get("state", {})
    parts = [state.get(name, "") for name in ("task", "history", "latest_segment")]
    if not all(isinstance(part, str) for part in parts):
        raise ValueError("Checkpoint state fields must be strings")
    task, history, segment = parts
    initial = checkpoint["initial"]
    discarded = nonnegative(checkpoint.get("discarded_prefix_tail_tokens", 0))
    # Whole-initial-generation averages may include unretained FUTURE tokens.
    # Only consume explicit retained statistics, or initial means when no tail
    # was removed. Never backfill these fields from continuation records.
    stats = {}
    retained = checkpoint.get("retained_features", {})
    if retained and retained.get("retained_stat_token_count") == len(checkpoint.get("retained_ids", [])) and retained.get("retained_stat_token_count", 0) > 0:
        stats = {"mean_entropy": retained.get("retained_mean_entropy"),
                 "mean_logprob": retained.get("retained_mean_logprob")}
    no_future_tokens = (discarded == 0 and initial.get("generated_tokens") == len(checkpoint.get("retained_ids", []))
                        and ("token_ids" not in initial or initial["token_ids"] == checkpoint.get("retained_ids")))
    if not stats and no_future_tokens:
        stats = initial
    words = segment.lower().split()
    generated = nonnegative(initial.get("generated_tokens"))
    numeric = np.asarray([
        len(checkpoint.get("prompt_ids", [])), len(checkpoint.get("retained_ids", [])),
        generated, discarded, max(0, budget - generated), budget,
        number(stats.get("mean_entropy")), number(stats.get("mean_logprob")),
        len(task), len(history), len(segment),
        len([part for part in history.split("\n\n") if part.strip()]),
        len(words), len(set(words)) / len(words) if words else 0,
    ], dtype=float)
    return "\n".join(parts), numeric


def episode_resources(row: dict, issues: list[str]) -> dict[str, float | None]:
    """Generation call sums are authoritative and include losing candidates."""
    calls = row.get("calls", [])
    if calls:
        for call in calls:
            if "token_ids" in call and call.get("generated_tokens") != len(call["token_ids"]):
                issues.append(f"{row['problem_id']}:{row['action']}:{row['repeat']}: generated count differs from emitted IDs")
        generated = sum(nonnegative(call.get("generated_tokens")) for call in calls)
        prompt = sum(nonnegative(call.get("prompt_tokens")) for call in calls)
        elapsed = sum(nonnegative(call.get("elapsed_seconds")) for call in calls)
        for key, measured in (("generated_tokens", generated), ("prompt_tokens_processed", prompt), ("elapsed_seconds", elapsed)):
            recorded = row.get(key)
            if recorded is not None and not math.isclose(nonnegative(recorded), measured, rel_tol=1e-6, abs_tol=1e-6):
                issues.append(f"{row['problem_id']}:{row['action']}:{row['repeat']}: {key} differs from call sum")
    else:
        generated = nonnegative(row.get("generated_tokens"))
        prompt = nonnegative(row.get("prompt_tokens_processed"))
        elapsed = nonnegative(row.get("elapsed_seconds"))
        if generated:
            issues.append(f"{row['problem_id']}:{row['action']}:{row['repeat']}: no calls; discarded-work accounting unverified")
    discarded_branch: float | None = 0.0
    overhead = row.get("overhead", [])
    if row["action"] == "branch":
        selector = next((item for item in overhead if item.get("kind") == "branch_selection"), None)
        if selector and type(selector.get("winner")) is int and selector["winner"] in (0, 1) and len(calls) >= 2:
            discarded_branch = nonnegative(calls[1 - selector["winner"]].get("generated_tokens"))
        else:
            discarded_branch = None
            issues.append(f"{row['problem_id']}: discarded branch breakdown unavailable; total call work still counted")
    return {
        "generated_tokens": generated, "prompt_tokens_processed": prompt,
        "elapsed_seconds": elapsed, "discarded_branch_tokens": discarded_branch,
        "repair_removed_tokens": sum(nonnegative(item.get("removed_tokens")) for item in overhead),
        "inserted_instruction_tokens": sum(nonnegative(item.get("tokens")) for item in overhead),
    }


def stratified_bootstrap_indices(families: list[str], draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.asarray(families) == family) for family in sorted(set(families))]
    if not groups:
        return np.empty((draws, 0), dtype=int)
    return np.concatenate([rng.choice(group, size=(draws, len(group)), replace=True) for group in groups], axis=1)


def estimate(values: list[float] | np.ndarray, indices: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n_problems": 0, "mean": None, "bootstrap_percentile_95": None}
    resampled = values[indices].mean(axis=1)
    return {
        "n_problems": len(values), "mean": float(values.mean()),
        "bootstrap_percentile_95": [float(x) for x in np.quantile(resampled, [0.025, 0.975])],
    }


def fit_predict(train_docs, test_docs, train_numeric, test_numeric, train_rows, train_ids, test_ids, alpha):
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4096, sublinear_tf=True)
    try:
        train_text = vectorizer.fit_transform(train_docs)
        test_text = vectorizer.transform(test_docs)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        train_text = sparse.csr_matrix((len(train_docs), 0))
        test_text = sparse.csr_matrix((len(test_docs), 0))
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    train_num = scaler.fit_transform(imputer.fit_transform(train_numeric))
    test_num = scaler.transform(imputer.transform(test_numeric))
    x_train = sparse.hstack([train_text, sparse.csr_matrix(train_num)], format="csr")
    x_test = sparse.hstack([test_text, sparse.csr_matrix(test_num)], format="csr")
    locations = {pid: index for index, pid in enumerate(train_ids)}
    predictions = np.empty((len(test_ids), len(ACTIONS)))
    for action_index, action in enumerate(ACTIONS):
        selected = [row for row in train_rows if row["action"] == action]
        x = x_train[[locations[row["problem_id"]] for row in selected]]
        # Every Bernoulli continuation is retained. No argmax-arm pseudo-labels.
        y = np.asarray([row["outcome"]["success"] for row in selected], dtype=float)
        counts = Counter(row["problem_id"] for row in selected)
        weights = np.asarray([1.0 / counts[row["problem_id"]] for row in selected])
        weights *= len(weights) / weights.sum()
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(x, y, sample_weight=weights)
        predictions[:, action_index] = np.clip(model.predict(x_test), 0, 1)
    return predictions, int(train_text.shape[1])


def crossfit(problems: list[dict], rows: list[dict], checkpoints: dict, budget: int,
             bootstrap: np.ndarray, alpha: float, local_features: dict) -> dict:
    from sklearn.model_selection import GroupKFold
    ids = [problem["problem_id"] for problem in problems]
    lookup = {pid: i for i, pid in enumerate(ids)}
    docs, numeric = zip(*(observable_features(checkpoints[pid], budget) for pid in ids))
    cheap = np.asarray(numeric)
    jev = np.vstack([semantic_features(checkpoints[pid].get("jev")) for pid in ids])
    local = np.vstack([semantic_features(local_features.get(pid, checkpoints[pid].get("local_judge"))) for pid in ids])
    representations = {"cheap_tfidf": cheap}
    if np.isfinite(jev).any():
        representations["cheap_tfidf_plus_jev"] = np.hstack([cheap, jev])
    if np.isfinite(local).any():
        representations["cheap_tfidf_plus_local"] = np.hstack([cheap, local])
    prediction = {name: np.empty((len(ids), len(ACTIONS))) for name in representations}
    constant_actions = np.empty(len(ids), dtype=int)
    family_constant_actions = np.empty(len(ids), dtype=int)
    fold_id = np.empty(len(ids), dtype=int)
    folds = []
    rates = np.asarray([[problem["actions"][action]["success_rate"] for action in ACTIONS] for problem in problems])
    for fold, (train, test) in enumerate(GroupKFold(n_splits=min(5, len(ids))).split(np.zeros(len(rows)), groups=[row["problem_id"] for row in rows])):
        train_rows = [rows[i] for i in train]
        train_ids = sorted({row["problem_id"] for row in train_rows})
        test_ids = sorted({rows[i]["problem_id"] for i in test})
        train_i = [lookup[pid] for pid in train_ids]
        test_i = [lookup[pid] for pid in test_ids]
        constant = int(np.argmax(rates[train_i].mean(axis=0)))
        constant_actions[test_i] = constant
        family_constants = {}
        for family in sorted({problems[i]["family"] for i in test_i}):
            family_train = [i for i in train_i if problems[i]["family"] == family]
            family_constants[family] = int(np.argmax(rates[family_train].mean(axis=0))) if family_train else constant
        for i in test_i:
            family_constant_actions[i] = family_constants[problems[i]["family"]]
        fold_id[test_i] = fold
        vocabulary_size = None
        for name, features in representations.items():
            predicted, vocabulary_size = fit_predict(
                [docs[i] for i in train_i], [docs[i] for i in test_i], features[train_i], features[test_i],
                train_rows, train_ids, test_ids, alpha,
            )
            prediction[name][test_i] = predicted
        folds.append({"fold": fold, "train_problems": train_ids, "test_problems": test_ids,
                      "training_selected_constant": ACTIONS[constant],
                      "training_selected_family_constants": {family: ACTIONS[action] for family, action in family_constants.items()},
                      "tfidf_vocabulary_size": vocabulary_size})
    selected = {name: values.argmax(axis=1) for name, values in prediction.items()}
    selected["training_selected_constant"] = constant_actions
    selected["training_selected_family_constant"] = family_constant_actions
    selected["always_continue"] = np.zeros(len(ids), dtype=int)
    values = {name: rates[np.arange(len(ids)), actions] for name, actions in selected.items()}
    policy = {}
    for name, actions in selected.items():
        resources = {}
        for metric in ("deployment_generated_tokens", "deployment_prompt_tokens_processed", "deployment_service_seconds"):
            resources[metric] = float(np.mean([problems[i]["actions"][ACTIONS[action]][metric] for i, action in enumerate(actions)]))
        for output_name, field in (("jev_uncached_equivalent_usd_per_problem", "input_cost_usd"),
                                   ("jev_recorded_service_seconds_per_problem", "elapsed_seconds")):
            measured = [checkpoints[pid].get("jev", {}).get(field) for pid in ids]
            resources[output_name] = (float(np.mean([nonnegative(value) for value in measured]))
                                      if all(value is not None for value in measured) else None) if name.endswith("plus_jev") else 0.0
        if name.endswith("plus_jev"):
            resources["jev_cost_measured_problems"] = sum(checkpoints[pid].get("jev", {}).get("input_cost_usd") is not None for pid in ids)
        if name.endswith("plus_local"):
            measured = [local_features.get(pid, checkpoints[pid].get("local_judge", {})).get("generation") for pid in ids]
            available = [record for record in measured if isinstance(record, dict)]
            resources["local_acquisition_measured_problems"] = len(available)
            resources["local_generated_tokens_per_measured_problem"] = float(np.mean([nonnegative(record.get("generated_tokens")) for record in available])) if available else None
            resources["local_prompt_tokens_per_measured_problem"] = float(np.mean([nonnegative(record.get("prompt_tokens")) for record in available])) if available else None
            resources["local_service_seconds_per_measured_problem"] = float(np.mean([nonnegative(record.get("elapsed_seconds")) for record in available])) if available else None
        policy[name] = {"success": estimate(values[name], bootstrap), "action_counts": dict(Counter(ACTIONS[action] for action in actions)), "costs": resources}
    brier = {}
    for name, predicted in prediction.items():
        per_problem = []
        for i, pid in enumerate(ids):
            losses = [(predicted[i, ACTIONS.index(row["action"])] - int(row["outcome"]["success"])) ** 2 for row in rows if row["problem_id"] == pid]
            per_problem.append(float(np.mean(losses)))
        brier[name] = estimate(per_problem, bootstrap)
    contrasts = {}
    pairs = [(name, "training_selected_constant") for name in prediction]
    pairs += [(name, "training_selected_family_constant") for name in prediction]
    pairs += [(name, "cheap_tfidf") for name in prediction if name != "cheap_tfidf"]
    if "cheap_tfidf_plus_jev" in prediction and "cheap_tfidf_plus_local" in prediction:
        pairs.append(("cheap_tfidf_plus_jev", "cheap_tfidf_plus_local"))
    for left, right in pairs:
        contrasts[f"{left}_minus_{right}"] = estimate(values[left] - values[right], bootstrap)
    families = {}
    for family in sorted({problem["family"] for problem in problems}):
        members = [i for i, problem in enumerate(problems) if problem["family"] == family]
        global_to_local = np.full(len(ids), -1, dtype=int)
        global_to_local[members] = np.arange(len(members))
        mapped = global_to_local[bootstrap]
        local_indices = mapped[mapped >= 0].reshape(len(bootstrap), len(members))
        families[family] = {
            "n_problems": len(members),
            "policies": {name: estimate(result[members], local_indices) for name, result in values.items()},
            "paired_policy_differences": {f"{left}_minus_{right}": estimate((values[left] - values[right])[members], local_indices)
                                          for left, right in pairs},
        }
    return {
        "status": "exploratory_grouped_crossfit", "n_problems": len(ids), "learner": "per-action Ridge; clipped to [0,1]",
        "ridge_alpha_fixed_without_test_tuning": alpha, "cheap_numeric_features": list(CHEAP_NAMES),
        "semantic_features": list(SEMANTIC_KEYS), "folds": folds, "policies": policy,
        "constant_baselines": {"training_selected_constant": "One global action selected using training problems only; stable ACTIONS-order ties.",
                               "training_selected_family_constant": "Action selected within each observed task family using training problems only; unseen families fall back to the training-global action."},
        "family_stratified": families,
        "paired_policy_differences": contrasts, "problem_weighted_bernoulli_brier": brier,
        "feature_coverage": {"jev_observed_fraction": float(np.isfinite(jev).mean()), "local_observed_fraction": float(np.isfinite(local).mean()),
                             "retained_entropy_observed_problems": int(np.isfinite(cheap[:, 6]).sum())},
        "per_problem": [{"problem_id": pid, "fold": int(fold_id[i]), "selected_actions": {name: ACTIONS[actions[i]] for name, actions in selected.items()},
                         "predicted_success_by_action": {name: dict(zip(ACTIONS, map(float, predicted[i]))) for name, predicted in prediction.items()}} for i, pid in enumerate(ids)],
        "limitations": [
            "Single-checkpoint all-arm replay value, conditional on checkpoint eligibility; not sequential deployment.",
            "Bootstrap intervals resample held-out problem contributions without refitting folds; training-instability uncertainty is not included.",
            "Generation envelope is shared; judge acquisition latency and money are additional and are reported separately, not proven cost-neutral.",
            "No hyperparameter selection on held-out outcomes; an independent frozen-policy test is still needed.",
            "Local-judge acquisition costs are not inferred when the local-feature file does not supply a measured cost ledger.",
            "Task family is observable and can confound pooled feature gains. Family-constant controls and within-family contrasts do not establish transfer to an unseen family.",
            "Missing judge values are training-fold-imputed with missingness indicators; gains may reflect acquisition failure patterns rather than semantic content.",
        ],
    }


def state_sha256(state):
    allowed = {name: state[name] for name in ("task", "history", "latest_segment")}
    if not all(isinstance(value, str) for value in allowed.values()):
        raise ValueError("Checkpoint state fields must be strings")
    return hashlib.sha256(json.dumps(allowed, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def public_source_omissions(run_dir, manifest):
    release = read_json(run_dir / "release_manifest.json")
    if release is None:
        return set()
    if (release.get("format_version") != "public-local-diagnostics-v1"
            or release.get("original_run_id") != manifest.get("original_run_id")):
        raise ValueError("Unsupported or mismatched public all-arm release")
    hashes = release.get("released_files_sha256", {})
    if not {"manifest.json", "checkpoints.jsonl", "outcomes.jsonl", "summary.json", "source_manifest.json"} <= hashes.keys():
        raise ValueError("Public release lacks required file hashes")
    for name, expected in hashes.items():
        path = run_dir / name
        if (not path.resolve().is_relative_to(run_dir.resolve()) or path.is_symlink()
                or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"Public release hash/path mismatch: {name}")
    sources = read_json(run_dir / "source_manifest.json")
    omitted = set()
    for item in sources.get("omitted_original_sources", []):
        name = item.get("original_filename")
        if (name != "jev.py" or name in omitted or name not in manifest.get("source_sha256", {})
                or item.get("original_declared_sha256") != manifest["source_sha256"][name]
                or (run_dir / "source" / name).exists()):
            raise ValueError("Public release may omit only its explicitly declared hosted adapter")
        omitted.add(name)
    entries = {item["released_path"]: item for item in sources.get("files", [])}
    for name, expected in manifest.get("source_sha256", {}).items():
        if name in omitted:
            continue
        relative = "source/" + name
        item = entries.get(relative, {})
        if (item.get("sha256") != expected or item.get("original_declared_sha256") != expected
                or item.get("status") != "captured_run_source" or hashes.get(relative) != expected
                or item.get("source_run_id") != manifest.get("original_run_id")):
            raise ValueError("Public captured source does not match original inventory")
    return omitted


def audit_sources_and_schedule(run_dir, manifest, checkpoints, skipped, rows):
    """Verify frozen artifacts, never compare them with mutable live sources."""
    mechanism = manifest.get("protocol", "").startswith("mechanism-v1")
    omitted = public_source_omissions(run_dir, manifest)
    verified = []
    for relative, expected_hash in manifest.get("source_sha256", {}).items():
        if relative in omitted:
            continue
        path = run_dir / "source" / relative
        if not path.resolve().is_relative_to((run_dir / "source").resolve()):
            raise ValueError("Source snapshot path escapes the run")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen source missing or hash mismatch: {relative}")
        verified.append(relative)
    for filename, field in (("schedule.json", "schedule_sha256"), ("rubric.json", "rubric_sha256")):
        expected_hash = manifest.get(field)
        if expected_hash:
            path = run_dir / filename
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Frozen {filename} hash mismatch")
        elif mechanism:
            raise ValueError(f"Mechanism run missing {field}")
    if mechanism and not verified:
        raise ValueError("Mechanism run missing frozen sources")
    plan = read_json(run_dir / "schedule.json")
    if mechanism and not isinstance(plan, list):
        raise ValueError("Mechanism run missing prespecified schedule")
    if plan is not None:
        if len(plan) != manifest["problems"]:
            raise ValueError("Schedule length differs from planned enrollment")
        by_id = {}
        for item in plan:
            pid = item["task"]["id"]
            if pid in by_id:
                raise ValueError("Duplicate scheduled problem")
            schedule = {(entry["action"], entry["repeat"]): entry for entry in item["schedule"]}
            expected = {(action, repeat) for action in ACTIONS for repeat in range(manifest["repeats"])}
            if len(schedule) != len(item["schedule"]) or set(schedule) != expected:
                raise ValueError("Prespecified schedule is not the complete fixed arm/replicate set")
            by_id[pid] = (item, schedule)
        for cp in list(checkpoints.values()) + skipped:
            pid = cp.get("problem_id", cp["task"]["id"])
            if pid not in by_id or cp["task"] != by_id[pid][0]["task"]:
                raise ValueError("Recorded problem differs from prespecified task")
            if cp["initial"].get("seed") != by_id[pid][0]["initial_seed"]:
                raise ValueError("Initial seed differs from prespecified schedule")
        for row in rows:
            planned = by_id[row["problem_id"]][1].get((row["action"], row["repeat"]))
            if planned is None or row.get("seed") != planned["seed"]:
                raise ValueError("Outcome seed differs from prespecified schedule")
    for cp in checkpoints.values():
        if mechanism:
            digest = hashlib.sha256(json.dumps({"prompt": cp["prompt_ids"], "retained": cp["retained_ids"]}, sort_keys=True).encode()).hexdigest()
            if cp.get("sha256") != digest:
                raise ValueError("Checkpoint token hash mismatch")
            initial = cp["initial"]
            if (initial.get("finish_reason") != "checkpoint" or initial.get("token_ids") != cp["retained_ids"]
                    or initial.get("generated_tokens") != len(cp["retained_ids"])
                    or cp.get("discarded_prefix_tail_tokens") != 0):
                raise ValueError("Mechanism checkpoint is not an exact online emitted prefix")
    return {"verified_source_files": verified, "schedule_verified": plan is not None,
            "declared_unverified_public_omissions": sorted(omitted),
            "status": "verified" if verified else "historical_source_identity_unavailable",
            "model_weight_identity": "recorded in manifest; external model files are not reread by analysis"}


def durable_resources(run_dir, issues):
    path = run_dir / "generation_events.jsonl"
    if not path.exists():
        return None
    events, starts = read_jsonl(path), read_jsonl(run_dir / "generation_started.jsonl")
    by_sequence = {event["sequence"]: event for event in events}
    start_sequences = [event["sequence"] for event in starts]
    if len(by_sequence) != len(events) or len(set(start_sequences)) != len(starts):
        raise ValueError("Duplicate durable generation sequence")
    unmatched = sorted(set(start_sequences) - set(by_sequence))
    if starts and set(by_sequence) - set(start_sequences):
        raise ValueError("Completed generation lacks its durable start")
    failed = [event for event in events if event.get("status") == "error"]
    if unmatched or failed:
        issues.append("Interrupted/failed calls have unknown generated or computed work; recorded totals are lower bounds")
    completed = [event for event in events if event.get("status") == "complete"]
    for event in completed:
        g = event["generation"]
        if (g.get("generated_tokens") != len(g.get("token_ids", []))
                or g.get("prompt_tokens") != len(event.get("prefix_ids", []))):
            raise ValueError("Durable generation counts disagree with emitted/prompt IDs")
    def total(field, phase=None):
        return sum(nonnegative(event["generation"].get(field)) for event in completed
                   if phase is None or event.get("phase") == phase)
    return {"completed_calls": len(completed), "failed_calls": len(failed), "unmatched_started_sequences": unmatched,
            "generated_tokens": total("generated_tokens"), "prompt_tokens_processed": total("prompt_tokens"),
            "model_service_seconds": total("elapsed_seconds"),
            "initial_generated_tokens": total("generated_tokens", "initial"),
            "continuation_generated_tokens": total("generated_tokens", "continuation"),
            "completed_generations": [event["generation"] for event in completed]}


def audit_local_features(run_dir, manifest, checkpoints, local_features):
    mechanism = manifest.get("protocol", "").startswith("mechanism-v1")
    provenance_path = run_dir / "local_judge_provenance.json"
    provenance_digest = hashlib.sha256(provenance_path.read_bytes()).hexdigest() if provenance_path.exists() else None
    if mechanism and set(local_features) & set(checkpoints):
        provenance = read_json(provenance_path, {})
        if (provenance.get("version") != "local-judge-v2"
                or not provenance.get("model_identity", {}).get("weight_revision_sha256")
                or provenance.get("checkpoint_file_sha256") != hashlib.sha256((run_dir / "checkpoints.jsonl").read_bytes()).hexdigest()
                or provenance.get("run_manifest_sha256") != hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
                or provenance.get("rubric_sha256") != manifest.get("rubric_sha256")
                or "scripts/local_judge.py" not in provenance.get("source_sha256", {})):
            raise ValueError("Mechanism local-judge provenance identity is incomplete or mismatched")
        source_root = run_dir / "local_judge_artifacts/source"
        for relative, expected in provenance["source_sha256"].items():
            path = source_root / relative
            if (not path.resolve().is_relative_to(source_root.resolve()) or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                raise ValueError("Local-judge frozen source identity mismatch")
    for pid, record in local_features.items():
        if pid not in checkpoints:
            continue
        if mechanism:
            cp = checkpoints[pid]
            if (record.get("checkpoint_sha256") != cp["sha256"]
                    or record.get("state_sha256") != state_sha256(cp["state"])
                    or record.get("rubric_sha256") != manifest.get("rubric_sha256")
                    or not provenance_digest or record.get("provenance_sha256") != provenance_digest):
                raise ValueError("Mechanism local-judge record lacks matching checkpoint/state/rubric/provenance identity")
    starts = read_jsonl(run_dir / "local_judge_started.jsonl")
    started_ids = [record["problem_id"] for record in starts]
    if len(set(started_ids)) != len(started_ids):
        raise ValueError("Duplicate local-judge started problem IDs")
    return {"identity_policy": "required_for_mechanism_v1; historical records remain explicitly unverified",
            "provenance_sha256": provenance_digest, "in_run_records": len(set(local_features) & set(checkpoints)),
            "unmatched_started_problem_ids": sorted(set(started_ids) & set(checkpoints) - set(local_features)),
            "outside_run_records": sorted(set(local_features) - set(checkpoints))}


def analyze(run_dir: Path, *, draws=2000, seed=20260927, learned=False, alpha=10.0, local_features=None) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("A run_screen manifest.json is required")
    checkpoint_rows = read_jsonl(run_dir / "checkpoints.jsonl")
    rows = read_jsonl(run_dir / "outcomes.jsonl")
    skipped = read_json(run_dir / "skipped.json", [])
    run_summary = read_json(run_dir / "summary.json", {})
    review_audit = read_json(run_dir / "review_audit.json", {})
    local_features = local_features or {}
    if draws < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Positive bootstrap draws and finite positive ridge alpha required")
    checkpoints = {}
    for checkpoint in checkpoint_rows:
        pid = checkpoint["problem_id"]
        if pid in checkpoints:
            raise ValueError("Duplicate problem checkpoint; this analyzer expects one checkpoint per problem")
        checkpoints[pid] = checkpoint
    skip_ids = [record.get("problem_id", record["task"]["id"]) for record in skipped]
    if len(set(skip_ids)) != len(skip_ids) or set(skip_ids) & set(checkpoints):
        raise ValueError("Duplicate enrollment across checkpoints and skipped problems")
    expected = int(manifest.get("repeats", 0))
    if expected <= 0:
        raise ValueError("Positive manifest repeats required for enrollment audit")
    budget = int(manifest.get("budget", 0))
    if budget <= 0:
        raise ValueError("Positive generation budget required")
    grouped = defaultdict(lambda: defaultdict(list))
    seen = set()
    ledger_issues = []
    for row in rows:
        pid, action, repeat = row["problem_id"], row["action"], row["repeat"]
        if pid not in checkpoints or action not in ACTIONS or type(repeat) is not int or not 0 <= repeat < expected:
            raise ValueError("Unknown checkpoint, action, or out-of-range repetition")
        if type(row.get("outcome", {}).get("success")) is not bool:
            raise ValueError("Outcomes must contain literal Boolean success")
        if (pid, action, repeat) in seen:
            raise ValueError("Duplicate problem/action/repetition; refusing double counting")
        seen.add((pid, action, repeat))
        if row.get("checkpoint_sha256") != checkpoints[pid].get("sha256"):
            raise ValueError("Outcome references a different checkpoint hash")
        row["_resources"] = episode_resources(row, ledger_issues)
        if row["_resources"]["generated_tokens"] + nonnegative(checkpoints[pid]["initial"].get("generated_tokens")) > budget:
            ledger_issues.append(f"{pid}:{action}:{repeat}: all-work generation exceeds manifest envelope")
        grouped[pid][action].append(row)
    source_audit = audit_sources_and_schedule(run_dir, manifest, checkpoints, skipped, rows)
    effective_local = {pid: local_features.get(pid, cp.get("local_judge")) for pid, cp in checkpoints.items()
                       if local_features.get(pid, cp.get("local_judge")) is not None}
    local_audit = audit_local_features(run_dir, manifest, checkpoints, {**local_features, **effective_local})
    durable = durable_resources(run_dir, ledger_issues)
    problems = []
    incomplete = []
    for pid in sorted(checkpoints):
        checkpoint = checkpoints[pid]
        counts = {action: len(grouped[pid][action]) for action in ACTIONS}
        if any(count != expected for count in counts.values()):
            incomplete.append({"problem_id": pid, "observed_repeats": counts})
            continue
        initial = checkpoint["initial"]
        action_results = {}
        for action in ACTIONS:
            arm = grouped[pid][action]
            action_results[action] = {
                "successes": sum(row["outcome"]["success"] for row in arm), "n": len(arm),
                "success_rate": float(np.mean([row["outcome"]["success"] for row in arm])),
                "deployment_generated_tokens": nonnegative(initial.get("generated_tokens")) + float(np.mean([row["_resources"]["generated_tokens"] for row in arm])),
                "deployment_prompt_tokens_processed": nonnegative(initial.get("prompt_tokens")) + float(np.mean([row["_resources"]["prompt_tokens_processed"] for row in arm])),
                "deployment_service_seconds": nonnegative(initial.get("elapsed_seconds")) + float(np.mean([row["_resources"]["elapsed_seconds"] for row in arm])),
            }
        problems.append({"problem_id": pid, "family": checkpoint["task"]["family"], "actions": action_results})
    indices = stratified_bootstrap_indices([problem["family"] for problem in problems], draws, seed)
    per_action = {action: estimate([problem["actions"][action]["success_rate"] for problem in problems], indices) for action in ACTIONS}
    differences = {f"{left}_minus_{right}": estimate([problem["actions"][left]["success_rate"] - problem["actions"][right]["success_rate"] for problem in problems], indices)
                   for left, right in (("repair", "continue"), ("branch", "continue"), ("branch", "repair"))}
    initials = [checkpoint["initial"] for checkpoint in checkpoint_rows] + [row["initial"] for row in skipped]
    new_judge = {}
    for checkpoint in checkpoint_rows:
        record = checkpoint.get("jev", {})
        if record and not record.get("cache_hit", False):
            new_judge.setdefault(record.get("request_sha256", checkpoint["problem_id"]), record)
    cost = {
        "collection_prefix_generated_tokens_counted_once": sum(nonnegative(initial.get("generated_tokens")) for initial in initials),
        "collection_continuation_generated_tokens_all_arms": sum(row["_resources"]["generated_tokens"] for row in rows),
        "collection_prompt_tokens_processed": sum(nonnegative(initial.get("prompt_tokens")) for initial in initials) + sum(row["_resources"]["prompt_tokens_processed"] for row in rows),
        "collection_model_service_seconds": sum(nonnegative(initial.get("elapsed_seconds")) for initial in initials) + sum(row["_resources"]["elapsed_seconds"] for row in rows),
        "discarded_prefix_tail_tokens_already_charged": sum(nonnegative(checkpoint.get("discarded_prefix_tail_tokens")) for checkpoint in checkpoint_rows),
        "discarded_branch_tokens_already_charged_known": sum(row["_resources"]["discarded_branch_tokens"] or 0 for row in rows),
        "unknown_discarded_branch_breakdown_episodes": sum(row["_resources"]["discarded_branch_tokens"] is None for row in rows),
        "removed_repair_tokens_already_charged": sum(row["_resources"]["repair_removed_tokens"] for row in rows),
        "inserted_instruction_tokens_not_generation": sum(row["_resources"]["inserted_instruction_tokens"] for row in rows),
        "jev_new_calls_in_this_run": len(new_judge),
        "jev_new_call_usd_in_this_run": sum(nonnegative(record.get("input_cost_usd")) for record in new_judge.values()),
        "jev_new_call_service_seconds": sum(nonnegative(record.get("elapsed_seconds")) for record in new_judge.values()),
        "jev_failed_checkpoints": sum(bool(cp.get("jev_error")) for cp in checkpoint_rows),
        "jev_records_without_measured_cost": sum(record.get("input_cost_usd") is None for record in new_judge.values()),
        "global_jev_budget_snapshot_if_present": run_summary.get("jev_budget"),
        "ledger_issues": ledger_issues,
        "note": "Discard categories are subsets of charged work; do not add them again. Service sums exclude loading/orchestration and are not end-to-end wall-clock latency.",
    }
    cost["collection_total_generated_tokens"] = cost["collection_prefix_generated_tokens_counted_once"] + cost["collection_continuation_generated_tokens_all_arms"]
    if durable is not None:
        canonical = lambda record: json.dumps(record, sort_keys=True, separators=(",", ":"))
        known = Counter(canonical(g) for g in initials + [g for row in rows for g in row.get("calls", [])])
        measured = Counter(canonical(g) for g in durable.pop("completed_generations"))
        if known - measured:
            raise ValueError("Recorded checkpoint/outcome call is absent or different in the durable ledger")
        cost["durable_call_audit"] = durable
        cost["recorded_outcome_collection_generated_tokens"] = cost["collection_total_generated_tokens"]
        cost["completed_calls_outside_recorded_checkpoints_or_outcomes"] = sum((measured - known).values())
        cost["collection_total_generated_tokens"] = durable["generated_tokens"]
        cost["collection_prefix_generated_tokens_counted_once"] = durable["initial_generated_tokens"]
        cost["collection_continuation_generated_tokens_all_arms"] = durable["continuation_generated_tokens"]
        cost["collection_prompt_tokens_processed"] = durable["prompt_tokens_processed"]
        cost["collection_model_service_seconds"] = durable["model_service_seconds"]
        cost["totals_are_lower_bounds"] = bool(durable["failed_calls"] or durable["unmatched_started_sequences"])
    local_generations = [record["generation"] for record in effective_local.values() if isinstance(record.get("generation"), dict)]
    cost["local_judge_acquisition"] = {
        "records": len(effective_local), "records_with_measured_generation": len(local_generations),
        "outside_run_records_excluded": len(set(local_features) - set(checkpoints)),
        "failed_records": sum(bool(record.get("error")) for record in effective_local.values()),
        "records_with_unknown_generation_work": sum(bool(record.get("generated_work_unknown")) for record in effective_local.values()),
        "unmatched_started_problem_ids": local_audit["unmatched_started_problem_ids"],
        "totals_are_lower_bounds": bool(local_audit["unmatched_started_problem_ids"] or
                                       any(record.get("generated_work_unknown") for record in effective_local.values())),
        "generated_tokens": sum(nonnegative(record.get("generated_tokens")) for record in local_generations),
        "prompt_tokens_processed": sum(nonnegative(record.get("prompt_tokens")) for record in local_generations),
        "model_service_seconds": sum(nonnegative(record.get("elapsed_seconds")) for record in local_generations),
        "note": "Additional diagnostic acquisition; not included in the generator-only token envelope or earlier collection totals.",
    }
    feature_quality = []
    for pid, checkpoint in sorted(checkpoints.items()):
        _, numeric = observable_features(checkpoint, budget)
        jev = semantic_features(checkpoint.get("jev"))
        local = semantic_features(local_features.get(pid, checkpoint.get("local_judge")))
        feature_quality.append({"problem_id": pid, "jev_observed_features": int(np.isfinite(jev).sum()),
                                "local_observed_features": int(np.isfinite(local).sum()),
                                "retained_entropy_available": bool(np.isfinite(numeric[6])),
                                "retained_logprob_available": bool(np.isfinite(numeric[7])),
                                "discarded_future_tail_tokens": nonnegative(checkpoint.get("discarded_prefix_tail_tokens"))})
    complete_ids = {problem["problem_id"] for problem in problems}
    finished_enrollment = (len(checkpoints) + len(skipped) == manifest.get("problems") and not incomplete)
    mechanism_complete = (not manifest.get("protocol", "").startswith("mechanism-v1") or
                          (manifest.get("status") == run_summary.get("status") == "complete" and finished_enrollment))
    if learned and len(problems) >= 24 and not ledger_issues and mechanism_complete:
        modeling = crossfit(problems, [row for row in rows if row["problem_id"] in complete_ids], checkpoints, budget, indices, alpha, effective_local)
    else:
        modeling = {"status": "not_run", "reason": "Requires --learned, at least 24 complete independent problems, no unresolved generation-ledger issues, and a finished fully enrolled mechanism run; partial runs remain descriptive."}
    attempts = int(manifest.get("problems", len(checkpoints) + len(skipped)))
    if attempts < len(checkpoints) + len(skipped):
        raise ValueError("Recorded enrollment exceeds the manifest plan")
    source_names = ("manifest.json", "checkpoints.jsonl", "outcomes.jsonl", "skipped.json", "summary.json", "review_audit.json", "run_screen_source.py",
                    "schedule.json", "rubric.json", "generation_started.jsonl", "generation_events.jsonl", "checkpoint_attempts.jsonl",
                    "local_judge_provenance.json", "local_judge_status.json", "local_judge_started.jsonl",
                    "release_manifest.json", "source_manifest.json")
    sources = [{"path": str((run_dir / name).resolve()), "sha256": hashlib.sha256((run_dir / name).read_bytes()).hexdigest()}
               for name in source_names if (run_dir / name).exists()]
    return {
        "analysis_version": 2, "run_directory": str(run_dir.resolve()), "run_kind": manifest.get("kind"),
        "raw_data_sources": sources,
        "implementation": {"version": manifest.get("implementation_version"),
                           "source_snapshot_note": manifest.get("source_snapshot_note"),
                           "source_audit": source_audit, "local_judge_audit": local_audit,
                           "review_audit": review_audit},
        "classification": "descriptive_phase0_or_exploratory_screen_not_confirmatory",
        "enrollment": {"planned_attempted_problems": attempts, "recorded_problems": len(checkpoints) + len(skipped),
                       "finished_planned_enrollment": finished_enrollment, "mechanism_run_complete": mechanism_complete,
                       "not_yet_recorded_problems": attempts - len(checkpoints) - len(skipped), "eligible_checkpoints": len(checkpoints),
                       "skipped_problems": len(skipped), "skip_reasons": dict(Counter(row.get("reason", "unknown") for row in skipped)),
                       "complete_all_arm_problems": len(problems), "incomplete_checkpoints": incomplete,
                       "observed_episodes": len(rows), "expected_repeats_per_action": expected,
                       "complete_problem_families": dict(Counter(problem["family"] for problem in problems)),
                       "all_arm_enrollment_rule": "Contrasts and learned comparisons use complete scheduled arms only; incomplete cases and all incurred costs remain reported."},
        "per_problem": problems, "per_action_problem_weighted_success": per_action,
        "paired_action_differences": differences, "costs": cost, "crossfit": modeling,
        "feature_quality": {"semantic_keys": list(SEMANTIC_KEYS), "per_checkpoint": feature_quality,
                            "jev_complete_checkpoints": sum(row["jev_observed_features"] == len(SEMANTIC_KEYS) for row in feature_quality),
                            "local_complete_checkpoints": sum(row["local_observed_features"] == len(SEMANTIC_KEYS) for row in feature_quality),
                            "retained_entropy_available_checkpoints": sum(row["retained_entropy_available"] for row in feature_quality),
                            "local_records_outside_this_run": sorted(set(local_features) - set(checkpoints)),
                            "restriction": "Whole-generation entropy/logprob are excluded whenever the checkpoint discarded future tokens; explicit retained-only statistics must match the retained token count. Feature presence is not evidence of predictive value."},
        "bootstrap": {"draws": draws, "seed": seed, "unit": "original problem", "strata": "task family", "interval": "2.5/97.5 percentile resampling summary; descriptive, no multiplicity adjustment; overlapping crossfit training induces dependence, so fixed-contribution resampling has no generic nominal-95% coverage",
                      "singleton_strata": [family for family, count in Counter(problem["family"] for problem in problems).items() if count == 1]},
        "interpretation": [
            "Phase 0 (especially n=6) checks instrumentation and headroom; it is inadequate for superiority or generalization claims.",
            "Repeated continuations are clustered within problems and are not independent problem samples.",
            "Results condition on observed eligible checkpoints; early-completed and boundary-ineligible problems are not silently treated as intervention failures.",
            "All-arm data collection is not evidence that a Jev-guided sequential policy improves task outcomes.",
            "Stratified bootstrap conditions on the observed family mix and cannot establish held-out-family generalization.",
            "Pooled feature differences may reflect task-family composition; use the training-family constant control and family-stratified contrasts before attributing gains to semantic features.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--learned", action="store_true", help="Enable exploratory GroupKFold analysis only when >=24 complete problems")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--local-features", type=Path, help="JSONL: problem_id and response.answers/features/probabilities; defaults to run_dir/local_judge.jsonl if present")
    args = parser.parse_args()
    if args.bootstrap_draws < 100 or not math.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0:
        parser.error("Use at least 100 bootstrap draws and a positive finite ridge alpha")
    try:
        local = {}
        local_path = args.local_features or args.run_dir / "local_judge.jsonl"
        if args.local_features and not local_path.exists():
            raise ValueError("Specified local-feature file does not exist")
        if local_path.exists():
            for record in read_jsonl(local_path):
                pid = record["problem_id"]
                if pid in local:
                    raise ValueError("Duplicate local-feature problem_id")
                local[pid] = record
        result = analyze(args.run_dir, draws=args.bootstrap_draws, seed=args.seed,
                         learned=args.learned, alpha=args.ridge_alpha, local_features=local)
        if local_path.exists():
            result["raw_data_sources"].append({"path": str(local_path.resolve()), "sha256": hashlib.sha256(local_path.read_bytes()).hexdigest()})
        result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError) as exc:
        print(f"Analysis refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps({"output": str(args.output.resolve()), "complete_problems": result["enrollment"]["complete_all_arm_problems"],
                      "ledger_issues": len(result["costs"]["ledger_issues"]), "crossfit_status": result["crossfit"]["status"]}))


if __name__ == "__main__":
    main()
