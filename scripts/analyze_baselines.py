#!/usr/bin/env python3
"""Exploratory paper-inspired baselines on audited, complete all-arm records."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_control.baselines import fit_action_forest, fit_sparse_pair_gate
import analyze_screen as screen


def compare(report, checkpoints, rows, local, budget, *, draws=2000, seed=20260927):
    if report["crossfit"]["status"] != "exploratory_grouped_crossfit":
        raise ValueError("A complete audited screen with at least 24 problems is required")
    problems = report["per_problem"]
    ids = [problem["problem_id"] for problem in problems]
    lookup = {pid: i for i, pid in enumerate(ids)}
    docs, numeric = zip(*(screen.observable_features(checkpoints[pid], budget) for pid in ids))
    numeric = np.asarray(numeric)
    representations = {"cheap_tfidf": numeric}
    for name, features in (("jev", [checkpoints[pid].get("jev") for pid in ids]),
                           ("local", [local.get(pid, checkpoints[pid].get("local_judge")) for pid in ids])):
        values = np.vstack([screen.semantic_features(record) for record in features])
        if np.isfinite(values).any():
            representations[f"cheap_tfidf_plus_{name}"] = np.hstack([numeric, values])
    methods = ("forest_mean", "forest_dispersion", "sparse_repair_gate", "sparse_branch_gate")
    selected = {f"{method}:{name}": np.full(len(ids), -1, dtype=int)
                for method in methods for name in representations}
    selected["entropy_median_gate:cheap_tfidf"] = np.full(len(ids), -1, dtype=int)
    for name in representations:
        if name != "cheap_tfidf":
            selected[f"validity_07_gate:{name}"] = np.full(len(ids), -1, dtype=int)
    diagnostics = []
    for fold in report["crossfit"]["folds"]:
        train_ids, test_ids = fold["train_problems"], fold["test_problems"]
        if set(train_ids) & set(test_ids):
            raise ValueError("Train/test problem leakage")
        train, test = [lookup[p] for p in train_ids], [lookup[p] for p in test_ids]
        locations = {pid: i for i, pid in enumerate(train_ids)}
        entropy = numeric[train, 6]
        threshold = float(np.median(entropy[np.isfinite(entropy)])) if np.isfinite(entropy).any() else np.inf
        selected["entropy_median_gate:cheap_tfidf"][test] = np.where(numeric[test, 6] > threshold, 2, 0)
        train_rows = [row for row in rows if row["problem_id"] in locations]
        pairs = {pid: {} for pid in train_ids}
        for row in train_rows:
            pairs[row["problem_id"]][row["action"], row["repeat"]] = int(row["outcome"]["success"])
        for name, features in representations.items():
            if name != "cheap_tfidf":
                validity = features[test, -1]  # Last frozen semantic key: locally_valid.
                selected[f"validity_07_gate:{name}"][test] = np.where(np.isfinite(validity) & (validity < .7), 1, 0)
            x_train, x_test, vocabulary = screen.representation_matrices(
                [docs[i] for i in train], [docs[i] for i in test], features[train], features[test])
            forest = fit_action_forest(
                x_train[[locations[row["problem_id"]] for row in train_rows]],
                [screen.ACTIONS.index(row["action"]) for row in train_rows],
                [int(row["outcome"]["success"]) for row in train_rows],
                [row["problem_id"] for row in train_rows], seed=seed + fold["fold"])
            predictions = forest.predict(x_test)
            selected[f"forest_mean:{name}"][test] = predictions["success_probability"].argmax(axis=1)
            selected[f"forest_dispersion:{name}"][test] = predictions["selection_score"].argmax(axis=1)
            pair_indices, pair_groups, base_y, alternative_y = [], [], [], {"repair": [], "branch": []}
            for pid in train_ids:
                repeats = sorted(repeat for action, repeat in pairs[pid] if action == "continue")
                for repeat in repeats:
                    pair_indices.append(locations[pid]); pair_groups.append(pid)
                    base_y.append(pairs[pid]["continue", repeat])
                    for action in alternative_y:
                        alternative_y[action].append(pairs[pid][action, repeat])
            gate_counts = {}
            for action in ("repair", "branch"):
                gate = fit_sparse_pair_gate(x_train[pair_indices], base_y, alternative_y[action], pair_groups,
                                            seed=seed + fold["fold"])
                probability = gate.predict(x_test)
                selected[f"sparse_{action}_gate:{name}"][test] = np.where(probability > .5, screen.ACTIONS.index(action), 0)
                gate_counts[action] = {"pairs": gate.total_pairs, "discordant_pairs": gate.discordant_pairs}
            diagnostics.append({"fold": fold["fold"], "representation": name, "training_vocabulary": vocabulary,
                                "gate_training_counts": gate_counts})
    if any((choice < 0).any() for choice in selected.values()):
        raise ValueError("Fold predictions do not cover every problem")
    # Original Ridge/family controls use exactly the same outer folds.
    for row in report["crossfit"]["per_problem"]:
        for name, action in row["selected_actions"].items():
            key = f"reference:{name}"
            selected.setdefault(key, np.full(len(ids), -1, dtype=int))[lookup[row["problem_id"]]] = screen.ACTIONS.index(action)
    rates = np.asarray([[p["actions"][a]["success_rate"] for a in screen.ACTIONS] for p in problems])
    values = {name: rates[np.arange(len(ids)), choices] for name, choices in selected.items()}
    families = [p["family"] for p in problems]
    bootstrap = screen.stratified_bootstrap_indices(families, draws, seed)
    policies = {}
    for name, choices in selected.items():
        representation = name.split(":", 1)[1]
        reference_costs = report["crossfit"]["policies"].get(representation, {}).get("costs", {})
        costs = {k: v for k, v in reference_costs.items() if not k.startswith("deployment_")}
        for metric in ("deployment_generated_tokens", "deployment_prompt_tokens_processed", "deployment_service_seconds"):
            costs[metric] = float(np.mean([p["actions"][screen.ACTIONS[a]][metric] for p, a in zip(problems, choices)]))
        policies[name] = {"success": screen.estimate(values[name], bootstrap),
                          "action_counts": dict(Counter(screen.ACTIONS[a] for a in choices)), "costs": costs}
    contrasts = {}
    for method in methods:
        for name in representations:
            left = f"{method}:{name}"
            for right in (f"{method}:cheap_tfidf", "reference:training_selected_family_constant"):
                if left != right:
                    contrasts[f"{left}_minus_{right}"] = screen.estimate(values[left] - values[right], bootstrap)
        if "cheap_tfidf_plus_jev" in representations and "cheap_tfidf_plus_local" in representations:
            left, right = f"{method}:cheap_tfidf_plus_jev", f"{method}:cheap_tfidf_plus_local"
            contrasts[f"{left}_minus_{right}"] = screen.estimate(values[left] - values[right], bootstrap)
        for name in representations:
            if name != "cheap_tfidf":
                left, right = f"{method}:{name}", f"validity_07_gate:{name}"
                contrasts[f"{left}_minus_{right}"] = screen.estimate(values[left] - values[right], bootstrap)
    randomization = {}
    for name, choices in selected.items():
        expected_value = np.empty(len(ids))
        for fold in report["crossfit"]["folds"]:
            test = [lookup[pid] for pid in fold["test_problems"]]
            for family in sorted(set(families)):
                members = [i for i in test if families[i] == family]
                if not members:
                    continue
                frequencies = np.bincount(choices[members], minlength=3) / len(members)
                expected_value[members] = rates[members] @ frequencies
        randomization[name] = {"expected_success": screen.estimate(expected_value, bootstrap),
                               "selected_minus_randomized": screen.estimate(values[name] - expected_value, bootstrap)}
    family_means = {family: {name: float(value[np.asarray(families) == family].mean()) for name, value in values.items()}
                    for family in sorted(set(families))}
    return {"status": "exploratory_adaptations_not_published_method_reproductions",
            "n_problems": len(ids), "policies": policies, "paired_differences": contrasts,
            "family_policy_means": family_means, "fold_diagnostics": diagnostics,
            "within_fold_family_rate_matched_diagnostic": randomization,
            "per_problem": [{"problem_id": pid, "selected_actions": {name: screen.ACTIONS[a[i]] for name, a in selected.items()}}
                            for i, pid in enumerate(ids)],
            "fixed_settings": {"forest_trees": 200, "forest_depth": 6, "forest_min_leaf": 2,
                               "forest_dispersion_beta": .5, "sparse_gate_C": 1., "sparse_gate_threshold": .5,
                               "validity_repair_threshold": .7, "entropy_branch_threshold": "training-fold median; missing -> continue",
                               "ties": "zero utility loss; omitted from sparse fit, weights based on all original pairs"},
            "limitations": report["crossfit"]["limitations"] + [
                "These are paper-inspired adaptations with different tasks/actions/features, not reproductions.",
                "Tree dispersion is not a validated confidence interval, especially with clustered repeated inputs.",
                "Sparse gates estimate conditional win probability among discordant pairs, not success probability or magnitude of advantage.",
                "Each sparse gate compares one fixed alternative with continue; it cannot select the other alternative.",
                "Rate-matched diagnostic averages random permutations of chosen actions within the same held-out fold and family. It is a replay diagnostic, not a fresh deployment run; never permute across folds because that can introduce outcome leakage.",
                "Many exploratory comparisons are reported without selecting a winner or making multiplicity-adjusted superiority claims."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--integrity-report", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new analysis artifact")
    integrity_digest = screen.validate_integrity_report(args.run, args.integrity_report)
    local = {}
    for row in screen.read_jsonl(args.run / "local_judge.jsonl"):
        if row["problem_id"] in local:
            raise ValueError("Duplicate local-feature problem")
        local[row["problem_id"]] = row
    report = screen.analyze(args.run, learned=True, local_features=local)
    checkpoints = {row["problem_id"]: row for row in screen.read_jsonl(args.run / "checkpoints.jsonl")}
    rows = screen.read_jsonl(args.run / "outcomes.jsonl")
    budget = screen.read_json(args.run / "manifest.json")["budget"]
    result = compare(report, checkpoints, rows, local, budget)
    result["independent_integrity_report_sha256"] = integrity_digest
    result["input_provenance"] = report["raw_data_sources"]
    source_paths = [Path(__file__), Path(screen.__file__),
                    Path(__file__).resolve().parents[1] / "src/jev_control/baselines.py",
                    Path(__file__).resolve().parents[1] / "src/jev_control/representation.py"]
    result["analysis_sources_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    local_path = args.run / "local_judge.jsonl"
    if local_path.exists():
        result["local_feature_file_sha256"] = hashlib.sha256(local_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "problems": result["n_problems"]}))


if __name__ == "__main__":
    main()
