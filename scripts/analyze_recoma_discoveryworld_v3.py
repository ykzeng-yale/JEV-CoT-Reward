#!/usr/bin/env python3
"""Complete-panel ReCoMA description after an independently passing v3 audit.

No inference, tokenizer loading, API calls, partial-panel rates, treatment
effects or population confidence intervals. All paths are hash-bound; remote
receipt paths may be relocated only by an explicit prefix map.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

SCHEMA = "recoma_complete_baseline_analysis_v3"
FAILURES = {"missing_action_json", "invalid_action_json", "non_object_action_json", "invalid_submit_arguments"}
DIFFICULTIES = ("Easy", "Normal", "Challenge")
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, "invalid " + name)
    return float(value)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "invalid " + name)
    return value


def close(actual, expected, name):
    require(math.isclose(number(actual, name), expected, rel_tol=1e-10, abs_tol=1e-10), name + " arithmetic mismatch")


def prediction_matches_official(value, official_progress):
    require(type(value) in (str, int, float), "prediction must be a numeric scalar or its JSON string serialization")
    try:
        predicted = float(value)
    except (TypeError, ValueError):
        raise ValueError("prediction is not a numeric official progress scalar") from None
    require(math.isfinite(predicted) and math.isclose(predicted, official_progress, rel_tol=1e-12, abs_tol=1e-12),
            "prediction does not match independently recomputed official progress")


class Resolver:
    """Exact prefix substitution, never ambiguous basename searching."""
    def __init__(self, path_map=None):
        self.path_map = {str(Path(k)): str(Path(v)) for k, v in (path_map or {}).items()}
        require(all(Path(k).is_absolute() and Path(v).is_absolute() for k, v in self.path_map.items()),
                "path-map prefixes must be absolute")

    def __call__(self, path):
        value = str(Path(path))
        matches = [p for p in self.path_map if value == p or value.startswith(p + "/")]
        if matches:
            prefix = max(matches, key=len)
            value = self.path_map[prefix] + value[len(prefix):]
        return Path(value)

    def verify(self, path, digest):
        local = self(path)
        require(isinstance(digest, str) and len(digest) == 64, "invalid frozen file digest")
        require(local.is_file() and sha256(local) == digest, "hash-bound file changed or missing: " + str(path))
        return local


def load_json(path):
    result = json.loads(Path(path).read_text())
    require(isinstance(result, dict), "JSON object required")
    return result


def verify_audit(audit_path, manifest_path, expected_auditor_sha256, resolver, source_root=None,
                 qualification_auditor_correction=None):
    audit = load_json(audit_path)
    require(audit.get("audit_status") == "PASS", "independent PASS audit required before analysis")
    require(audit.get("auditor_sha256") == expected_auditor_sha256, "unapproved independent auditor digest")
    require(audit.get("manifest_sha256") == sha256(manifest_path), "audit/manifest hash mismatch")
    manifest = load_json(manifest_path)
    require(audit.get("audit_scope") == manifest.get("audit_scope"), "audit/manifest scope mismatch")
    if audit["audit_scope"] == "runtime_qualification":
        require(manifest.get("qualification_only") is True and manifest.get("seeds") == [5]
                and audit.get("n_tasks") == len(manifest.get("task_instances", [])) == 2,
                "qualification audit must retain two runtime-only seed-5 tasks")
    pinned = manifest.get("artifact_sha256", {}).get("audit_recoma_discoveryworld_v3.py")
    if qualification_auditor_correction is not None:
        require(audit["audit_scope"] == "runtime_qualification", "auditor source correction is qualification-only")
        correction_path = resolver.verify(qualification_auditor_correction["receipt_file"],
                                          qualification_auditor_correction["receipt_sha256"])
        correction = load_json(correction_path)
        scope = correction.get("correction_scope")
        scope_ok = (isinstance(scope, str) and scope in {"numeric_predicted_serialization",
                    "numeric_predicted_and_prompt_key_order_serialization"}) or \
            (isinstance(scope, list) and len(scope) == 2
             and all(isinstance(item, str) for item in scope)
             and set(scope) == {"numeric_scalar_json_serialization", "prompt_message_key_order_serialization"})
        require(correction.get("schema") == "recoma_qualification_auditor_correction_v3"
                and correction.get("original_manifest_sha256") == audit["manifest_sha256"]
                and correction.get("original_auditor_sha256") == pinned
                and correction.get("effective_auditor_sha256") == expected_auditor_sha256
                and scope_ok,
                "qualification correction provenance does not bind original/effective source and manifest")
        for field in ("generation_rerun", "task_assignment_changed", "inference_source_changed",
                      "raw_inputs_modified", "scientific_endpoint_changed"):
            require(correction.get(field) is False, "qualification correction changed frozen science: " + field)
        require(isinstance(correction.get("reason"), str) and correction["reason"], "source-proven correction reason required")
    elif pinned is not None:
        require(pinned == expected_auditor_sha256, "auditor differs from prospectively frozen manifest")
    hashes = audit.get("input_sha256")
    require(isinstance(hashes, dict) and set(hashes) == {"all_data", "task_records", "task_events", "calls", "context_stress"},
            "complete independent input-hash inventory required")
    for kind, files in hashes.items():
        require(isinstance(files, dict) and (files or kind == "context_stress"), "empty audit input group: " + kind)
        for path, digest in files.items():
            resolver.verify(path, digest)
    runtime_hashes = audit.get("runtime_report_sha256", {})
    require(isinstance(runtime_hashes, dict), "invalid runtime receipt hashes")
    for path, digest in runtime_hashes.items():
        resolver.verify(path, digest)
    inventory = audit.get("runtime_source_inventory")
    if manifest.get("source_manifest_sha256") is not None:
        require(isinstance(inventory, dict) and inventory.get("sha256") == manifest["source_manifest_sha256"],
                "original runtime source inventory binding missing or changed")
        resolver.verify(inventory["path"], inventory["sha256"])
    content = audit.get("model_content_verification")
    if manifest.get("model_content_contract") is not None:
        require(isinstance(content, dict), "production model-content receipt verification missing")
    if content is not None:
        roles = {"checksums", "verifier", "before", "after"}
        require(content.get("status") == "PASS_PRE_POST_FROZEN_MODEL_CONTENT"
                and set(content.get("input_paths", {})) == set(content.get("input_sha256", {})) == roles,
                "model-content hashes need exact original artifact paths")
        for role in roles:
            resolver.verify(content["input_paths"][role], content["input_sha256"][role])
    if source_root is not None:
        root = resolver(source_root)
        for relative, digest in audit.get("source_sha256", {}).items():
            path = root.parent / relative if relative.startswith("recoma/") else root / relative
            resolver.verify(path, digest)
        require(audit.get("source_sha256"), "source hash inventory required")
    return manifest, audit


def hash_bound_rows(audit, kind, resolver):
    rows = []
    for path in audit["input_sha256"][kind]:
        for line in resolver(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), "invalid durable row")
                rows.append(row)
    return rows


def task_identity(row):
    scenario = row.get("scenario_name", row.get("scenario"))
    difficulty = row.get("difficulty")
    seed = row.get("random_seed", row.get("seed"))
    require(isinstance(scenario, str) and scenario and difficulty in DIFFICULTIES, "invalid frozen task identity")
    integer(seed, "task seed")
    task = f"{scenario}_{difficulty}_{seed}"
    require(row.get("task_id", task) == task, "task ID does not match scenario/difficulty/seed")
    return task, scenario, difficulty, seed


def indexed(rows):
    result = {}
    for row in rows:
        task, *_ = task_identity(row)
        require(task not in result, "duplicate assigned task")
        result[task] = row
    return result


def describe(rows):
    require(rows, "no rows to summarize")
    n = len(rows)
    return {
        "assigned_tasks": n, "official_progress_mean": statistics.mean(r["official_progress"] for r in rows),
        "official_completed_count": sum(r["official_completed"] for r in rows),
        "official_success_count": sum(r["official_success"] for r in rows),
        "official_success_rate": sum(r["official_success"] for r in rows) / n,
        "failure_adjusted_success_count": sum(r["failure_adjusted_success"] for r in rows),
        "failure_adjusted_success_rate": sum(r["failure_adjusted_success"] for r in rows) / n,
        "scientific_failure_count": sum(r["scientific_failure"] is not None for r in rows),
        "scientific_failure_codes": dict(sorted(Counter(r["scientific_failure"] for r in rows if r["scientific_failure"] is not None).items())),
        "environment_action_cap_reached_count": sum(r["environment_action_cap_reached"] for r in rows),
        "model_call_cap_reached_count": sum(r["model_call_cap_reached"] for r in rows),
        "uncompleted_without_format_failure_at_cap_count": sum(r["uncompleted_without_format_failure_at_cap"] for r in rows),
        "environment_actions": sum(r["environment_actions"] for r in rows),
        "model_calls": sum(r["model_calls"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "generated_tokens": sum(r["generated_tokens"] for r in rows),
        "generated_tokens_in_final_format_failures": sum(r["final_format_failure_tokens"] for r in rows),
        "model_service_seconds": sum(r["model_service_seconds"] for r in rows),
        "task_wall_seconds_sum": sum(r["task_wall_seconds"] for r in rows),
    }


def quantile(values, fraction):
    ordered = sorted(values)
    offset = (len(ordered) - 1) * fraction
    low, high = math.floor(offset), math.ceil(offset)
    return ordered[low] + (offset - low) * (ordered[high] - ordered[low])


def theme_reweighting(theme_summaries, draws):
    """Sensitivity to reweighting eight observed families, not sampling inference."""
    if not draws:
        return None
    require(type(draws) is int and 1000 <= draws <= 100000, "descriptive draws outside frozen envelope")
    rng = random.Random(20260930)
    names = sorted(theme_summaries)
    require(len(names) == 8, "eight theme families required for descriptive reweighting")
    metrics = ("official_progress_mean", "official_success_rate", "failure_adjusted_success_rate")
    samples = {metric: [] for metric in metrics}
    for _ in range(draws):
        selected = rng.choices(names, k=8)
        for metric in metrics:
            samples[metric].append(statistics.mean(theme_summaries[name][metric] for name in selected))
    return {
        "unit": "eight observed scenario families; all difficulties/seeds remain together",
        "draws": draws, "rng_seed": 20260930,
        "middle_95_percent_reweighting_ranges": {key: [quantile(value, .025), quantile(value, .975)] for key, value in samples.items()},
        "population_confidence_interval": False, "causal_effect": False,
        "interpretation": "Descriptive sensitivity to reweighting the observed eight families; not uncertainty for the observed finite census, unseen tasks, new templates, or published-model replication.",
    }


def allocation(resources):
    state = resources.get("state")
    require(state in TERMINAL, "phase costs require terminal parent Slurm accounting")
    require(isinstance(resources.get("exit_code"), str), "terminal Slurm exit code missing")
    require(resources.get("account") == "pi_fl426", "phase account differs from project PI")
    require(resources.get("requested_qos", resources.get("qos")) == "normal", "phase request is not normal tier")
    elapsed = number(resources.get("elapsed_seconds"), "Slurm elapsed seconds")
    gpu = integer(resources.get("gpu_count"), "allocated GPU count")
    cpu = integer(resources.get("cpu_count"), "allocated CPU count")
    if gpu == 0 and cpu == 0:
        require(elapsed == 0, "unallocated job must have zero allocation time")
    return elapsed * gpu / 3600, elapsed * cpu / 3600


def phase_costs(inventory_path, production_audit_path, production_audit, expected_auditor_sha256, resolver):
    if inventory_path is None:
        return {"inventory_supplied": False, "allocation_inventory_complete": False, "generation_accounting_complete": False,
                "scope": "Production allocation only; qualification/failures/cancellations not inventoried",
                "production_actual_gpu_hours": production_audit["actual_gpu_hours_from_slurm"],
                "production_actual_reserved_cpu_hours": production_audit["actual_reserved_cpu_hours_from_slurm"]}
    inventory = load_json(inventory_path)
    require(inventory.get("schema") == "recoma_phase_cost_inventory_v3", "phase cost schema mismatch")
    expected = inventory.get("expected_job_ids")
    require(isinstance(expected, list) and expected and len(expected) == len(set(expected)) and all(isinstance(x, str) for x in expected),
            "explicit unique phase job inventory required")
    jobs = inventory.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == len(expected), "phase job coverage missing")
    require({x.get("job_id") for x in jobs} == set(expected), "phase job identities differ from expected inventory")
    seen, totals, summaries, incomplete = set(), Counter(), [], []
    production_count = 0
    for job in jobs:
        identity = job["job_id"]
        require(identity not in seen and "." not in identity and "_" not in identity, "duplicate or non-parent Slurm identity")
        seen.add(identity)
        receipts = job.get("accounting_receipt_sha256")
        require(isinstance(receipts, dict) and receipts, "parent Slurm accounting receipt hashes required")
        for path, digest in receipts.items():
            resolver.verify(path, digest)
        resources = job.get("resources")
        require(isinstance(resources, dict) and str(resources.get("job_id")) == identity, "cost parent identity mismatch")
        gpu_hours, cpu_hours = allocation(resources)
        generation = job.get("generation_accounting", {})
        mode = generation.get("mode")
        costs = {"prompt_tokens": 0, "generated_tokens": 0, "model_calls": 0, "model_service_seconds": 0.0,
                 "context_stress_prompt_tokens": 0, "context_stress_generated_tokens": 0, "context_stress_model_service_seconds": 0.0}
        generation_diagnostics = {"failed_stress_attempted_input_tokens_not_certified_processed": 0}
        complete = True
        role = job.get("role")
        require(role in {"production", "qualification", "failed_attempt", "cancelled_unallocated", "cpu_audit"}, "invalid phase cost role")
        if mode == "audit":
            audit_file = resolver.verify(generation["audit_file"], generation["audit_sha256"])
            manifest_file = resolver(generation["manifest_file"])
            _, audited = verify_audit(audit_file, manifest_file,
                generation.get("expected_auditor_sha256", expected_auditor_sha256), resolver,
                qualification_auditor_correction=generation.get("qualification_auditor_correction"))
            require(audited.get("slurm_accounting") == resources, "audit/cost Slurm accounting mismatch")
            if role == "production":
                production_count += 1
                require(sha256(audit_file) == sha256(production_audit_path), "phase production audit differs from analyzed production")
            else:
                require(role == "qualification" and audited["audit_scope"] == "runtime_qualification", "nonproduction scientific audits cannot be pooled")
            costs.update(prompt_tokens=audited["all_prompt_tokens_including_runtime_stress"],
                generated_tokens=audited["all_generated_tokens_including_failed_final_calls_and_runtime_stress"],
                model_calls=audited["model_calls"] + (1 if audited["context_stress_prompt_tokens"] else 0),
                model_service_seconds=audited["model_service_seconds"],
                context_stress_prompt_tokens=audited["context_stress_prompt_tokens"],
                context_stress_generated_tokens=audited["context_stress_completion_tokens"],
                context_stress_model_service_seconds=audited["context_stress_model_service_seconds"])
            close(audited["actual_gpu_hours_from_slurm"], gpu_hours, "actual audited GPU hours")
            close(audited["actual_reserved_cpu_hours_from_slurm"], cpu_hours, "actual audited CPU hours")
        elif mode == "no_generation":
            require(generation.get("generation_proved_not_started") is True and isinstance(generation.get("reason"), str)
                    and generation["reason"], "zero generation must be explicitly evidenced")
            for path, digest in generation.get("evidence_sha256", {}).items():
                resolver.verify(path, digest)
            require(generation.get("evidence_sha256"), "zero-generation evidence missing")
        elif mode == "partial_logs":
            require(role == "failed_attempt", "partial logs are cost-only failed attempts")
            require(generation.get("all_attempts_recoverable") is False, "failed generation cannot assert unverified complete cost")
            complete = False
            for path, digest in generation.get("call_sha256", {}).items():
                local = resolver.verify(path, digest)
                for line in local.read_text().splitlines():
                    if not line.strip():
                        continue
                    call = json.loads(line)
                    costs["model_calls"] += 1
                    costs["model_service_seconds"] += number(call.get("elapsed_seconds", 0), "partial call service seconds")
                    if call.get("status") == "ok":
                        costs["prompt_tokens"] += integer(call.get("input_tokens"), "partial call prefill")
                        costs["generated_tokens"] += integer(call.get("output_tokens"), "partial call completion")
            for path, digest in generation.get("stress_sha256", {}).items():
                stress = load_json(resolver.verify(path, digest))
                costs["model_calls"] += integer(stress.get("model_calls"), "partial stress calls")
                attempted_prompt = integer(stress.get("processed_prompt_tokens", 0), "partial stress attempted prefill")
                output = integer(stress.get("generated_tokens", 0), "partial stress completion")
                if stress.get("status") == "completed" or output > 0:
                    require(isinstance(stress.get("output_token_ids"), list) and len(stress["output_token_ids"]) == output,
                            "recovered stress output token IDs required")
                    prompt = attempted_prompt
                else:
                    # The helper writes processed_prompt_tokens immediately
                    # before model.generate. An OOM may occur before prefill;
                    # those submitted tokens are not certified processed work.
                    prompt = 0
                    generation_diagnostics["failed_stress_attempted_input_tokens_not_certified_processed"] += attempted_prompt
                seconds = number(stress.get("generation_elapsed_seconds", 0), "partial stress service seconds")
                costs["prompt_tokens"] += prompt; costs["generated_tokens"] += output
                costs["model_service_seconds"] += seconds
                costs["context_stress_prompt_tokens"] += prompt
                costs["context_stress_generated_tokens"] += output
                costs["context_stress_model_service_seconds"] += seconds
            require(isinstance(generation.get("failure_explanation"), str) and generation["failure_explanation"], "unknown failed-generation cost explanation required")
            incomplete.append(identity)
        else:
            raise ValueError("cost generation mode must be audit, no_generation or partial_logs")
        require(role != "production" or mode == "audit", "production cost must bind its passing audit")
        if role == "cpu_audit":
            require(resources["gpu_count"] == 0 and mode == "no_generation", "CPU audit must have zero GPUs and evidenced no generation")
        if role == "cancelled_unallocated":
            require(resources["state"] in {"CANCELLED", "DEADLINE"} and gpu_hours == 0 and cpu_hours == 0 and mode == "no_generation", "unallocated terminal identity/cost mismatch")
        totals["actual_gpu_hours"] += gpu_hours; totals["actual_reserved_cpu_hours"] += cpu_hours
        for key, value in costs.items():
            totals[key] += value
        summaries.append({"job_id": identity, "role": role, "state": resources["state"], "exit_code": resources["exit_code"],
            "actual_gpu_hours": gpu_hours, "actual_reserved_cpu_hours": cpu_hours, "generation_accounting_complete": complete,
            "generation_costs_accounted_lower_bounds": costs, "generation_diagnostics": generation_diagnostics,
            "qualification_auditor_correction": generation.get("qualification_auditor_correction"),
            "accounting_receipt_sha256": receipts})
    require(production_count == 1, "exactly one analyzed production job required")
    return {"inventory_supplied": True, "inventory_scope": inventory.get("scope"), "expected_job_ids": expected,
        "allocation_inventory_complete": True, "allocation_inventory_boundary": "Complete for the explicitly receipt-enumerated phase jobs; not a proof of all historical project/Mac compute or monetary cost.",
        "generation_accounting_complete": not incomplete, "unknown_generation_cost_job_ids": incomplete,
        "actual_gpu_hours": totals.pop("actual_gpu_hours"), "actual_reserved_cpu_hours": totals.pop("actual_reserved_cpu_hours"),
        "generation_costs_accounted_lower_bounds": dict(totals), "generation_costs_exact_if_complete": dict(totals) if not incomplete else None,
        "jobs": summaries, "inventory_sha256": sha256(inventory_path), "jev_and_hosted_calls": "No new Jev/hosted calls authorized; this inventory does not attest to any unlogged external activity."}


def analyze(manifest_path, audit_path, expected_auditor_sha256, *, source_root, path_map=None,
            phase_cost_inventory_path=None, descriptive_draws=20000):
    resolver = Resolver(path_map)
    manifest_path, audit_path = resolver(manifest_path), resolver(audit_path)
    manifest, audit = verify_audit(audit_path, manifest_path, expected_auditor_sha256, resolver, source_root)
    scope = manifest["audit_scope"]
    require(scope in {"full_panel", "balanced_census"}, "qualification/incomplete outputs have no production efficacy analysis")
    require(manifest.get("artifact_sha256", {}).get("analyze_recoma_discoveryworld_v3.py") == sha256(Path(__file__)),
            "analysis script must match its prospectively frozen production manifest")
    require(manifest.get("no_jev_calls") is True, "explicit no-Jev production contract required")
    seeds = manifest.get("seeds")
    require((scope == "full_panel" and seeds == [0, 1, 2, 3, 4]) or
            (scope == "balanced_census" and seeds in ([0], [0, 1]) and manifest.get("timing_based_fallback") is True
             and manifest.get("selection_without_outcome_rates") is True), "prospectively frozen panel tier required")
    expected = indexed(manifest["task_instances"])
    records = indexed(hash_bound_rows(audit, "task_records", resolver))
    outcomes = indexed(hash_bound_rows(audit, "all_data", resolver))
    require(set(records) == set(outcomes) == set(expected) and len(records) == 24 * len(seeds) == audit["n_tasks"],
            "complete assigned denominator required; no survivor analysis")
    grouped_calls = defaultdict(list)
    for call in hash_bound_rows(audit, "calls", resolver):
        require(call.get("task_id") in expected and call.get("status") == "ok", "call identity/status differs from passing audit")
        grouped_calls[call["task_id"]].append(call)
    require(set(grouped_calls) == set(expected), "calls do not cover all assigned tasks")
    env_cap = integer(manifest["max_environment_actions_per_episode"], "environment cap", 1)
    model_cap = integer(manifest["max_llm_calls_per_episode"], "model cap", 1)
    by_stratum, by_theme, by_difficulty, by_seed = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    summaries = []
    prefix = f"hf_torch.{manifest['model']}"
    safe_rows = audit["per_task_audited_summary"]
    safe_audit = {row["task_id"]: row for row in safe_rows}
    require(len(safe_rows) == len(safe_audit) == len(expected) and set(safe_audit) == set(expected), "audited per-task summary denominator mismatch")
    for task in sorted(expected):
        record = records[task]
        _, theme, difficulty, seed = task_identity(record)
        meta = record["metadata"]
        require(meta == outcomes[task]["metadata"], "durable/outcome metadata differs")
        card = meta["final_scorecard"]
        require(isinstance(card, list) and len(card) == 1, "official scorecard missing")
        card = card[0]
        raw_score, maximum = number(card["score"], "raw score"), number(card["maxScore"], "max score", 1e-300)
        require(raw_score <= maximum, "official score exceeds max")
        progress = raw_score / maximum
        close(card["scoreNormalized"], progress, "official normalized progress")
        prediction_matches_official(record["predicted"], progress)
        prediction_matches_official(outcomes[task]["predicted"], progress)
        require(type(card["completed"]) is bool and type(card["completedSuccessfully"]) is bool, "Boolean official endpoint required")
        failure = meta.get("scientific_task_failure")
        code = None if failure is None else failure.get("code")
        require(code is None or code in FAILURES, "undeclared scientific failure")
        require(record["scientific_task_status"] == ("completed" if code is None else "scientific_failure"), "scientific failure status mismatch")
        adjusted = card["completedSuccessfully"] and code is None
        require(record["failure_adjusted_completed_successfully"] == adjusted, "failure-adjusted endpoint mismatch")
        calls = grouped_calls[task]
        actions = integer(meta["num_steps"], "environment actions")
        require(actions <= env_cap and 0 < len(calls) <= model_cap and seed in seeds, "task cap/seed violation")
        prompt = sum(integer(c["input_tokens"], "prompt tokens") for c in calls)
        generated = sum(integer(c["output_tokens"], "generated tokens") for c in calls)
        service = sum(number(c["elapsed_seconds"], "model service time") for c in calls)
        require(meta[prefix + ".calls"] == len(calls) and meta[prefix + ".prompt_tokens"] == prompt
                and meta[prefix + ".completion_tokens"] == generated, "terminal/call cost counters mismatch")
        row = {"task_id": task, "scenario": theme, "difficulty": difficulty, "seed": seed,
            "official_raw_score": raw_score, "official_max_score": maximum, "official_progress": progress,
            "official_completed": card["completed"], "official_success": card["completedSuccessfully"],
            "failure_adjusted_success": adjusted, "scientific_failure": code,
            "environment_actions": actions, "model_calls": len(calls), "prompt_tokens": prompt,
            "generated_tokens": generated, "model_service_seconds": service,
            "task_wall_seconds": number(record["task_wall_seconds"], "task wall time"),
            "final_format_failure_tokens": calls[-1]["output_tokens"] if code else 0,
            "environment_action_cap_reached": actions == env_cap, "model_call_cap_reached": len(calls) == model_cap,
            "uncompleted_without_format_failure_at_cap": not card["completed"] and code is None and (actions == env_cap or len(calls) == model_cap)}
        accepted = safe_audit[task]
        close(accepted["official_progress_score"], progress, "audited task progress")
        require(accepted["official_completed_successfully"] == row["official_success"]
                and accepted["failure_adjusted_completed_successfully"] == adjusted
                and accepted["environment_actions"] == actions and accepted["model_calls"] == len(calls), "audited task endpoint/cost summary differs")
        summaries.append(row)
        by_stratum[(theme, difficulty)].append(row); by_theme[theme].append(row)
        by_difficulty[difficulty].append(row); by_seed[seed].append(row)
    require(len(by_stratum) == 24 and len(by_theme) == 8 and set(by_difficulty) == set(DIFFICULTIES)
            and all(sorted(r["seed"] for r in rows) == seeds for rows in by_stratum.values()), "balanced complete 24-stratum census required")
    overall = describe(summaries)
    strata = [{"scenario": t, "difficulty": d, **describe(rows)} for (t, d), rows in sorted(by_stratum.items())]
    themes = {t: describe(rows) for t, rows in sorted(by_theme.items())}
    for output_key, audit_key in (("official_progress_mean", "primary_official_progress_score"),
            ("official_success_rate", "official_completed_successfully_rate"),
            ("failure_adjusted_success_rate", "failure_adjusted_completed_successfully_rate")):
        equal_stratum_mean = statistics.mean(row[output_key] for row in strata)
        close(overall[output_key], equal_stratum_mean, "balanced/equal-stratum " + output_key)
        close(audit[audit_key]["mean"], equal_stratum_mean, "independent-audit " + output_key)
    for key, audit_key in (("scientific_failure_count", "scientific_failure_count"), ("model_calls", "model_calls"),
                          ("prompt_tokens", "prompt_tokens"), ("generated_tokens", "completion_tokens"),
                          ("model_service_seconds", "episode_model_service_seconds")):
        close(audit[audit_key], overall[key], "aggregate " + key)
    require(overall["scientific_failure_codes"] == audit["scientific_failure_codes"], "failure code census mismatch")
    close(audit["mean_environment_actions"], overall["environment_actions"] / len(summaries), "environment action mean")
    gpu_hours, cpu_hours = allocation(audit["slurm_accounting"])
    close(audit["actual_gpu_hours_from_slurm"], gpu_hours, "production allocation GPU hours")
    close(audit["actual_reserved_cpu_hours_from_slurm"], cpu_hours, "production allocation CPU hours")
    return {"schema": SCHEMA, "analysis_status": "COMPLETE_AUDITED_PANEL", "audit_scope": scope,
        "official_five_seed_panel": scope == "full_panel", "assigned_denominator": len(summaries), "strata_count": 24,
        "theme_family_count": 8, "frozen_seeds": seeds, "model": manifest["model"], "model_revision": manifest["model_revision"],
        "aggregation": "Mean within each scenario/difficulty stratum, then equal mean of all 24 strata; balanced rows give the same whole-panel mean.",
        "primary_raw_official_progress": overall["official_progress_mean"], "whole_panel": overall,
        "by_stratum": strata, "by_theme": themes, "by_difficulty": {key: describe(rows) for key, rows in by_difficulty.items()},
        "by_seed": {str(key): describe(rows) for key, rows in sorted(by_seed.items())}, "per_task": summaries,
        "descriptive_theme_reweighting": theme_reweighting(themes, descriptive_draws),
        "budget_boundary": "Cap-reached indicators are directly observed counters. Uncompleted/no-format-failure at a cap is an operational exhaustion diagnostic; durable records do not identify an exclusive termination cause.",
        "scientific_failure_boundary": "Only four frozen format failures are typed; valid but environment-denied actions are not reclassified. All failure rows and their partial progress/call costs remain in the assigned denominator.",
        "interpretation_boundary": "One-arm, finite benchmark adaptation: no paired treatment effect, Jev increment, generalization/transfer claim, population CI, independent world-state replay or reproduction of published larger-model performance. Seed-specific scores are dependent variations of the same template family.",
        "intervention_count": 0, "new_jev_calls_authorized": 0, "production_context_stress_pooled": False,
        "production_allocation": {"slurm_accounting": audit["slurm_accounting"], "actual_gpu_hours": gpu_hours,
            "actual_reserved_cpu_hours": cpu_hours, "service_seconds_are_summed_across_calls": True,
            "task_wall_seconds_are_summed_across_tasks_not_user_latency": True},
        "audited_trace_format_diagnostics": audit.get("trace_format_diagnostics"),
        "audited_model_content_verification": audit.get("model_content_verification"),
        "audited_prompt_serialization_contract": audit.get("prompt_serialization_contract"),
        "trace_boundary": "Thought/action-field frequency is a visible-format diagnostic, not hidden chain-of-thought access, action usefulness, intervention eligibility or causal reasoning benefit.",
        "phase_costs": phase_costs(resolver(phase_cost_inventory_path) if phase_cost_inventory_path else None, audit_path, audit, expected_auditor_sha256, resolver),
        "provenance": {"manifest_sha256": sha256(manifest_path), "independent_audit_sha256": sha256(audit_path),
            "expected_auditor_sha256": expected_auditor_sha256, "analysis_script_sha256": sha256(Path(__file__)),
            "all_input_hashes_reverified": True, "source_hashes_reverified": True, "explicit_path_map": resolver.path_map,
            "model_content_artifact_hashes_reverified": audit.get("model_content_verification") is not None,
            "runtime_source_inventory_reverified": audit.get("runtime_source_inventory") is not None,
            "independent_input_sha256": audit["input_sha256"], "source_sha256": audit["source_sha256"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "audit", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--path-map", type=Path, help="JSON object of exact absolute remote-prefix to local-prefix mappings")
    parser.add_argument("--phase-cost-inventory", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refuse to overwrite a prior analysis")
    mapping = load_json(args.path_map) if args.path_map else None
    result = analyze(args.manifest, args.audit, args.expected_auditor_sha256, source_root=args.source_root,
                     path_map=mapping, phase_cost_inventory_path=args.phase_cost_inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("analysis_status", "audit_scope", "assigned_denominator", "primary_raw_official_progress")}, sort_keys=True))


if __name__ == "__main__":
    main()
