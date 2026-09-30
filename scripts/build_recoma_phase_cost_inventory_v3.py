#!/usr/bin/env python3
"""Build receipt-grounded final-phase costs without opening task outcomes.

Final inventory requires terminal parent sacct rows. A snapshot has a distinct
schema and cannot be mistaken for the analyzer's terminal inventory. Recovery
of a failed run's token counts remains cost-only and explicitly incomplete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
ROLES = {"production", "qualification", "failed_attempt", "cancelled_unallocated", "cpu_audit"}
CORE = ("job_id", "state", "exit_code", "elapsed_seconds", "gpu_count", "cpu_count", "account", "qos", "partition")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def count(text, name):
    require(isinstance(text, str) and re.fullmatch(r"\d+", text) is not None, "invalid sacct " + name)
    return int(text)


def tres(text):
    result = {}
    if not text:
        return result
    for item in text.split(","):
        require(item.count("=") == 1, "malformed AllocTRES")
        key, value = item.split("=")
        require(key not in result and key and value, "duplicate/empty AllocTRES field")
        result[key] = value
    return result


def memory_gib(text):
    if text is None:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", text)
    require(match is not None, "unrecognized Slurm memory unit")
    amount = float(match.group(1))
    # Slurm TRES memory without a suffix is reported in MiB.
    factor = {"K": 1 / 1024**2, "M": 1 / 1024, "": 1 / 1024, "G": 1, "T": 1024}[match.group(2)]
    return amount * factor


def parent_accounting(path, job_id):
    require(isinstance(job_id, str) and re.fullmatch(r"\d+", job_id), "exact numeric parent Slurm ID required")
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    require(rows and {"JobID", "State", "ExitCode", "ElapsedRaw", "AllocTRES", "Account", "QOS", "Partition"}.issubset(rows[0]),
            "sacct receipt needs explicit pipe-delimited headers")
    matches = [row for row in rows if row["JobID"] == job_id]
    require(len(matches) == 1, "missing or ambiguous exact parent Slurm row")
    row = matches[0]
    require(all(row.get(key) is not None for key in CORE if key in row), "truncated sacct parent row")
    state = row["State"].split(" ")[0].rstrip("+")
    require(re.fullmatch(r"\d+:\d+", row["ExitCode"]) is not None, "Slurm exit code missing")
    allocation = tres(row["AllocTRES"])
    cpu = count(allocation.get("cpu", "0"), "CPU count")
    typed = {key[len("gres/gpu:"):]: count(value, "typed GPU count") for key, value in allocation.items() if key.startswith("gres/gpu:")}
    generic = count(allocation["gres/gpu"], "GPU count") if "gres/gpu" in allocation else sum(typed.values())
    if typed:
        require(sum(typed.values()) == generic, "generic and typed GPU AllocTRES disagree")
    elapsed = count(row["ElapsedRaw"], "elapsed seconds")
    if cpu == generic == 0:
        require(elapsed == 0, "nonzero elapsed with no allocated CPU/GPU resources")
    require(row["Account"] == "pi_fl426" and row["QOS"] == "normal", "scheduler account/tier is outside the frozen phase")
    result = {"job_id": job_id, "state": state, "exit_code": row["ExitCode"], "elapsed_seconds": elapsed,
        "gpu_count": generic, "cpu_count": cpu, "gpu_type": next(iter(typed)) if len(typed) == 1 else None,
        "memory_gib": memory_gib(allocation.get("mem")), "account": row["Account"], "qos": row["QOS"],
        "requested_qos": "normal", "actual_qos": row["QOS"], "partition": row["Partition"]}
    for source, target in (("Submit", "submit_time"), ("Start", "start_time"), ("End", "end_time")):
        if source in row:
            result[target] = row[source]
    return result


def compare_resources(resources, parsed):
    for key in CORE:
        require(resources.get(key) == parsed[key], "normalized resources disagree with sacct parent: " + key)
    require(resources.get("requested_qos", "normal") == "normal", "non-normal requested tier")
    require(resources.get("actual_qos", parsed["qos"]) == parsed["qos"], "actual QoS differs from sacct")
    for key in ("gpu_type", "memory_gib"):
        if key in resources:
            require(resources[key] == parsed[key], "normalized resources disagree with AllocTRES: " + key)


def bound_files(paths):
    require(isinstance(paths, list), "evidence paths must be explicit lists")
    require(len(paths) == len(set(paths)), "duplicate evidence path")
    return {str(Path(path).resolve(strict=True)): sha256(path) for path in paths}


def generation(job, parsed):
    settings = job.get("generation_accounting", {})
    mode = settings.get("mode")
    scheduler_zero = parsed["state"] in {"CANCELLED", "DEADLINE"} and parsed["elapsed_seconds"] == 0 \
        and parsed["gpu_count"] == parsed["cpu_count"] == 0 \
        and parsed.get("start_time", "None") in {"None", "Unknown", ""}
    if job["role"] == "cancelled_unallocated" or (mode == "no_generation" and scheduler_zero):
        require(parsed["state"] in {"CANCELLED", "DEADLINE"} and parsed["elapsed_seconds"] == 0
                and parsed["gpu_count"] == parsed["cpu_count"] == 0
                and parsed.get("start_time", "None") in {"None", "Unknown", ""},
                "cancelled-unallocated job had a scheduler allocation/start")
        require(mode in (None, "no_generation"), "unallocated cancellation cannot contain model work")
        return {"mode": "no_generation", "generation_proved_not_started": True,
            "reason": "Exact terminal scheduler parent: " + parsed["state"] + " before allocation, zero ElapsedRaw/CPU/GPU and no Start.",
            "evidence_sha256": bound_files([job["accounting_file"]])}
    if mode == "audit":
        audit_path = Path(settings["audit_file"]).resolve(strict=True)
        manifest = Path(settings["manifest_file"]).resolve(strict=True)
        audit = load(audit_path)
        require(audit.get("audit_status") == "PASS", "cost audit has not passed")
        require(audit.get("manifest_sha256") == sha256(manifest), "cost audit/manifest identity mismatch")
        require(audit.get("auditor_sha256") == settings.get("expected_auditor_sha256"), "explicit approved cost auditor digest required")
        require(job["role"] in {"production", "qualification"}, "passing-audit cost role mismatch")
        require((job["role"] == "qualification" and audit.get("audit_scope") == "runtime_qualification")
                or (job["role"] == "production" and audit.get("audit_scope") in {"full_panel", "balanced_census"}),
                "cost audit scope differs from role")
        compare_resources(audit["slurm_accounting"], parsed)
        result = {"mode": "audit", "audit_file": str(audit_path), "audit_sha256": sha256(audit_path),
            "manifest_file": str(manifest), "expected_auditor_sha256": settings["expected_auditor_sha256"]}
        if settings.get("qualification_auditor_correction"):
            require(job["role"] == "qualification", "source-correction cost receipt is qualification-only")
            correction = settings["qualification_auditor_correction"]
            receipt_path = Path(correction["receipt_file"]).resolve(strict=True)
            require(sha256(receipt_path) == correction["receipt_sha256"], "qualification correction receipt hash mismatch")
            result["qualification_auditor_correction"] = {"receipt_file": str(receipt_path), "receipt_sha256": correction["receipt_sha256"]}
        return result
    if job["role"] == "cpu_audit":
        require(parsed["gpu_count"] == 0 and mode == "no_generation", "CPU audit requires zero GPUs and no generation")
        proof_path = Path(settings["execution_receipt_file"]).resolve(strict=True)
        proof = load(proof_path)
        require(str(proof.get("job_id")) == parsed["job_id"] and proof.get("status") in {"completed", "infrastructure_failure"}
                and proof.get("performed_generation") is False, "CPU execution receipt does not prove no generation")
        for field in ("model_generations", "hosted_calls", "jev_calls"):
            require(type(proof.get(field)) is int and proof[field] == 0, "CPU audit has missing/nonzero inference counter: " + field)
        reason = settings.get("reason")
        require(isinstance(reason, str) and reason, "CPU no-generation reason required")
        evidence = bound_files([str(proof_path), settings["batch_file"], settings["log_file"]]
                               + settings.get("additional_evidence_files", []))
        return {"mode": mode, "generation_proved_not_started": True, "reason": reason, "evidence_sha256": evidence}
    require(job["role"] == "failed_attempt", "unaudited terminal model job must be an explicit failed attempt")
    require(parsed["state"] != "COMPLETED", "a completed qualification requires its independent audit")
    if mode == "no_generation":
        report_path = Path(settings["runtime_report_file"]).resolve(strict=True)
        report = load(report_path)
        require(str(report.get("slurm_job_id", report.get("job_id"))) == parsed["job_id"], "no-generation runtime report job identity mismatch")
        require(report.get("performed_generation") is False and report.get("status") == "infrastructure_failure",
                "runtime report does not prove generation was never started")
        reason = settings.get("reason")
        require(isinstance(reason, str) and reason, "explicit no-generation failure reason required")
        evidence = [str(report_path)] + settings.get("additional_evidence_files", [])
        return {"mode": mode, "generation_proved_not_started": True, "reason": reason, "evidence_sha256": bound_files(evidence)}
    require(mode == "partial_logs", "failed generation requires no-generation proof or explicit partial logs")
    require(settings.get("all_attempts_recoverable") is False, "failed generation cost cannot assert unverified completeness")
    explanation = settings.get("failure_explanation")
    require(isinstance(explanation, str) and explanation, "partial failed-generation explanation missing")
    # Hash binary bytes only: this helper does not read task results or partial
    # success rates, and does not recompute scientific outcomes from failed runs.
    return {"mode": mode, "all_attempts_recoverable": False,
        "call_sha256": bound_files(settings.get("call_files", [])),
        "stress_sha256": bound_files(settings.get("stress_files", [])), "failure_explanation": explanation}


def build(spec_path, *, snapshot=False):
    spec_path = Path(spec_path).resolve(strict=True)
    spec = load(spec_path)
    require(spec.get("schema") == "recoma_phase_cost_spec_v3", "cost specification schema mismatch")
    expected = spec.get("expected_job_ids")
    require(isinstance(expected, list) and expected and len(expected) == len(set(expected))
            and all(isinstance(x, str) and re.fullmatch(r"\d+", x) for x in expected), "unique numeric phase parent job inventory required")
    entries = spec.get("jobs")
    require(isinstance(entries, list) and len(entries) == len(expected) and {j.get("job_id") for j in entries} == set(expected),
            "cost specification does not cover expected phase jobs exactly")
    result, open_jobs = [], []
    gpu_hours = cpu_hours = 0.0
    for job in entries:
        require(job.get("role") in ROLES, "invalid phase cost role")
        parsed = parent_accounting(job["accounting_file"], job["job_id"])
        terminal = parsed["state"] in TERMINAL
        if not terminal:
            require(snapshot and parsed["state"] in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"},
                    "final costs require a terminal exact parent row; no live snapshot extrapolation")
            open_jobs.append(job["job_id"])
        receipts = bound_files([job["accounting_file"]] + job.get("additional_accounting_files", []))
        resources = parsed
        if job.get("resources_file"):
            resources = load(job["resources_file"])
            compare_resources(resources, parsed)
            receipts.update(bound_files([job["resources_file"]]))
        if terminal and job.get("generation_accounting", {}).get("mode") == "audit":
            resources = load(job["generation_accounting"]["audit_file"])["slurm_accounting"]
            compare_resources(resources, parsed)
        gen = generation(job, parsed) if terminal else {"mode": "unresolved_live_job", "not_zero_generation": True}
        result.append({"job_id": job["job_id"], "role": job["role"], "resources": resources,
            "accounting_receipt_sha256": receipts, "generation_accounting": gen})
        gpu_hours += parsed["elapsed_seconds"] * parsed["gpu_count"] / 3600
        cpu_hours += parsed["elapsed_seconds"] * parsed["cpu_count"] / 3600
    return {"schema": "recoma_phase_cost_inventory_snapshot_v3" if snapshot else "recoma_phase_cost_inventory_v3",
        "scope": spec.get("scope", "Explicit receipt-enumerated final-phase parent Slurm jobs"),
        "expected_job_ids": expected, "jobs": result, "open_job_ids": open_jobs,
        "terminal_inventory": not snapshot, "accounting_scope_boundary": "Parent records only; no batch/extern/step double counting. Snapshot elapsed is an as-of lower bound, never a final forecast.",
        "receipt_accounted_gpu_hours": gpu_hours, "receipt_accounted_reserved_cpu_hours": cpu_hours,
        "spec_sha256": sha256(spec_path), "builder_sha256": sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", action="store_true", help="Distinct non-final schema; the analysis gate will reject it")
    args = parser.parse_args()
    require(not args.output.exists(), "refuse to overwrite a prior phase inventory")
    result = build(args.spec, snapshot=args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("schema", "expected_job_ids", "open_job_ids", "receipt_accounted_gpu_hours")}, sort_keys=True))


if __name__ == "__main__":
    main()
