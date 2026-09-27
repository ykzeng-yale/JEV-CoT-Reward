#!/usr/bin/env python3
"""Export existing diagnostic records without inference, API calls, or rescoring.

Only explicit local-experiment fields are released. Hosted Jev response objects,
machine paths, and credential/network adapters are excluded. Suspected secrets
or machine addresses fail the export before any destination file is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


MANIFEST_FIELDS = (
    "problems", "repeats", "budget", "prefix_tokens", "final_reserve",
    "quantize_bits", "seed", "kind", "schema", "started_unix", "selector",
    "note", "implementation_version", "implementation_sha256",
    "source_snapshot_note", "source_sha256", "package_versions",
)
CHECKPOINT_FIELDS = (
    "problem_id", "task", "prompt_ids", "retained_ids", "initial",
    "discarded_prefix_tail_tokens", "retained_features", "state", "sha256",
)
TASK_FIELDS = ("id", "family", "prompt", "data")
TASK_DATA_FIELDS = ("n", "edges", "numbers", "target")
STATE_FIELDS = ("task", "history", "latest_segment")
RETAINED_FEATURE_FIELDS = ("retained_stat_token_count", "retained_mean_logprob", "retained_mean_entropy")
LABEL_FIELDS = ("success", "reason", "path_cost", "value")
OVERHEAD_FIELDS = ("kind", "tokens", "removed_tokens", "winner", "selector")
LOCAL_PROBABILITY_FIELDS = ("hypothesis", "contradiction", "unsupported", "repeated_failure",
                            "localized_error", "testable", "locally_valid")
GENERATION_FIELDS = (
    "token_ids", "text", "prompt_tokens", "generated_tokens", "mean_logprob",
    "mean_entropy", "elapsed_seconds", "finish_reason", "peak_memory_gb",
    "prefix_sha256", "seed", "token_logprobs", "token_entropies",
)
OUTCOME_FIELDS = (
    "action", "seed", "outcome", "text", "generated_tokens",
    "prompt_tokens_processed", "elapsed_seconds", "calls", "overhead",
    "problem_id", "family", "checkpoint_sha256", "repeat",
    "shared_prefix_generated_tokens", "episode_id", "checkpoint_id",
)
SUMMARY_FIELDS = (
    "kind", "attempted_problems", "eligible_problems", "skipped_problems",
    "episodes", "actions", "continuation_generated_tokens",
    "continuation_seconds", "actual_shared_prefix_tokens", "interpretation",
)
LOCAL_JUDGE_FIELDS = (
    "problem_id", "schema", "quantization_bits", "kind", "probabilities",
    "error", "generation",
)
AUDIT_FIELDS = (
    "implementation_review", "episodes", "problems",
    "incomplete_final_reserve_affected", "generated_token_budget_violations",
    "exact_initial_prefix_hash_mismatches", "checkpoints_with_discarded_future_tail",
    "restriction",
)
SOURCE_FILES = ("__init__.py", "features.py", "mlx_backend.py", "tasks.py", "run_screen.py")
# Match values, never print matched content. Literal schema/rubric names are not
# credentials; the omitted Jev adapter is not needed for outcome verification.
UNSAFE_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"apikey_[a-z0-9]{16,}(?:_[a-z0-9]+)?",
    r"\bsk-[a-z0-9_-]{20,}",
    r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----",
    r"(?:/Users/|/home/|/private/|/Volumes/|/mnt/|/tmp/)[^\s\"']+",
    r"\b[A-Z]:\\(?:Users|Documents and Settings|Windows)\\",
    r"https?://[^\s\"']+",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    r"\b[a-z0-9][a-z0-9-]*\.local\b",
    r"\b(?:ssh|sftp)://",
    r"\b(?:authorization|api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"'][^\"']{12,}[\"']",
))


class UnsafeExportError(ValueError):
    """Exception payload contains only relative paths, never matched content."""


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keep(record: dict, fields: tuple[str, ...]) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Expected a record object")
    return {key: record[key] for key in fields if key in record}


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def jsonl_bytes(rows) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                   for row in rows).encode()


def public_task(record: dict) -> dict:
    task = keep(record, TASK_FIELDS)
    task["data"] = keep(record["data"], TASK_DATA_FIELDS)
    return task


def model_identity(value: str) -> dict:
    """Recover a model repository/revision, never retain a local cache path."""
    if not isinstance(value, str):
        raise ValueError("Expected a model identifier string")
    parts = value.replace("\\", "/").split("/")
    repository = None
    for part in parts:
        if part.startswith("models--"):
            decoded = part[len("models--"):].split("--")
            if len(decoded) == 2 and all(re.fullmatch(r"[\w.-]+", piece) for piece in decoded):
                repository = "/".join(decoded)
    if repository is None and re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        repository = value
    revision = None
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts) and re.fullmatch(r"[0-9a-f]{7,64}", parts[index + 1]):
            revision = parts[index + 1]
    return {"repository": repository, "revision": revision,
            "local_path_omitted": True}


def scan_public_files(files: dict[str, bytes]) -> list[str]:
    """Return relative filenames only for potential secrets/host information."""
    return sorted(name for name, content in files.items()
                  if any(pattern.search(content.decode("utf-8")) for pattern in UNSAFE_PATTERNS))


def verify_record_counts(checkpoints: list[dict], outcomes: list[dict], summary: dict) -> None:
    ids = [row["problem_id"] for row in checkpoints]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate checkpoint problem IDs")
    episodes = [(row["problem_id"], row["action"], row["repeat"]) for row in outcomes]
    if len(episodes) != len(set(episodes)):
        raise ValueError("Duplicate original episode identity tuple")
    if any(row["problem_id"] not in ids for row in outcomes):
        raise ValueError("Outcome references an unknown checkpoint")
    if summary.get("episodes") != len(outcomes) or summary.get("eligible_problems") != len(checkpoints):
        raise ValueError("Original summary count does not match records")
    # Compare stored booleans only. Never run a verifier or recompute an answer.
    for action, counts in summary.get("actions", {}).items():
        selected = [row for row in outcomes if row["action"] == action]
        if counts != {"successes": sum(row["outcome"]["success"] for row in selected), "n": len(selected)}:
            raise ValueError("Original summary action counts do not match recorded labels")


def source_files(run_dir: Path, manifest: dict, reference_run: Path | None) -> tuple[dict[str, bytes], dict]:
    files = {}
    provenance = []
    declared = manifest.get("source_sha256", {})
    for leaf in SOURCE_FILES:
        source = run_dir / "source" / leaf
        status = "captured_run_source"
        expected = declared.get(leaf)
        if leaf == "run_screen.py" and not source.exists() and (run_dir / "run_screen_source.py").exists():
            source = run_dir / "run_screen_source.py"
            status = "reconstructed_runner_as_declared_in_original_manifest"
            expected = manifest.get("implementation_sha256")
        if not source.exists():
            continue
        if source.is_symlink():
            raise ValueError(f"Refusing symlink source: {leaf}")
        content = source.read_bytes()
        actual = digest(content)
        if expected is not None and expected != actual:
            raise ValueError(f"Source hash mismatch: {leaf}")
        name = f"source/{leaf}"
        files[name] = content
        provenance.append({"released_path": name, "sha256": actual,
                           "original_declared_sha256": expected, "status": status,
                           "source_run_id": run_dir.name})
    # The earliest run did not save its helpers. Supply only the later captured
    # verifier as a transparently labeled reference, never fabricate provenance.
    if "source/tasks.py" not in files and reference_run is not None:
        reference = reference_run / "source" / "tasks.py"
        if reference.is_symlink():
            raise ValueError("Refusing symlink verification reference")
        reference_manifest = read_json(reference_run / "manifest.json")
        content = reference.read_bytes()
        expected = reference_manifest.get("source_sha256", {}).get("tasks.py")
        actual = digest(content)
        if expected is None or actual != expected:
            raise ValueError("Unverified later-run verification reference")
        name = "source/tasks_reference.py"
        files[name] = content
        provenance.append({"released_path": name, "sha256": actual,
                           "original_declared_sha256": expected,
                           "status": "later_run_verification_reference_not_attested_original_helper",
                           "source_run_id": reference_run.name})
    omitted = [{"original_filename": leaf, "original_declared_sha256": sha,
                "reason": "Credential/network adapter excluded from public outcome release"
                if leaf == "jev.py" else "Outside local-verification source allowlist"}
               for leaf, sha in declared.items() if leaf not in SOURCE_FILES]
    return files, {"files": provenance, "omitted_original_sources": omitted,
                   "original_runner_sha256": manifest.get("implementation_sha256"),
                   "original_snapshot_note": manifest.get("source_snapshot_note"),
                   "note": "Snapshots preserve historical bytes; omitted adapters are not needed to verify stored task outcomes. No source is represented as captured when it was reconstructed or borrowed as a later reference."}


def build_release(run_dir: Path, reference_run: Path | None = None) -> dict[str, bytes]:
    manifest = read_json(run_dir / "manifest.json")
    original_checkpoints = read_jsonl(run_dir / "checkpoints.jsonl")
    original_outcomes = read_jsonl(run_dir / "outcomes.jsonl")
    original_summary = read_json(run_dir / "summary.json")
    verify_record_counts(original_checkpoints, original_outcomes, original_summary)
    public_manifest = keep(manifest, MANIFEST_FIELDS)
    identity = model_identity(manifest.get("model", ""))
    public_manifest.update({"model": identity["repository"], "model_identity": identity,
                            "release_kind": "public_local_outputs_without_hosted_judge_responses",
                            "original_run_id": run_dir.name,
                            "hosted_judge_responses_omitted": True,
                            "recorded_labels_copied_without_rescoring": True})
    checkpoints = []
    for original in original_checkpoints:
        row = keep(original, CHECKPOINT_FIELDS)
        row["task"] = public_task(original["task"])
        row["initial"] = keep(original["initial"], GENERATION_FIELDS)
        row["state"] = keep(original["state"], STATE_FIELDS)
        if "retained_features" in original:
            row["retained_features"] = keep(original["retained_features"], RETAINED_FEATURE_FIELDS)
        checkpoints.append(row)
    outcomes = []
    for original in original_outcomes:
        row = keep(original, OUTCOME_FIELDS)
        row["outcome"] = keep(original["outcome"], LABEL_FIELDS)
        row["calls"] = [keep(call, GENERATION_FIELDS) for call in original.get("calls", [])]
        row["overhead"] = [keep(item, OVERHEAD_FIELDS) for item in original.get("overhead", [])]
        outcomes.append(row)
    files = {
        "manifest.json": json_bytes(public_manifest),
        "tasks.jsonl": jsonl_bytes(row["task"] for row in checkpoints),
        "checkpoints.jsonl": jsonl_bytes(checkpoints),
        "outcomes.jsonl": jsonl_bytes(outcomes),
        "summary.json": json_bytes(keep(original_summary, SUMMARY_FIELDS)),
    }
    local_file = run_dir / "local_judge.jsonl"
    if local_file.exists():
        local_rows = []
        for original in read_jsonl(local_file):
            row = keep(original, LOCAL_JUDGE_FIELDS)
            row["model_identity"] = model_identity(original.get("model", ""))
            row["generation"] = keep(original["generation"], GENERATION_FIELDS)
            row["probabilities"] = keep(original.get("probabilities") or {}, LOCAL_PROBABILITY_FIELDS)
            local_rows.append(row)
        files["local_judge.jsonl"] = jsonl_bytes(local_rows)
    audit = run_dir / "review_audit.json"
    if audit.exists():
        files["review_audit.json"] = json_bytes(keep(read_json(audit), AUDIT_FIELDS))
    skipped = run_dir / "skipped.json"
    if skipped.exists():
        skipped_rows = []
        for original in read_json(skipped):
            row = keep(original, ("reason",))
            row["task"] = public_task(original["task"])
            row["initial"] = keep(original["initial"], GENERATION_FIELDS)
            skipped_rows.append(row)
        files["skipped.json"] = json_bytes(skipped_rows)
    sources, source_manifest = source_files(run_dir, manifest, reference_run)
    files.update(sources)
    files["source_manifest.json"] = json_bytes(source_manifest)
    original_names = ("manifest.json", "checkpoints.jsonl", "outcomes.jsonl", "summary.json",
                      "local_judge.jsonl", "review_audit.json", "skipped.json")
    release_manifest = {
        "format_version": "public-local-diagnostics-v1",
        "original_run_id": run_dir.name,
        "checkpoint_records": len(checkpoints), "outcome_records": len(outcomes),
        "episode_identity": ["problem_id", "action", "repeat"],
        "original_input_sha256": {name: digest((run_dir / name).read_bytes())
                                  for name in original_names if (run_dir / name).exists()},
        "released_files_sha256": {name: digest(content) for name, content in sorted(files.items())},
        "omitted_data": ["All hosted Jev response, probability, feature, request, and usage objects",
                         "API key, credential/cache storage, endpoints and host/network configuration",
                         "Absolute filesystem paths and unrelated model/cache inventory",
                         "Historical credential/network adapter source (original hash retained)"],
        "reproducibility_limits": [
            "Original local outputs, token IDs, costs, source hashes and recorded outcome labels are preserved; labels were not regenerated or rescored.",
            "Jev feature ablations cannot be reproduced from this release; hosted output publication requires separate terms clarification.",
            "Independent local-judge records, when present, were generated without Jev outputs and are separately identified.",
            "Earlier reconstructed or reference source files retain explicit provenance limitations.",
            "These are development diagnostics, not independent held-out policy tests.",
        ],
    }
    files["release_manifest.json"] = json_bytes(release_manifest)
    unsafe = scan_public_files(files)
    if unsafe:
        raise UnsafeExportError("\n".join(unsafe))
    return files


def export_run(run_dir: Path, output_dir: Path, reference_run: Path | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError("Destination already exists; public exports are not overwritten")
    files = build_release(run_dir, reference_run)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    for name, content in sorted(files.items()):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = json.loads(files["release_manifest.json"])
    return {"run_id": manifest["original_run_id"], "files": len(files),
            "checkpoints": manifest["checkpoint_records"], "episodes": manifest["outcome_records"],
            "labels_rescored": False, "hosted_responses_released": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--verification-reference-run", type=Path)
    args = parser.parse_args()
    try:
        report = export_run(args.run_dir, args.output_dir, args.verification_reference_run)
    except UnsafeExportError as exc:
        # Print ONLY relative matching paths, never patterns or matched values.
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
