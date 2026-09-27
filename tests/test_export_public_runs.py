"""Outcome release preserves labels and drops private/hosted fields by design."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "public_export", Path(__file__).resolve().parents[1] / "scripts" / "export_public_runs.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture_run(tmp_path, name="diagnostic"):
    run = tmp_path / name
    run.mkdir()
    source = run / "source"
    source.mkdir()
    script = b'"""Recorded local experiment source."""\nVALUE = 1\n'
    verifier = b'"""Recorded verifier source; exporter never executes it."""\nVALUE = 2\n'
    (source / "run_screen.py").write_bytes(script)
    (source / "tasks.py").write_bytes(verifier)
    (source / "jev.py").write_text("# This credential/network adapter must not be exported.\n")
    generation = {"token_ids": [11, 12], "text": "Recorded original explanation.",
                  "prompt_tokens": 10, "generated_tokens": 2, "mean_logprob": -0.4,
                  "mean_entropy": 0.8, "elapsed_seconds": 0.25, "finish_reason": "length",
                  "peak_memory_gb": 3.8, "prefix_sha256": "a" * 64, "seed": 123,
                  "token_logprobs": [-0.2, -0.6], "token_entropies": [0.7, 0.9]}
    task = {"id": "task-1", "family": "arithmetic_construction", "prompt": "Use 2 and 3 to get 5.",
            "data": {"numbers": [2, 3], "target": 5}}
    path = "/Users/researcher/cache/models--Org--Model/snapshots/" + "c" * 40
    manifest = {"model": path, "output": "/Users/researcher/private-run", "problems": 1,
                "repeats": 1, "budget": 32, "prefix_tokens": 2, "final_reserve": 4,
                "quantize_bits": 4, "seed": 123, "jev": True, "kind": "diagnostic",
                "schema": "semantic-v1", "selector": "mean_base_logprob",
                "implementation_sha256": exporter.digest(script),
                "source_sha256": {p.name: exporter.digest(p.read_bytes()) for p in source.iterdir()}}
    checkpoint = {"problem_id": "task-1", "task": task, "prompt_ids": [1, 2, 3],
                  "retained_ids": [11, 12], "initial": generation,
                  "discarded_prefix_tail_tokens": 0,
                  "retained_features": {"retained_stat_token_count": 2, "retained_mean_logprob": -0.4,
                                        "retained_mean_entropy": 0.8},
                  "state": {"task": task["prompt"], "history": "", "latest_segment": generation["text"]},
                  "sha256": "b" * 64,
                  "jev": {"response": {"answers": {"private_hosted_feature": 0.917}},
                          "request": "PRIVATE_HOSTED_RESPONSE_MARKER"}}
    rows = [{"problem_id": "task-1", "family": task["family"], "action": action, "repeat": 0,
             "episode_id": f"original-{action}", "seed": 123 + i,
             "outcome": {"success": action == "continue", "reason": "original_label"},
             "text": "FINAL: 2+3", "generated_tokens": 2, "prompt_tokens_processed": 10,
             "elapsed_seconds": 0.25, "calls": [generation], "overhead": [],
             "checkpoint_sha256": "b" * 64, "shared_prefix_generated_tokens": 2}
            for i, action in enumerate(("continue", "repair", "branch"))]
    summary = {"kind": "diagnostic", "attempted_problems": 1, "eligible_problems": 1,
               "skipped_problems": 0, "episodes": 3,
               "actions": {row["action"]: {"n": 1, "successes": int(row["outcome"]["success"])} for row in rows},
               "continuation_generated_tokens": 6, "continuation_seconds": 0.75,
               "actual_shared_prefix_tokens": 2,
               "jev_budget": {"private_usage": "OMITTED_USAGE_MARKER"}}
    local = {"problem_id": "task-1", "model": path, "schema": "semantic-v1",
             "quantization_bits": 4, "kind": "independent_local_judge",
             "probabilities": {"hypothesis": 0.2}, "error": None, "generation": generation}
    for name, data in [("manifest.json", manifest), ("summary.json", summary)]:
        (run / name).write_bytes(exporter.json_bytes(data))
    for name, records in [("checkpoints.jsonl", [checkpoint]), ("outcomes.jsonl", rows),
                          ("local_judge.jsonl", [local])]:
        (run / name).write_bytes(exporter.jsonl_bytes(records))
    return run, checkpoint, rows, manifest


def test_preserves_original_outputs_ids_costs_and_labels(tmp_path):
    run, checkpoint, rows, manifest = fixture_run(tmp_path)
    original_hashes = {str(p.relative_to(run)): exporter.digest(p.read_bytes()) for p in run.rglob("*") if p.is_file()}
    destination = tmp_path / "release"
    report = exporter.export_run(run, destination)
    exported = exporter.read_jsonl(destination / "outcomes.jsonl")
    assert exported == rows  # No labels were recomputed, including deliberately inconsistent ones.
    public_checkpoint = exporter.read_jsonl(destination / "checkpoints.jsonl")[0]
    assert public_checkpoint == {key: value for key, value in checkpoint.items() if key != "jev"}
    assert report["labels_rescored"] is False
    assert exporter.read_json(destination / "manifest.json")["implementation_sha256"] == manifest["implementation_sha256"]
    assert original_hashes == {str(p.relative_to(run)): exporter.digest(p.read_bytes()) for p in run.rglob("*") if p.is_file()}


def test_omits_hosted_outputs_paths_and_adapter_but_keeps_local_judge(tmp_path):
    run, _, _, manifest = fixture_run(tmp_path)
    files = exporter.build_release(run)
    combined = b"\n".join(files.values()).decode()
    assert "PRIVATE_HOSTED_RESPONSE_MARKER" not in combined
    assert "private_hosted_feature" not in combined
    assert "OMITTED_USAGE_MARKER" not in combined
    assert "/Users/" not in combined
    assert "source/jev.py" not in files
    public_manifest = json.loads(files["manifest.json"])
    assert public_manifest["model"] == "Org/Model"
    assert public_manifest["model_identity"]["revision"] == "c" * 40
    assert "output" not in public_manifest
    assert "local_judge.jsonl" in files
    assert json.loads(files["local_judge.jsonl"])["probabilities"] == {"hypothesis": 0.2}
    provenance = json.loads(files["source_manifest.json"])
    assert provenance["omitted_original_sources"][0]["original_declared_sha256"] == manifest["source_sha256"]["jev.py"]


def test_nested_unknown_hosted_fields_are_not_released(tmp_path):
    run, checkpoint, _, _ = fixture_run(tmp_path)
    for container in [checkpoint["state"], checkpoint["initial"], checkpoint["retained_features"], checkpoint["task"]["data"]]:
        container["jev_response"] = "NESTED_HOSTED_MARKER"
    (run / "checkpoints.jsonl").write_bytes(exporter.jsonl_bytes([checkpoint]))
    files = exporter.build_release(run)
    assert b"NESTED_HOSTED_MARKER" not in b"\n".join(files.values())


def test_export_fails_before_writing_and_reports_only_relative_path(tmp_path):
    run, _, rows, _ = fixture_run(tmp_path)
    secret = "apikey_" + "a" * 32 + "_" + "b" * 64
    rows[0]["text"] = secret
    (run / "outcomes.jsonl").write_bytes(exporter.jsonl_bytes(rows))
    destination = tmp_path / "release"
    with pytest.raises(exporter.UnsafeExportError) as error:
        exporter.export_run(run, destination)
    assert str(error.value) == "outcomes.jsonl"
    assert secret not in str(error.value)
    assert not destination.exists()


def test_rejects_stale_source_hash(tmp_path):
    run, _, _, _ = fixture_run(tmp_path)
    (run / "source" / "run_screen.py").write_text("CHANGED = True\n")
    with pytest.raises(ValueError, match="Source hash mismatch: run_screen.py"):
        exporter.build_release(run)


def test_later_verifier_reference_is_not_claimed_as_original(tmp_path):
    run, _, _, manifest = fixture_run(tmp_path, "earlier")
    later, _, _, _ = fixture_run(tmp_path, "later")
    (run / "source" / "tasks.py").unlink()
    manifest["source_sha256"].pop("tasks.py")
    (run / "manifest.json").write_bytes(exporter.json_bytes(manifest))
    files = exporter.build_release(run, later)
    assert "source/tasks.py" not in files
    assert "source/tasks_reference.py" in files
    entry = next(row for row in json.loads(files["source_manifest.json"])["files"]
                 if row["released_path"] == "source/tasks_reference.py")
    assert entry["status"] == "later_run_verification_reference_not_attested_original_helper"
    assert entry["source_run_id"] == "later"


def test_release_hash_manifest_and_no_overwrite(tmp_path):
    run, _, _, _ = fixture_run(tmp_path)
    destination = tmp_path / "release"
    exporter.export_run(run, destination)
    release = exporter.read_json(destination / "release_manifest.json")
    for name, sha in release["released_files_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == sha
    with pytest.raises(FileExistsError):
        exporter.export_run(run, destination)


def test_duplicate_episode_identity_is_rejected(tmp_path):
    run, checkpoint, rows, _ = fixture_run(tmp_path)
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="Duplicate original episode identity"):
        exporter.verify_record_counts([checkpoint], rows, exporter.read_json(run / "summary.json"))


def test_network_and_machine_path_scan_emits_filenames_only():
    files = {"clean.txt": b"public text", "endpoint.txt": b"https://private.example.org",
             "host.txt": b"10.2.3.4", "path.txt": b"/Users/private/person/data"}
    assert exporter.scan_public_files(files) == ["endpoint.txt", "host.txt", "path.txt"]


def fixture_development(tmp_path):
    run, old_checkpoint, old_rows, original_manifest = fixture_run(tmp_path, "development-fixture")
    (run / "checkpoints.jsonl").unlink()
    (run / "local_judge.jsonl").unlink()
    for original_path in exporter.DEVELOPMENT_SOURCE_PATHS:
        path = run / "source" / Path(original_path).name
        if not path.exists():
            path.write_text(f'"""Captured fixture source for {Path(original_path).name}."""\n')
    sources = {name: exporter.digest((run / "source" / Path(name).name).read_bytes())
               for name in exporter.DEVELOPMENT_SOURCE_PATHS}
    sources["src/jev_control/jev.py"] = exporter.digest((run / "source" / "jev.py").read_bytes())
    manifest = {"model": original_manifest["model"], "output": original_manifest["output"],
                "protocol": exporter.DEVELOPMENT_PROTOCOL, "problems": 2, "repeats": 2,
                "task_seed": 111, "sampling_seed": 222, "budget": 1024,
                "checkpoint_target": 256, "checkpoint_cap": 384, "final_reserve": 96,
                "prompt_suffix": "Use concise intermediate calculations.",
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                "temperature": 0.7, "top_p": 0.9, "source_sha256": sources,
                "packages": {"mlx-lm": "0.31.3"}, "finalization": "Recorded common final reserve.",
                "resume_rng": "Fresh recorded seed."}
    tasks, rows = [], []
    for index in range(2):
        task = copy.deepcopy(old_checkpoint["task"])
        task["id"] = f"dev-task-{index}"
        tasks.append({"task": task, "prompt_ids": [1, 2, 3]})
        for repeat in range(2):
            for condition in ("baseline", "sham"):
                # Store deliberately chosen labels and costs. The exporter
                # copies these values and must never call a verifier or runner.
                row = {"problem_id": task["id"], "family": task["family"],
                       "condition": condition, "seed": 222 + index * 100 + repeat,
                       "text": old_rows[0]["text"], "outcome": {"success": repeat == 0, "reason": "recorded"},
                       "calls": copy.deepcopy(old_rows[0]["calls"]), "generated_tokens": 2,
                       "elapsed_seconds": 0.25, "prompt_tokens_processed": 10, "overhead": [],
                       "checkpoint": None if condition == "baseline" else {
                           "token_ids": [11, 12], "text": "2 + 3 = 5\n", "prefix_sha256": "d" * 64,
                           "position": 2, "calculation_marker": True},
                       "checkpoint_status": "not_requested" if condition == "baseline" else "eligible",
                       "repeat": repeat, "episode_id": f"original-dev-{index}-{condition}-{repeat}"}
                rows.append(row)
    summary = {"kind": "development_framing_and_online_sham_not_jev_effect", "episodes": 8,
               "independent_problems": 2,
               "conditions": {condition: {"n": 4, "successes": 2, "generated_tokens": 8,
                                           "elapsed_seconds": 1.0,
                                           "checkpoint_status": {"not_requested" if condition == "baseline" else "eligible": 4},
                                           "calculation_marker_checkpoints": 0 if condition == "baseline" else 4,
                                           "families": {"arithmetic_construction": {"n": 4, "successes": 2}}}
                              for condition in ("baseline", "sham")},
               "warning": "Development only."}
    for name, data in [("manifest.json", manifest), ("summary.json", summary)]:
        (run / name).write_bytes(exporter.json_bytes(data))
    for name, records in [("tasks.jsonl", tasks), ("outcomes.jsonl", rows)]:
        (run / name).write_bytes(exporter.jsonl_bytes(records))
    return run, tasks, rows, manifest


def test_development_mode_preserves_tasks_calls_checkpoints_labels_and_costs(tmp_path):
    run, tasks, rows, manifest = fixture_development(tmp_path)
    original = {str(p.relative_to(run)): exporter.digest(p.read_bytes()) for p in run.rglob("*") if p.is_file()}
    destination = tmp_path / "dev-release"
    exporter.export_run(run, destination, mode="development-v1")
    assert exporter.read_jsonl(destination / "tasks.jsonl") == tasks
    assert exporter.read_jsonl(destination / "outcomes.jsonl") == rows
    assert not (destination / "checkpoints.jsonl").exists()  # Checkpoints retain original per-episode context.
    assert exporter.read_json(destination / "manifest.json")["source_sha256"] == manifest["source_sha256"]
    assert original == {str(p.relative_to(run)): exporter.digest(p.read_bytes()) for p in run.rglob("*") if p.is_file()}


def test_development_auto_route_and_explicit_no_version_mixing(tmp_path):
    run, _, _, _ = fixture_development(tmp_path)
    files = exporter.build_release(run)
    assert json.loads(files["release_manifest.json"])["format_version"] == "public-local-development-v1"
    with pytest.raises(ValueError, match="All-arm export mode"):
        exporter.build_release(run, mode="all-arm")
    with pytest.raises(ValueError, match="cannot borrow"):
        exporter.build_release(run, reference_run=run)
    ordinary, _, _, _ = fixture_run(tmp_path, "old-run")
    with pytest.raises(ValueError, match="requires its recorded protocol"):
        exporter.build_release(ordinary, mode="development-v1")


def test_development_requires_completion_marker_and_full_identity_grid(tmp_path):
    run, _, rows, _ = fixture_development(tmp_path)
    summary = (run / "summary.json").read_bytes()
    (run / "summary.json").unlink()
    with pytest.raises(ValueError, match="no completion summary"):
        exporter.build_release(run)
    (run / "summary.json").write_bytes(summary)
    (run / "outcomes.jsonl").write_bytes(exporter.jsonl_bytes(rows[:-1]))
    with pytest.raises(ValueError, match="identity grid is incomplete"):
        exporter.build_release(run)


def test_development_source_hashes_use_flat_captured_files_and_original_keys(tmp_path):
    run, _, _, manifest = fixture_development(tmp_path)
    files = exporter.build_release(run)
    source = json.loads(files["source_manifest.json"])
    backend = next(item for item in source["files"] if item["original_repository_path"] == "src/jev_control/mlx_backend.py")
    assert backend["original_declared_sha256"] == manifest["source_sha256"]["src/jev_control/mlx_backend.py"]
    assert files["source/mlx_backend.py"] == (run / "source" / "mlx_backend.py").read_bytes()
    assert "source/jev.py" not in files
    (run / "source" / "mlx_backend.py").write_text("LATER_VERSION = True\n")
    with pytest.raises(ValueError, match="Development source hash mismatch: mlx_backend.py"):
        exporter.build_release(run)


def test_development_rejects_all_arm_rows_and_unknown_protocol(tmp_path):
    run, _, rows, manifest = fixture_development(tmp_path)
    rows[0]["action"] = "continue"
    (run / "outcomes.jsonl").write_bytes(exporter.jsonl_bytes(rows))
    with pytest.raises(ValueError, match="all-arm/version mixing"):
        exporter.build_release(run)
    manifest["protocol"] = "development-v2-unknown"
    (run / "manifest.json").write_bytes(exporter.json_bytes(manifest))
    with pytest.raises(ValueError, match="Unsupported recorded protocol"):
        exporter.build_release(run)
