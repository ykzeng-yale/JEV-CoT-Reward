"""Adversarial finite-panel analysis checks; no model, GPU or partial outcomes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("analysis", Path(__file__).parents[1] / "scripts/analyze_recoma_discoveryworld_v3.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
AUDITOR = "a" * 64


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(root, seeds=(0, 1, 2, 3, 4), *, job="25001", qualification=False):
    root.mkdir(parents=True)
    source = root / "source/discoveryworld"
    source.mkdir(parents=True)
    (source / "endpoint.py").write_text("# independently audited frozen source\n")
    inventory = [{"scenario_name": f"Theme{theme}", "difficulty": difficulty, "random_seed": seed}
                 for theme in range(8) for difficulty in analysis.DIFFICULTIES for seed in seeds]
    if qualification:
        inventory = inventory[:2]
    scope = "runtime_qualification" if qualification else ("full_panel" if len(seeds) == 5 else "balanced_census")
    manifest = {"model": MODEL, "model_revision": REVISION, "audit_scope": scope,
                "seeds": list(seeds), "task_instances": inventory, "no_jev_calls": True,
                "max_environment_actions_per_episode": 30, "max_llm_calls_per_episode": 62,
                "timing_based_fallback": len(seeds) != 5, "selection_without_outcome_rates": True,
                "artifact_sha256": {"audit_recoma_discoveryworld_v3.py": AUDITOR,
                                    "analyze_recoma_discoveryworld_v3.py": analysis.sha256(Path(analysis.__file__))},
                "qualification_only": qualification}
    records, calls, outcomes, events, safe = [], [], [], [], []
    prefix = f"hf_torch.{MODEL}"
    for i, identity in enumerate(inventory):
        task, *_ = analysis.task_identity(identity)
        score = 1 if i == 0 else (i % 4) / 4
        success = i == 0
        failure = "missing_action_json" if i in (0, 1) else None
        actions = 30 if i == 2 else 2
        metadata = {"num_steps": actions, prefix + ".calls": 1, prefix + ".prompt_tokens": 100,
                    prefix + ".completion_tokens": 10,
                    "final_scorecard": [{"score": score, "maxScore": 1, "scoreNormalized": score,
                                         "completed": success, "completedSuccessfully": success}]}
        if failure:
            metadata["scientific_task_failure"] = {"code": failure, "output_sha256": "f" * 64, "retry_or_resample": False}
        status = "scientific_failure" if failure else "completed"
        record = {**identity, "task_id": task, "metadata": metadata, "predicted": score,
                  "scientific_task_status": status, "failure_adjusted_completed_successfully": success and not failure,
                  "task_wall_seconds": 5}
        records.append(record)
        outcomes.append(copy.deepcopy(record))
        calls.append({"task_id": task, "status": "ok", "input_tokens": 100, "output_tokens": 10, "elapsed_seconds": 2})
        events.append({"task_id": task})
        safe.append({"task_id": task, "official_progress_score": score,
                     "official_completed_successfully": success, "failure_adjusted_completed_successfully": success and not failure,
                     "environment_actions": actions, "model_calls": 1})
    files = {}
    for name, rows in [("task_records", records), ("all_data", outcomes), ("calls", calls), ("task_events", events)]:
        path = root / (name + ".jsonl")
        write_rows(path, rows)
        files[name] = path
    files["manifest"] = root / "manifest.json"
    write_json(files["manifest"], manifest)
    mean = sum(row["official_progress_score"] for row in safe) / len(inventory)
    resources = {"job_id": job, "state": "COMPLETED", "exit_code": "0:0", "account": "pi_fl426",
                 "requested_qos": "normal", "qos": "normal", "actual_qos": "normal",
                 "elapsed_seconds": 600, "gpu_count": 2, "cpu_count": 8}
    audit = {"audit_status": "PASS", "audit_scope": scope, "auditor_sha256": AUDITOR,
             "manifest_sha256": analysis.sha256(files["manifest"]), "n_tasks": len(inventory),
             "input_sha256": {kind: {str(path): analysis.sha256(path)} for kind, path in files.items() if kind != "manifest"},
             "source_sha256": {"endpoint.py": analysis.sha256(source / "endpoint.py")},
             "per_task_audited_summary": safe, "scientific_failure_count": 2,
             "scientific_failure_codes": {"missing_action_json": 2}, "model_calls": len(inventory),
             "prompt_tokens": len(inventory) * 100, "completion_tokens": len(inventory) * 10,
             "episode_model_service_seconds": len(inventory) * 2, "model_service_seconds": len(inventory) * 2,
             "context_stress_prompt_tokens": 0, "context_stress_completion_tokens": 0, "context_stress_model_service_seconds": 0,
             "all_prompt_tokens_including_runtime_stress": len(inventory) * 100,
             "all_generated_tokens_including_failed_final_calls_and_runtime_stress": len(inventory) * 10,
             "primary_official_progress_score": {"mean": mean}, "official_completed_successfully_rate": {"mean": 1 / len(inventory)},
             "failure_adjusted_completed_successfully_rate": {"mean": 0}, "mean_environment_actions": sum(r["metadata"]["num_steps"] for r in records) / len(inventory),
             "actual_gpu_hours_from_slurm": 1 / 3, "actual_reserved_cpu_hours_from_slurm": 4 / 3,
             "slurm_accounting": resources}
    audit["input_sha256"]["context_stress"] = {}
    files["audit"] = root / "audit.json"
    write_json(files["audit"], audit)
    return {"root": root, "files": files, "source": source, "manifest": manifest, "audit": audit, "records": records, "resources": resources}


def run(inputs, **kwargs):
    return analysis.analyze(inputs["files"]["manifest"], inputs["files"]["audit"], AUDITOR,
                            source_root=inputs["source"], descriptive_draws=1000, **kwargs)


def rebind(inputs, kind, rows):
    path = inputs["files"][kind]
    write_rows(path, rows)
    inputs["audit"]["input_sha256"][kind][str(path)] = analysis.sha256(path)
    write_json(inputs["files"]["audit"], inputs["audit"])


def test_failure_success_separation_partial_progress_and_all_costs(tmp_path):
    inputs = fixture(tmp_path / "production")
    result = run(inputs)
    panel = result["whole_panel"]
    assert result["assigned_denominator"] == 120 and len(result["by_stratum"]) == 24
    assert panel["official_success_count"] == 1 and panel["failure_adjusted_success_count"] == 0
    assert panel["scientific_failure_count"] == 2 and panel["generated_tokens_in_final_format_failures"] == 20
    assert panel["prompt_tokens"] == 12000 and panel["generated_tokens"] == 1200
    first_id = inputs["records"][0]["task_id"]
    assert next(row for row in result["per_task"] if row["task_id"] == first_id)["official_progress"] == 1
    assert panel["uncompleted_without_format_failure_at_cap_count"] == 1
    assert "exclusive termination cause" in result["budget_boundary"]
    assert result["descriptive_theme_reweighting"]["population_confidence_interval"] is False
    assert result["phase_costs"]["allocation_inventory_complete"] is False


@pytest.mark.parametrize("seeds", [(0,), (0, 1)])
def test_fallback_retains_all_strata_and_cannot_be_full_comparator(tmp_path, seeds):
    result = run(fixture(tmp_path / "production", seeds))
    assert result["assigned_denominator"] == 24 * len(seeds)
    assert result["official_five_seed_panel"] is False
    assert len(result["by_theme"]) == 8 and len(result["by_difficulty"]) == 3
    assert all(row["assigned_tasks"] == len(seeds) for row in result["by_stratum"])


def test_updated_hashes_do_not_permit_survivor_analysis(tmp_path):
    inputs = fixture(tmp_path / "production")
    rebind(inputs, "task_records", inputs["records"][2:])  # selectively remove two failures
    with pytest.raises(ValueError, match="complete assigned denominator"):
        run(inputs)


@pytest.mark.parametrize("damage", ["call_file", "source_file", "wrong_auditor", "no_pass", "qual_scope"])
def test_gate_rejects_changed_evidence_or_unapproved_scope(tmp_path, damage):
    inputs = fixture(tmp_path / "production")
    if damage == "call_file":
        inputs["files"]["calls"].write_text("tampered\n")
    elif damage == "source_file":
        (inputs["source"] / "endpoint.py").write_text("tampered\n")
    else:
        key, value = {"wrong_auditor": ("auditor_sha256", "b" * 64), "no_pass": ("audit_status", "FAIL"),
                      "qual_scope": ("audit_scope", "runtime_qualification")}[damage]
        inputs["audit"][key] = value
        if damage == "qual_scope":
            inputs["manifest"]["audit_scope"] = value
            write_json(inputs["files"]["manifest"], inputs["manifest"])
            inputs["audit"]["manifest_sha256"] = analysis.sha256(inputs["files"]["manifest"])
        write_json(inputs["files"]["audit"], inputs["audit"])
    with pytest.raises(ValueError):
        run(inputs)


def test_recomputed_endpoint_disagrees_with_audit_and_is_rejected(tmp_path):
    inputs = fixture(tmp_path / "production")
    records = copy.deepcopy(inputs["records"])
    records[3]["metadata"]["final_scorecard"][0].update(score=1, scoreNormalized=1)
    records[3]["predicted"] = 1
    rebind(inputs, "task_records", records)
    rebind(inputs, "all_data", records)
    with pytest.raises(ValueError, match="audited task progress"):
        run(inputs)


def test_explicit_remote_prefix_relocation_no_basename_guess(tmp_path):
    inputs = fixture(tmp_path / "production")
    local = str(inputs["root"])
    remote = "/nfs/frozen-production"
    for kind, files in inputs["audit"]["input_sha256"].items():
        inputs["audit"]["input_sha256"][kind] = {remote + path[len(local):]: digest for path, digest in files.items()}
    write_json(inputs["files"]["audit"], inputs["audit"])
    with pytest.raises(ValueError, match="changed or missing"):
        run(inputs)
    result = run(inputs, path_map={remote: local})
    assert result["assigned_denominator"] == 120


def audited_cost_job(inputs, role="production"):
    receipt = inputs["root"] / "sacct.txt"
    receipt.write_text("Synthetic terminal parent Slurm accounting receipt\n")
    return {"job_id": inputs["resources"]["job_id"], "role": role, "resources": inputs["resources"],
            "accounting_receipt_sha256": {str(receipt): analysis.sha256(receipt)},
            "generation_accounting": {"mode": "audit", "audit_file": str(inputs["files"]["audit"]),
                 "audit_sha256": analysis.sha256(inputs["files"]["audit"]), "manifest_file": str(inputs["files"]["manifest"])}}


def cost_inventory(inputs, jobs):
    path = inputs["root"] / "costs.json"
    write_json(path, {"schema": "recoma_phase_cost_inventory_v3", "scope": "receipt-enumerated phase",
                      "expected_job_ids": [job["job_id"] for job in jobs], "jobs": jobs})
    return path


def test_failed_attempt_charged_without_inventing_unrecoverable_tokens(tmp_path):
    inputs = fixture(tmp_path / "production")
    failed_log = inputs["root"] / "failed_calls.jsonl"
    write_rows(failed_log, [{"status": "ok", "input_tokens": 500, "output_tokens": 7, "elapsed_seconds": 3},
                            {"status": "error", "elapsed_seconds": 4}])
    receipt = inputs["root"] / "failed_sacct.txt"
    receipt.write_text("Failure allocation includes work before abort\n")
    stress = inputs["root"] / "failed_stress.json"
    write_json(stress, {"status": "infrastructure_failure", "model_calls": 1,
                        "processed_prompt_tokens": 32768, "generated_tokens": 0})
    failure = {"job_id": "25002", "role": "failed_attempt", "resources": {
        "job_id": "25002", "state": "FAILED", "exit_code": "1:0", "account": "pi_fl426", "qos": "normal",
        "elapsed_seconds": 360, "gpu_count": 1, "cpu_count": 4},
        "accounting_receipt_sha256": {str(receipt): analysis.sha256(receipt)},
        "generation_accounting": {"mode": "partial_logs", "all_attempts_recoverable": False,
            "call_sha256": {str(failed_log): analysis.sha256(failed_log)},
            "stress_sha256": {str(stress): analysis.sha256(stress)},
            "failure_explanation": "Aborted generation has no recoverable full token count"}}
    inventory = cost_inventory(inputs, [audited_cost_job(inputs), failure])
    result = run(inputs, phase_cost_inventory_path=inventory)["phase_costs"]
    assert result["actual_gpu_hours"] == pytest.approx(1 / 3 + .1)
    assert result["actual_reserved_cpu_hours"] == pytest.approx(4 / 3 + .4)
    assert result["generation_costs_accounted_lower_bounds"]["generated_tokens"] == 1207
    assert result["generation_costs_accounted_lower_bounds"]["prompt_tokens"] == 12500
    assert result["generation_costs_exact_if_complete"] is None
    assert result["unknown_generation_cost_job_ids"] == ["25002"]
    assert result["generation_accounting_complete"] is False
    assert result["jobs"][1]["generation_diagnostics"]["failed_stress_attempted_input_tokens_not_certified_processed"] == 32768


def test_parent_job_cannot_be_counted_twice_or_omitted(tmp_path):
    inputs = fixture(tmp_path / "production")
    job = audited_cost_job(inputs)
    path = cost_inventory(inputs, [job, copy.deepcopy(job)])
    with pytest.raises(ValueError, match="unique phase job"):
        run(inputs, phase_cost_inventory_path=path)
    ledger = json.loads(path.read_text()); ledger["expected_job_ids"] = ["25001", "25002"]; ledger["jobs"] = [job]
    write_json(path, ledger)
    with pytest.raises(ValueError, match="coverage missing"):
        run(inputs, phase_cost_inventory_path=path)


def test_qualification_stress_charged_separately_from_scientific_denominator(tmp_path):
    inputs = fixture(tmp_path / "production")
    qualification = fixture(tmp_path / "qualification", (5,), job="25002", qualification=True)
    qualification["audit"].update(audit_scope="runtime_qualification", manifest_sha256=analysis.sha256(qualification["files"]["manifest"]),
        context_stress_prompt_tokens=32768, context_stress_completion_tokens=400, context_stress_model_service_seconds=10,
        all_prompt_tokens_including_runtime_stress=32968,
        all_generated_tokens_including_failed_final_calls_and_runtime_stress=420,
        model_service_seconds=14)
    stress = qualification["root"] / "stress.json"
    write_json(stress, {"separate_runtime_only": True})
    qualification["audit"]["input_sha256"]["context_stress"] = {str(stress): analysis.sha256(stress)}
    write_json(qualification["files"]["audit"], qualification["audit"])
    ledger = cost_inventory(inputs, [audited_cost_job(inputs), audited_cost_job(qualification, "qualification")])
    result = run(inputs, phase_cost_inventory_path=ledger)
    assert result["assigned_denominator"] == 120  # qualification/stress never inflate N
    costs = result["phase_costs"]["generation_costs_exact_if_complete"]
    assert costs["prompt_tokens"] == 44968
    assert costs["generated_tokens"] == 1620
    assert costs["context_stress_generated_tokens"] == 400
    assert costs["model_calls"] == 123
    assert result["phase_costs"]["generation_accounting_complete"] is True


def test_end_to_end_actual_independent_auditor_and_durable_records(tmp_path):
    """Exercise the real auditor→analysis boundary, not just a hand-built PASS."""
    path = Path(__file__).with_name("test_recoma_discoveryworld_v3_audit.py")
    spec = importlib.util.spec_from_file_location("auditor_test_fixtures", path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    inputs = fixtures.fixture(tmp_path, "full_panel")
    fixtures.make_failure(inputs, "not json", "missing_action_json")
    real_auditor_sha = analysis.sha256(Path(__file__).parents[1] / "scripts/audit_recoma_discoveryworld_v3.py")
    inputs["manifest"]["no_jev_calls"] = True
    inputs["manifest"]["artifact_sha256"] = {
        "audit_recoma_discoveryworld_v3.py": real_auditor_sha,
        "analyze_recoma_discoveryworld_v3.py": analysis.sha256(Path(analysis.__file__))}
    inputs["resources"]["job_id"] = "25003"
    inputs["runtime"]["cuda_total_memory_bytes"] = 80 * 1024**3
    inputs["manifest"]["qualified_gpu"] = {"name": inputs["runtime"]["cuda_device"],
                                          "total_memory_bytes": inputs["runtime"]["cuda_total_memory_bytes"]}
    manifest = tmp_path / "manifest.json"
    write_json(manifest, inputs["manifest"])
    audited = fixtures.run(inputs)
    audited.update(manifest_sha256=analysis.sha256(manifest), auditor_sha256=real_auditor_sha)
    audit_path = tmp_path / "passing_audit.json"
    write_json(audit_path, audited)
    result = analysis.analyze(manifest, audit_path, real_auditor_sha, source_root=inputs["source"], descriptive_draws=1000)
    assert result["assigned_denominator"] == 120
    assert result["primary_raw_official_progress"] == .5
    assert result["whole_panel"]["scientific_failure_count"] == 1
    assert result["whole_panel"]["generated_tokens"] == audited["completion_tokens"]
    assert result["descriptive_theme_reweighting"]["middle_95_percent_reweighting_ranges"]["official_progress_mean"] == [.5, .5]
    assert result["provenance"]["all_input_hashes_reverified"] is True


def test_cpu_only_allocation_is_charged_without_fabricating_gpu_time():
    resources = {"state": "COMPLETED", "exit_code": "0:0", "account": "pi_fl426", "qos": "normal",
                 "elapsed_seconds": 600, "gpu_count": 0, "cpu_count": 4}
    gpu, cpu = analysis.allocation(resources)
    assert gpu == 0 and cpu == pytest.approx(2 / 3)
    resources["cpu_count"] = 0
    with pytest.raises(ValueError, match="unallocated"):
        analysis.allocation(resources)


def test_upstream_numeric_string_prediction_preserves_exact_official_endpoint(tmp_path):
    inputs = fixture(tmp_path / "production")
    outcomes = copy.deepcopy(inputs["records"])
    for row in outcomes:
        row["predicted"] = str(row["predicted"])
    rebind(inputs, "all_data", outcomes)
    result = run(inputs)
    assert result["assigned_denominator"] == 120
    assert result["primary_raw_official_progress"] == inputs["audit"]["primary_official_progress_score"]["mean"]


@pytest.mark.parametrize("value", [False, "NaN", "Infinity", "bad", "0.9"])
def test_serialization_does_not_excuse_false_or_wrong_official_prediction(tmp_path, value):
    inputs = fixture(tmp_path / "production")
    outcomes = copy.deepcopy(inputs["records"])
    outcomes[4]["predicted"] = value  # true raw score is0; Boolean False is not an endpoint scalar
    rebind(inputs, "all_data", outcomes)
    with pytest.raises(ValueError, match="prediction"):
        run(inputs)


def qualification_correction(inputs):
    old = "b" * 64
    inputs["manifest"]["artifact_sha256"]["audit_recoma_discoveryworld_v3.py"] = old
    write_json(inputs["files"]["manifest"], inputs["manifest"])
    inputs["audit"]["manifest_sha256"] = analysis.sha256(inputs["files"]["manifest"])
    write_json(inputs["files"]["audit"], inputs["audit"])
    receipt = inputs["root"] / "correction.json"
    contents = {"schema": "recoma_qualification_auditor_correction_v3",
                "original_manifest_sha256": inputs["audit"]["manifest_sha256"],
                "original_auditor_sha256": old, "effective_auditor_sha256": AUDITOR,
                "correction_scope": "numeric_predicted_serialization",
                "reason": "Source-proven numeric scalar versus scalar-string representation; both match official raw ratio",
                "generation_rerun": False, "task_assignment_changed": False, "inference_source_changed": False,
                "raw_inputs_modified": False, "scientific_endpoint_changed": False}
    write_json(receipt, contents)
    return {"receipt_file": str(receipt), "receipt_sha256": analysis.sha256(receipt)}, contents


def test_source_correction_requires_explicit_qualification_only_provenance(tmp_path):
    inputs = fixture(tmp_path / "qualification", (5,), qualification=True)
    correction, _ = qualification_correction(inputs)
    with pytest.raises(ValueError, match="frozen manifest"):
        analysis.verify_audit(inputs["files"]["audit"], inputs["files"]["manifest"], AUDITOR, analysis.Resolver())
    manifest, passed = analysis.verify_audit(inputs["files"]["audit"], inputs["files"]["manifest"], AUDITOR,
                                            analysis.Resolver(), qualification_auditor_correction=correction)
    assert passed["audit_scope"] == "runtime_qualification" and manifest["seeds"] == [5]
    production = fixture(tmp_path / "production")
    correction, _ = qualification_correction(production)
    with pytest.raises(ValueError, match="qualification-only"):
        analysis.verify_audit(production["files"]["audit"], production["files"]["manifest"], AUDITOR,
                              analysis.Resolver(), qualification_auditor_correction=correction)


@pytest.mark.parametrize("field,value", [("generation_rerun", True), ("scientific_endpoint_changed", True),
                                        ("raw_inputs_modified", True), ("original_manifest_sha256", "c" * 64)])
def test_source_correction_cannot_hide_redesign_rerun_or_wrong_provenance(tmp_path, field, value):
    inputs = fixture(tmp_path / "qualification", (5,), qualification=True)
    correction, contents = qualification_correction(inputs)
    contents[field] = value
    write_json(Path(correction["receipt_file"]), contents)
    correction["receipt_sha256"] = analysis.sha256(correction["receipt_file"])
    with pytest.raises(ValueError, match="correction"):
        analysis.verify_audit(inputs["files"]["audit"], inputs["files"]["manifest"], AUDITOR,
                              analysis.Resolver(), qualification_auditor_correction=correction)


@pytest.mark.parametrize("scope,accepted", [
    (["numeric_scalar_json_serialization", "prompt_message_key_order_serialization"], True),
    ("numeric_predicted_and_prompt_key_order_serialization", True),
    (["prompt_message_key_order_serialization"], False),
    (["numeric_scalar_json_serialization", "prompt_message_key_order_serialization", "new_endpoint"], False),
    ("ignore_bad_prompts", False)])
def test_only_source_proven_serialization_scopes_accepted(tmp_path, scope, accepted):
    inputs = fixture(tmp_path / "qualification", (5,), qualification=True)
    correction, contents = qualification_correction(inputs)
    contents["correction_scope"] = scope
    write_json(Path(correction["receipt_file"]), contents)
    correction["receipt_sha256"] = analysis.sha256(correction["receipt_file"])
    if accepted:
        analysis.verify_audit(inputs["files"]["audit"], inputs["files"]["manifest"], AUDITOR,
                              analysis.Resolver(), qualification_auditor_correction=correction)
    else:
        with pytest.raises(ValueError, match="correction provenance"):
            analysis.verify_audit(inputs["files"]["audit"], inputs["files"]["manifest"], AUDITOR,
                                  analysis.Resolver(), qualification_auditor_correction=correction)
