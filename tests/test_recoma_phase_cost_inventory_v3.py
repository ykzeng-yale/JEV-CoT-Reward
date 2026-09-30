"""Receipt reconciliation, allocation deduplication and unknown-work guards."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


builder = module("cost_builder", "build_recoma_phase_cost_inventory_v3.py")
analyzer = module("cost_analyzer", "analyze_recoma_discoveryworld_v3.py")
HEADER = "JobID|State|ExitCode|Submit|Start|End|ElapsedRaw|AllocTRES|Account|QOS|Partition\n"


def receipt(path, job="27975076", *, state="FAILED", elapsed=113, allocation="cpu=4,gres/gpu:h200=1,gres/gpu=1,mem=32G,node=1", start="2026-09-30T17:16:15"):
    row = f"{job}|{state}|1:0|2026-09-30T17:15:39|{start}|2026-09-30T17:18:08|{elapsed}|{allocation}|pi_fl426|normal|gpu_devel\n"
    path.write_text(HEADER + row)
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def no_generation_job(tmp_path):
    accounting = receipt(tmp_path / "sacct.txt")
    runtime = tmp_path / "runtime.json"
    write_json(runtime, {"status": "infrastructure_failure", "slurm_job_id": "27975076", "performed_generation": False})
    return {"job_id": "27975076", "role": "failed_attempt", "accounting_file": str(accounting),
        "generation_accounting": {"mode": "no_generation", "runtime_report_file": str(runtime), "reason": "Torch import failed before generation"}}


def spec(tmp_path, jobs):
    path = tmp_path / "spec.json"
    write_json(path, {"schema": "recoma_phase_cost_spec_v3", "scope": "Explicit test parent jobs",
                      "expected_job_ids": [j["job_id"] for j in jobs], "jobs": jobs})
    return path


def test_parent_tres_and_steps_never_double_gpu_cost(tmp_path):
    path = receipt(tmp_path / "sacct.txt")
    text = path.read_text()
    row = text.splitlines()[1]
    path.write_text(text + row.replace("27975076|", "27975076.batch|") + "\n" + row.replace("27975076|", "27975076.extern|") + "\n" + row.replace("27975076|", "27975076.0|") + "\n")
    actual = builder.parent_accounting(path, "27975076")
    assert actual["gpu_count"] == 1 and actual["cpu_count"] == 4 and actual["memory_gib"] == 32
    assert actual["gpu_type"] == "h200" and actual["elapsed_seconds"] == 113


def test_disagreeing_generic_and_typed_gpu_counts_rejected(tmp_path):
    path = receipt(tmp_path / "sacct.txt", allocation="cpu=4,gres/gpu:h200=1,gres/gpu=2,mem=32G")
    with pytest.raises(ValueError, match="GPU AllocTRES disagree"):
        builder.parent_accounting(path, "27975076")


def test_cancelled_by_uid_empty_allocation_proves_zero_generation(tmp_path):
    path = receipt(tmp_path / "sacct.txt", job="27972098", state="CANCELLED by 22470", elapsed=0, allocation="", start="None")
    job = {"job_id": "27972098", "role": "cancelled_unallocated", "accounting_file": str(path)}
    result = builder.build(spec(tmp_path, [job]))
    assert result["receipt_accounted_gpu_hours"] == result["receipt_accounted_reserved_cpu_hours"] == 0
    assert result["jobs"][0]["generation_accounting"]["generation_proved_not_started"] is True
    assert result["jobs"][0]["resources"]["state"] == "CANCELLED"


def test_early_import_failure_allocation_is_charged_even_without_generation(tmp_path):
    result = builder.build(spec(tmp_path, [no_generation_job(tmp_path)]))
    assert result["receipt_accounted_gpu_hours"] == pytest.approx(113 / 3600)
    assert result["receipt_accounted_reserved_cpu_hours"] == pytest.approx(113 * 4 / 3600)
    assert result["jobs"][0]["generation_accounting"]["mode"] == "no_generation"


def test_no_generation_cannot_hide_actual_started_generation(tmp_path):
    job = no_generation_job(tmp_path)
    write_json(Path(job["generation_accounting"]["runtime_report_file"]), {
        "status": "infrastructure_failure", "slurm_job_id": "27975076", "performed_generation": True})
    with pytest.raises(ValueError, match="never started"):
        builder.build(spec(tmp_path, [job]))


def test_resource_json_must_agree_with_actual_scheduler_allocation(tmp_path):
    job = no_generation_job(tmp_path)
    resource_path = tmp_path / "resources.json"
    resources = builder.parent_accounting(job["accounting_file"], job["job_id"])
    resources["elapsed_seconds"] = 1
    write_json(resource_path, resources)
    job["resources_file"] = str(resource_path)
    with pytest.raises(ValueError, match="elapsed_seconds"):
        builder.build(spec(tmp_path, [job]))


def test_live_job_snapshot_cannot_pass_as_final_and_opens_no_results(tmp_path):
    path = receipt(tmp_path / "sacct.txt", job="27978394", state="RUNNING", elapsed=97)
    job = {"job_id": "27978394", "role": "qualification", "accounting_file": str(path),
           "generation_accounting": {"mode": "audit", "audit_file": "/does/not/exist/never/opened.json"}}
    frozen = spec(tmp_path, [job])
    with pytest.raises(ValueError, match="terminal exact parent"):
        builder.build(frozen)
    snapshot = builder.build(frozen, snapshot=True)
    assert snapshot["schema"] == "recoma_phase_cost_inventory_snapshot_v3"
    assert snapshot["open_job_ids"] == ["27978394"]
    assert snapshot["receipt_accounted_gpu_hours"] == pytest.approx(97 / 3600)
    inventory_path = tmp_path / "snapshot.json"
    write_json(inventory_path, snapshot)
    with pytest.raises(ValueError, match="cost schema mismatch"):
        analyzer.phase_costs(inventory_path, None, {}, "a" * 64, analyzer.Resolver())


def test_exact_parent_required_no_basename_or_multiple_parent_guess(tmp_path):
    path = receipt(tmp_path / "sacct.txt")
    with pytest.raises(ValueError, match="numeric parent"):
        builder.parent_accounting(path, "27975076.batch")
    path.write_text(path.read_text() + path.read_text().splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="ambiguous exact parent"):
        builder.parent_accounting(path, "27975076")


def test_partial_failed_logs_are_bound_not_interpreted_as_outcomes(tmp_path):
    accounting = receipt(tmp_path / "sacct.txt", job="27976321", elapsed=709,
                         allocation="cpu=4,gres/gpu:rtx_pro_6000_blackwell=1,gres/gpu=1,mem=32G")
    calls = tmp_path / "calls.jsonl"
    calls.write_text("Opaque archived bytes; cost builder must not interpret task endpoints\n")
    job = {"job_id": "27976321", "role": "failed_attempt", "accounting_file": str(accounting),
        "generation_accounting": {"mode": "partial_logs", "all_attempts_recoverable": False,
            "call_files": [str(calls)], "failure_explanation": "Failed before hardware stress; recovered calls remain lower bounds"}}
    result = builder.build(spec(tmp_path, [job]))
    assert result["receipt_accounted_gpu_hours"] == pytest.approx(709 / 3600)
    assert result["jobs"][0]["generation_accounting"]["call_sha256"][str(calls.resolve())] == builder.sha256(calls)
    assert result["jobs"][0]["generation_accounting"]["all_attempts_recoverable"] is False


def test_builder_to_analyzer_charges_failures_and_preserves_unknown_totals(tmp_path):
    path = Path(__file__).with_name("test_recoma_discoveryworld_v3_analysis.py")
    loader = importlib.util.spec_from_file_location("analysis_fixtures", path)
    fixtures = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(fixtures)
    production = fixtures.fixture(tmp_path / "production")
    production["resources"]["partition"] = "gpu_devel"
    write_json(production["files"]["audit"], production["audit"])
    prod_receipt = receipt(production["root"] / "sacct.txt", job="25001", state="COMPLETED", elapsed=600,
                           allocation="cpu=8,gres/gpu:h200=2,gres/gpu=2,mem=64G")
    # Match the synthetic independent audit's COMPLETED0:0 exit precisely.
    prod_receipt.write_text(prod_receipt.read_text().replace("|COMPLETED|1:0|", "|COMPLETED|0:0|"))
    prod_job = {"job_id": "25001", "role": "production", "accounting_file": str(prod_receipt),
        "generation_accounting": {"mode": "audit", "audit_file": str(production["files"]["audit"]),
            "manifest_file": str(production["files"]["manifest"]), "expected_auditor_sha256": fixtures.AUDITOR}}
    failed_receipt = receipt(tmp_path / "failed_sacct.txt", job="27976321", elapsed=709)
    calls = tmp_path / "failed_calls.jsonl"
    calls.write_text(json.dumps({"status": "ok", "input_tokens": 100, "output_tokens": 10, "elapsed_seconds": 2}) + "\n")
    failure = {"job_id": "27976321", "role": "failed_attempt", "accounting_file": str(failed_receipt),
        "generation_accounting": {"mode": "partial_logs", "all_attempts_recoverable": False,
            "call_files": [str(calls)], "failure_explanation": "Unlogged failed generation remains unknown"}}
    specification = spec(tmp_path, [prod_job, failure])
    inventory = tmp_path / "inventory.json"
    write_json(inventory, builder.build(specification))
    result = fixtures.run(production, phase_cost_inventory_path=inventory)
    assert result["assigned_denominator"] == 120
    assert result["phase_costs"]["actual_gpu_hours"] == pytest.approx(1 / 3 + 709 / 3600)
    assert result["phase_costs"]["generation_costs_accounted_lower_bounds"]["generated_tokens"] == 1210
    assert result["phase_costs"]["generation_costs_exact_if_complete"] is None


def test_unallocated_label_cannot_hide_an_actual_allocation(tmp_path):
    path = receipt(tmp_path / "cancelled.txt", job="27972098", state="CANCELLED by 22470")
    job = {"job_id": "27972098", "role": "cancelled_unallocated", "accounting_file": str(path)}
    with pytest.raises(ValueError, match="scheduler allocation/start"):
        builder.build(spec(tmp_path, [job]))


def test_unallocated_deadline_has_zero_cost_and_needs_no_runtime_report(tmp_path):
    path = receipt(tmp_path / "deadline.txt", job="25004", state="DEADLINE", elapsed=0, allocation="", start="None")
    job = {"job_id": "25004", "role": "failed_attempt", "accounting_file": str(path),
           "generation_accounting": {"mode": "no_generation"}}
    result = builder.build(spec(tmp_path, [job]))
    assert result["receipt_accounted_gpu_hours"] == result["receipt_accounted_reserved_cpu_hours"] == 0
    assert result["jobs"][0]["generation_accounting"]["generation_proved_not_started"] is True
    assert "DEADLINE" in result["jobs"][0]["generation_accounting"]["reason"]


def test_allocated_deadline_keeps_time_and_cannot_claim_no_generation(tmp_path):
    path = receipt(tmp_path / "deadline.txt", job="25004", state="DEADLINE", elapsed=709)
    job = {"job_id": "25004", "role": "failed_attempt", "accounting_file": str(path),
        "generation_accounting": {"mode": "partial_logs", "all_attempts_recoverable": False,
                                  "failure_explanation": "Deadline terminated allocated work; token counts unrecoverable"}}
    result = builder.build(spec(tmp_path, [job]))
    assert result["receipt_accounted_gpu_hours"] == pytest.approx(709 / 3600)
    assert result["receipt_accounted_reserved_cpu_hours"] == pytest.approx(709 * 4 / 3600)
    assert result["jobs"][0]["generation_accounting"]["all_attempts_recoverable"] is False


def test_completed_cpu_audit_has_own_role_and_charges_only_reserved_cpu(tmp_path):
    accounting = receipt(tmp_path / "cpu_sacct.txt", job="25005", state="COMPLETED", elapsed=360,
                         allocation="cpu=4,mem=8G,node=1")
    execution = tmp_path / "cpu_execution.json"
    write_json(execution, {"job_id": "25005", "status": "completed", "performed_generation": False,
                           "model_generations": 0, "hosted_calls": 0, "jev_calls": 0})
    batch, log = tmp_path / "audit.sbatch", tmp_path / "audit.log"
    batch.write_text("# CPU-only tokenizer/record/byte audit; no model inference\n")
    log.write_text("CPU audit complete with0new model generations\n")
    job = {"job_id": "25005", "role": "cpu_audit", "accounting_file": str(accounting),
        "generation_accounting": {"mode": "no_generation", "execution_receipt_file": str(execution),
            "batch_file": str(batch), "log_file": str(log), "reason": "CPU tokenizer/record/weight-byte verification only"}}
    result = builder.build(spec(tmp_path, [job]))
    assert result["receipt_accounted_gpu_hours"] == 0
    assert result["receipt_accounted_reserved_cpu_hours"] == pytest.approx(.4)
    assert result["jobs"][0]["role"] == "cpu_audit"
    assert len(result["jobs"][0]["generation_accounting"]["evidence_sha256"]) == 3
    proof = json.loads(execution.read_text()); proof["model_generations"] = 1
    write_json(execution, proof)
    with pytest.raises(ValueError, match="inference counter"):
        builder.build(spec(tmp_path, [job]))


@pytest.mark.parametrize("text,value", [("32G", 32), ("32768M", 32), ("33554432K", 32), ("1T", 1024), ("32768", 32)])
def test_slurm_memory_units(text, value):
    assert builder.memory_gib(text) == value
