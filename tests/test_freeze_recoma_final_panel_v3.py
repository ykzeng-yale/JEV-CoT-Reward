"""Timing-only planning tests: no model, simulator, outcomes, SSH or submission."""
import importlib.util
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("freeze_panel", ROOT/"scripts/freeze_recoma_final_panel_v3.py")
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def forecast(seconds):
    return {"task_seconds_with_margin":seconds, "setup_seconds_with_margin":60,
            "cleanup_reserve_seconds":60}


def test_full_panel_uses_smallest_adequate_allocation():
    plan, considered = freeze.choose_plan(forecast(60), .75, freeze.DEADLINE-5*3600, 8)
    assert plan["episodes"] == 120 and plan["workers"] == 1
    assert plan["requested_wall_minutes"] == 122
    assert plan["requested_wall_minutes"] < plan["available_wall_cap_minutes"]
    assert plan["requested_gpu_hour_ceiling"] + .75 <= 16
    assert all(p["episodes"] == 120 for p in considered)


@pytest.mark.parametrize(("seconds", "episodes", "seeds"), [(600,48,[0,1]), (1300,24,[0])])
def test_fallback_follows_only_timing_and_retains_every_stratum(seconds, episodes, seeds):
    plan, _ = freeze.choose_plan(forecast(seconds), .75, freeze.DEADLINE-7200, 8, True, True)
    assert plan["episodes"] == episodes and plan["seeds"] == seeds
    assert plan["workers"] == 8
    assert plan["requested_wall_minutes"] == (62 if episodes == 48 else 67)
    assert plan["requested_gpu_hour_ceiling"] + .75 <= 16
    tasks, workers = freeze.task_assignment(seeds, plan["workers"])
    assert len(tasks) == len({t["task_id"] for t in tasks}) == episodes
    assert {t["scenario"] for t in tasks} == set(freeze.SCENARIOS)
    assert {t["difficulty"] for t in tasks} == {"Easy", "Normal", "Challenge"}
    assert sorted(t["task_id"] for t in tasks) == sorted(t for w in workers for t in w["task_ids"])


def test_fallback_requires_explicit_permission_and_no_arbitrary_partial_tier():
    with pytest.raises(ValueError, match="no complete"):
        freeze.choose_plan(forecast(1300), .75, freeze.DEADLINE-7200, 8)
    with pytest.raises(ValueError, match="no complete"):
        freeze.choose_plan(forecast(10000), .75, freeze.DEADLINE-7200, 8, True, True)
    with pytest.raises(ValueError, match="deadline"):
        freeze.choose_plan(forecast(60), 16, freeze.DEADLINE-7200, 8, True, True)


def test_gpu_devel_two_worker_cap_changes_complete_tier_forecast_and_rejects_overallocation():
    with pytest.raises(ValueError,match="at most two"):
        freeze.choose_plan(forecast(60), .75, freeze.DEADLINE-1800, 8, True, True,["gpu_devel"])
    ordinary,_=freeze.choose_plan(forecast(60), .75, freeze.DEADLINE-1800, 8, True, True,["gpu_h200"])
    devel,_=freeze.choose_plan(forecast(60), .75, freeze.DEADLINE-1800, 2, True, True,["gpu_devel"])
    assert ordinary["episodes"]==120 and ordinary["workers"]==8
    assert devel["episodes"]==48 and devel["workers"]==2 and devel["requested_wall_minutes"]==26
    with pytest.raises(ValueError,match="at most two"):
        freeze.render_batch((ROOT/"cluster/bouchet/recoma_final_panel_v3.sbatch").read_text(),
                           "/nfs/roberts/project/pi_fl426/yz2324/production",3,26,["gpu_devel"])


def test_qualified_gpu_uses_measured_capacity_and_only_unambiguous_scheduler_type():
    audit={"runtime":[{"status":"completed","performed_generation":True,"cuda_device":"NVIDIA A40",
                       "cuda_total_memory_bytes":48000000000}],
           "peak_allocated_gpu_bytes":18000000000,"peak_reserved_gpu_bytes":20000000000,
           "slurm_accounting":{"gpu_type":"a40","node_list":"a1122u02n01","partition":"gpu_devel"}}
    gpu,placement=freeze.qualified_gpu_placement(audit)
    assert gpu=={"name":"NVIDIA A40","total_memory_bytes":48000000000}
    assert placement["gpu_type"]=="a40" and placement["node_list"]=="a1122u02n01"
    audit["slurm_accounting"]["gpu_type"]="h200"
    with pytest.raises(ValueError,match="does not match"):
        freeze.qualified_gpu_placement(audit)
    audit["slurm_accounting"]["gpu_type"]="a40"
    audit["runtime"][0]["cuda_total_memory_bytes"]=19000000000
    with pytest.raises(ValueError,match="physical GPU capacity"):
        freeze.qualified_gpu_placement(audit)
    del audit["runtime"][0]["cuda_total_memory_bytes"]
    with pytest.raises(ValueError,match="physical memory"):
        freeze.qualified_gpu_placement(audit)


@pytest.mark.parametrize(("gpu_type", "device_name", "capacity"), [
    ("rtx_pro_6000_blackwell", "NVIDIA RTX PRO 6000 Blackwell Server Edition", 96000000000),
    ("b200", "NVIDIA B200", 180000000000),
    ("h200", "NVIDIA H200", 141000000000),
])
def test_actual_audited_gpu_types_are_accepted_without_transporting_hardware_qualification(gpu_type,device_name,capacity):
    audit={"runtime":[{"status":"completed","performed_generation":True,"cuda_device":device_name,
                       "cuda_total_memory_bytes":capacity}],
           "peak_allocated_gpu_bytes":20000000000,"peak_reserved_gpu_bytes":24000000000,
           "slurm_accounting":{"gpu_type":gpu_type,"node_gres":f"gpu:{gpu_type}:8",
                               "node_list":"qualified_node", "partition":"gpu_devel"}}
    gpu,placement=freeze.qualified_gpu_placement(audit)
    assert gpu=={"name":device_name,"total_memory_bytes":capacity}
    assert placement["gpu_type"]==gpu_type
    rendered=freeze.render_batch((ROOT/"cluster/bouchet/recoma_final_panel_v3.sbatch").read_text(),
                                "/nfs/roberts/project/pi_fl426/yz2324/qualified-panel",2,50,
                                ["gpu_devel"],placement["gpu_type"],placement["node_list"])
    assert f"#SBATCH --gres=gpu:{gpu_type}:2" in rendered
    assert "#SBATCH --nodelist=qualified_node" in rendered
    # A receipt that claims some other card cannot downgrade silently to generic GPU placement.
    audit["slurm_accounting"]["gpu_type"]="a40"
    with pytest.raises(ValueError,match="unambiguous"):
        freeze.qualified_gpu_placement(audit)
    audit["slurm_accounting"].pop("node_gres")
    with pytest.raises(ValueError,match="does not match"):
        freeze.qualified_gpu_placement(audit)


def test_production_uses_only_actual_qualified_partition_and_never_scavenge():
    qualified={"resources":{"partitions":["gpu_h200","gpu_b200","gpu_h100","gpu_rtx6000","gpu_devel"]}}
    audit={"slurm_accounting":{"partition":"gpu_devel"}}
    assert freeze.measured_production_partitions(qualified,audit)==["gpu_devel"]
    audit["slurm_accounting"]["partition"]="scavenge_gpu"
    with pytest.raises(ValueError,match="actual audited"):
        freeze.measured_production_partitions(qualified,audit)


def test_parent_resource_accounting_charges_failure_and_idle_and_rejects_live():
    qual = {"job_id":"1", "state":"COMPLETED", "account":"pi_fl426", "qos":"normal", "gpu_count":1,
            "elapsed_seconds":1800}
    fail = {**qual, "job_id":"2", "state":"FAILED", "elapsed_seconds":600, "gpu_count":2}
    assert freeze.accounting_total([qual, fail], qual) == pytest.approx(5/6)
    with pytest.raises(ValueError, match="live"):
        freeze.accounting_total([qual, {**fail, "state":"PENDING"}], qual)
    with pytest.raises(ValueError, match="unique"):
        freeze.accounting_total([qual, qual], qual)
    deadline={**fail,"state":"DEADLINE","gpu_count":0,"elapsed_seconds":0}
    assert freeze.accounting_total([qual,deadline],qual)==.5
    assert freeze.accounting_total([qual,{**deadline,"gpu_count":1,"elapsed_seconds":60}],qual)==pytest.approx(.5+1/60)


def test_forecast_consumes_exact_hash_bound_full_envelope_not_early_failure_speed(tmp_path):
    tasks = [{"task_id":"a", "task_wall_seconds":30}, {"task_id":"b", "task_wall_seconds":40}]
    calls = [{"status":"ok", "task_id":"a", "elapsed_seconds":2},
             {"status":"ok", "task_id":"b", "elapsed_seconds":3}]
    paths = {}
    for kind, rows in [("task_records",tasks), ("calls",calls)]:
        path=tmp_path/(kind+".jsonl")
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        paths[kind]={str(path):freeze.sha256(path)}
    manifest={"audit_scope":"runtime_qualification", "qualification_only":True,
              "model_revision":freeze.REVISION, "max_llm_calls_per_episode":62,
              "max_new_tokens_per_generation":400, "qualified_context_token_ceiling":32768}
    audit={"audit_status":"PASS", "audit_scope":"runtime_qualification", "input_sha256":paths,
           "context_stress_model_service_seconds":10, "context_stress_prompt_tokens":32768,
           "context_stress_completion_tokens":400, "slurm_accounting":{"elapsed_seconds":100}}
    result=freeze.qualified_forecast(manifest,audit)
    assert result["task_seconds_with_margin"] == 1.25*(37+62*10)
    assert result["setup_seconds_with_margin"] == 25
    # Scientific output fields are not forecast inputs.
    tasks[0]["made_up_success_rate"]=1.0
    p=Path(next(iter(paths["task_records"])))
    p.write_text("".join(json.dumps(row)+"\n" for row in tasks))
    with pytest.raises(ValueError, match="changed qualification"):
        freeze.qualified_forecast(manifest,audit)


def test_rendered_batch_shell_syntax_and_hard_limits(tmp_path):
    template=(ROOT/"cluster/bouchet/recoma_final_panel_v3.sbatch").read_text()
    text=freeze.render_batch(template,"/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3-production",8,114)
    assert "#SBATCH --gres=gpu:8" in text and "#SBATCH --mem=256G" in text
    assert "#SBATCH --time=01:54:00" in text and "--account=pi_fl426" in text
    assert "--qos=normal" in text and "--partition=gpu_h200,gpu_b200" in text
    assert f"#SBATCH --deadline={freeze.SCHEDULER_DEADLINE}" in text
    assert int(datetime.fromisoformat(freeze.SCHEDULER_DEADLINE).replace(tzinfo=ZoneInfo("America/New_York")).timestamp())==freeze.DEADLINE
    assert "2097152" in text and "4194304" in text and "trap finalize EXIT" in text
    assert "CUDA_VISIBLE_DEVICES=\"${gpu_devices[$index]}\"" in text
    script=tmp_path/"run.sbatch";script.write_text(text)
    subprocess.run(["bash","-n",str(script)],check=True)
    with pytest.raises(ValueError, match="safe"):
        freeze.render_batch(template,"/tmp/project;rm -rf nope",8,114)
    broad=freeze.render_batch(template,"/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3-production",4,62,
                             ["gpu_h200","gpu_b200","gpu_h100","gpu_rtx6000"])
    assert "#SBATCH --partition=gpu_h200,gpu_b200,gpu_h100,gpu_rtx6000" in broad
    with pytest.raises(ValueError, match="qualified"):
        freeze.render_batch(template,"/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3-production",4,62,["paid_gpu"])


@pytest.mark.parametrize("template_name", ["recoma_final_qualification_v3.sbatch", "recoma_final_panel_v3.sbatch"])
def test_module_runtime_pythonpath_survives_source_prepend_and_remains_importable(tmp_path, template_name):
    """Execute the actual export expression: replacing PYTHONPATH must fail this import."""
    template=(ROOT/"cluster/bouchet"/template_name).read_text()
    export_line=next(line for line in template.splitlines() if line.startswith("export PYTHONPATH="))
    module_prefix=tmp_path/"installed_module_prefix";module_prefix.mkdir()
    (module_prefix/"module_environment_probe.py").write_text("identity='inherited installed module runtime'\n")
    inputs=tmp_path/"immutable_inputs"
    env=dict(os.environ, INPUTS=str(inputs), PYTHONPATH=str(module_prefix))
    probe=("import os, sys, module_environment_probe; "
           "assert module_environment_probe.identity == 'inherited installed module runtime'; "
           "assert os.environ['PYTHONPATH'].split(':') == "
           "[os.environ['INPUTS']+'/source/recoma', os.environ['INPUTS']+'/source/discoveryworld', "
           +repr(str(module_prefix))+"]")
    subprocess.run(["bash","-c",export_line+"\n"+shlex.quote(sys.executable)+" -c "+shlex.quote(probe)],
                   env=env,check=True,capture_output=True,text=True)
    # An unset inherited path must not add an empty entry (which imports from the process cwd).
    env.pop("PYTHONPATH")
    empty_probe=("import os; assert os.environ['PYTHONPATH'].split(':') == "
                 "[os.environ['INPUTS']+'/source/recoma', os.environ['INPUTS']+'/source/discoveryworld']")
    subprocess.run(["bash","-c",export_line+"\n"+shlex.quote(sys.executable)+" -c "+shlex.quote(empty_probe)],
                   env=env,check=True,capture_output=True,text=True)


def test_end_to_end_freeze_binds_receipts_sources_resources_and_refuses_overwrite(tmp_path, monkeypatch):
    inputs=tmp_path/"qualified-inputs";inputs.mkdir()
    (inputs/"source").mkdir();(inputs/"source/frozen.txt").write_text("fixture only; no inference")
    census_sources={}
    for relative in freeze.CENSUS_SOURCE_FILES:
        path=inputs/"source/discoveryworld"/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("CPU census source fixture "+relative);census_sources[relative]=freeze.sha256(path)
    source_files=sorted(p for p in (inputs/"source").rglob("*") if p.is_file())
    (inputs/"source/SHA256SUMS").write_text("".join(
        f"{freeze.sha256(path)}  {path.relative_to(inputs/'source')}\n" for path in source_files))
    (inputs/"wheelhouse").mkdir();(inputs/"wheelhouse/pinned.whl").write_text("CPU test fixture")
    names=("run_bounded_recoma_worker_v3.py", "recoma_discoveryworld_react_full_v1.jsonnet",
           "recoma_discoveryworld_hf_v1.lock", "audit_recoma_discoveryworld_v3.py",
           "final_six_hour_validation_protocol_20260930.md")
    for name in names:(inputs/name).write_text("fixture "+name)
    manifest={"audit_scope":"runtime_qualification", "qualification_only":True,
              "model_revision":freeze.REVISION, "max_llm_calls_per_episode":62,
              "max_new_tokens_per_generation":400, "qualified_context_token_ceiling":32768,
              "source_manifest_sha256":freeze.sha256(inputs/"source/SHA256SUMS"),
              "artifact_sha256":{name:freeze.sha256(inputs/name) for name in names},
              "resources":{"partitions":["gpu_h200","gpu_b200","gpu_h100","gpu_rtx6000","gpu_devel"]},
              "seeds_source":"old qualification seed source", "reader_policy":"old qualification reader",
              "selection_rule":"old qualification selection", "supersedes_unallocated_job":"old-job",
              "submission_change_reason":"old qualification request broadening",
              "execution_adapter":{"note":"old qualification-only note"}}
    manifest_path=inputs/"qualification_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    paths={}
    for kind,rows in [("task_records", [{"task_id":"a","task_wall_seconds":2},{"task_id":"b","task_wall_seconds":3}]),
                      ("calls", [{"task_id":"a","status":"ok","elapsed_seconds":1},
                                 {"task_id":"b","status":"ok","elapsed_seconds":1}])]:
        path=tmp_path/(kind+".jsonl");path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        paths[kind]={str(path):freeze.sha256(path)}
    receipt={"job_id":"test-parent", "state":"COMPLETED", "account":"pi_fl426", "qos":"normal",
             "gpu_count":1, "elapsed_seconds":10,"partition":"gpu_devel","gpu_type":"h200",
             "node_list":"a1122u02n01"}
    audit={"audit_status":"PASS", "audit_scope":"runtime_qualification", "input_sha256":paths,
           "manifest_sha256":freeze.sha256(manifest_path), "auditor_sha256":freeze.sha256(inputs/names[3]),
           "context_stress_model_service_seconds":1, "context_stress_prompt_tokens":32768,
           "context_stress_completion_tokens":400, "slurm_accounting":receipt,
           "runtime":[{"status":"completed","performed_generation":True,"cuda_device":"NVIDIA H200",
                        "cuda_total_memory_bytes":141000000000}],
           "peak_allocated_gpu_bytes":12000000000,"peak_reserved_gpu_bytes":14000000000}
    audit_path=tmp_path/"audit.json";audit_path.write_text(json.dumps(audit))
    receipt_path=tmp_path/"parent.json";receipt_path.write_text(json.dumps(receipt))
    output=tmp_path/"production-inputs"
    analyzer=tmp_path/"analyze_recoma_visible_instrument_census_v3.py";analyzer.write_text("CPU fixture analyzer")
    contract={"version":"visible_instrument_census_v3", "analyzer_sha256":freeze.sha256(analyzer),
              "source_sha256":census_sources, "recognized_instrument_names":freeze.CENSUS_INSTRUMENTS,
              "checkpoint_rule":"first_classifiable_candidate_per_task",
              "selection_uses_visible_fields_only":True,"selection_uses_terminal_outcomes":False}
    contract_path=tmp_path/"recoma_visible_instrument_census_v3.json";contract_path.write_text(json.dumps(contract))
    model_checksums=tmp_path/"weight_SHA256SUMS";model_checksums.write_text("CPU model content fixture")
    model_verifier=tmp_path/"verify_model_snapshot_v3.py";model_verifier.write_text("CPU verifier fixture")
    model_verification=tmp_path/"weight_verified.json"
    model_verification.write_text(json.dumps({"status":"PASS_MODEL_SNAPSHOT_CONTENT","snapshot_revision":freeze.REVISION,
        "checksum_file_sha256":freeze.sha256(model_checksums),"verifier_sha256":freeze.sha256(model_verifier),
        "file_count":1,"model_generations":0,"elapsed_seconds":1}))
    analysis_script=tmp_path/"analyze_recoma_discoveryworld_v3.py";analysis_script.write_text("CPU analysis fixture")
    analysis_contract=tmp_path/"recoma_baseline_analysis_contract_v3.md";analysis_contract.write_text("CPU contract fixture")
    analysis_tests=tmp_path/"test_recoma_discoveryworld_v3_analysis.py";analysis_tests.write_text("CPU analysis test fixture")
    args=SimpleNamespace(qualified_inputs=inputs,qualification_audit=audit_path,
        qualification_auditor=inputs/"audit_recoma_discoveryworld_v3.py",
        production_auditor=inputs/"audit_recoma_discoveryworld_v3.py",
        visible_census_analyzer=analyzer,visible_census_contract=contract_path,
        model_checksums=model_checksums,model_verifier=model_verifier,qualification_model_verification=model_verification,
        analysis_script=analysis_script,analysis_contract=analysis_contract,analysis_tests=analysis_tests,capacity_receipt=None,
        phase_resource_receipts=[receipt_path],output=output,max_workers=8,
        allow_48_fallback=True,allow_24_fallback=True,
        batch_template=ROOT/"cluster/bouchet/recoma_final_panel_v3.sbatch",
        remote_root="/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3-production")
    monkeypatch.setattr(freeze.time,"time",lambda:freeze.DEADLINE-5*3600)
    result=freeze.freeze(args)
    frozen=json.loads((output/"panel_manifest.json").read_text())
    assert result["status"]=="FROZEN_NOT_SUBMITTED" and result["episodes"]==120
    assert frozen["audit_scope"]=="full_panel" and len(frozen["task_instances"])==120
    assert frozen["qualification_audit_sha256"]==freeze.sha256(audit_path)
    assert frozen["resources"]["gpu_count"]==result["gpu_count"]==1
    assert frozen["resources"]["partitions"]==["gpu_devel"]
    assert frozen["resource_planning"]["effective_qualified_partition_worker_cap"]==2
    assert frozen["qualified_gpu"]=={"name":"NVIDIA H200","total_memory_bytes":141000000000}
    for field in ("seeds_source","reader_policy","selection_rule"):
        assert "old qualification" not in frozen[field]
    assert "supersedes_unallocated_job" not in frozen and "submission_change_reason" not in frozen
    assert frozen["parent_qualification_job_id"]=="test-parent"
    assert frozen["visible_census_contract"]==contract
    assert frozen["visible_census_contract_sha256"]==freeze.sha256(contract_path)
    assert frozen["artifact_sha256"][analyzer.name]==freeze.sha256(analyzer)
    assert frozen["artifact_sha256"]["model_snapshot_SHA256SUMS"]==freeze.sha256(model_checksums)
    assert frozen["model_content_contract"]["require_compute_allocation_pre_and_post_verification"] is True
    assert frozen["prospective_outcome_analysis"]["script_sha256"]==freeze.sha256(analysis_script)
    assert frozen["artifact_sha256"][analysis_contract.name]==freeze.sha256(analysis_contract)
    assert frozen["qualification_auditor_provenance"]["original_frozen_auditor_sha256"]==freeze.sha256(inputs/names[3])
    assert (output/"qualification_original_frozen_auditor.py").read_bytes()==(inputs/names[3]).read_bytes()
    assert (output/"qualification_independent_auditor.py").read_bytes()==(inputs/names[3]).read_bytes()
    assert frozen["resources"]["gpu_count"]*frozen["resources"]["wall_minutes"]/60+10/3600<=16
    assert frozen["scheduler_deadline"]==freeze.SCHEDULER_DEADLINE and frozen["scheduler_deadline_timezone"]=="America/New_York"
    assert (output/"source/frozen.txt").read_text()=="fixture only; no inference"
    for line in (output/"SHA256SUMS").read_text().splitlines():
        expected,relative=line.split("  ",1)
        assert freeze.sha256(output/relative)==expected
    subprocess.run(["bash","-n",str(output/"recoma_final_panel_v3.sbatch")],check=True)
    rendered=(output/"recoma_final_panel_v3.sbatch").read_text()
    assert "#SBATCH --partition=gpu_devel" in rendered
    assert "#SBATCH --gres=gpu:h200:1" in rendered and "#SBATCH --nodelist=a1122u02n01" in rendered
    assert "torch.cuda.get_device_properties(i).total_memory==qualified['total_memory_bytes']" in rendered
    assert "model-verification-before.json" in rendered and "model-verification-after.json" in rendered
    assert rendered.index('model-verification-before.json') < rendered.index('import torch')
    with pytest.raises(ValueError,match="overwrite"):
        freeze.freeze(args)
    # A corrected independent re-audit source is an explicit artifact. Never mutate the original qualifier pins.
    corrected_dir=tmp_path/"corrected-independent";corrected_dir.mkdir()
    corrected=corrected_dir/"audit_recoma_discoveryworld_v3.py";corrected.write_text("source-proven serialization audit correction")
    args.qualification_auditor=corrected;args.production_auditor=corrected;args.output=tmp_path/"production-corrected-auditor"
    with pytest.raises(ValueError,match="explicit independent"):
        freeze.freeze(args)
    audit["auditor_sha256"]=freeze.sha256(corrected);audit_path.write_text(json.dumps(audit))
    freeze.freeze(args)
    new=json.loads((args.output/"panel_manifest.json").read_text())
    assert new["qualification_auditor_provenance"]["effective_independent_qualification_auditor_sha256"]==freeze.sha256(corrected)
    assert new["artifact_sha256"]["audit_recoma_discoveryworld_v3.py"]==freeze.sha256(corrected)
    assert (args.output/"qualification_original_frozen_auditor.py").read_bytes()==(inputs/names[3]).read_bytes()
    assert freeze.sha256(inputs/names[3])==manifest["artifact_sha256"][names[3]]


def test_same_family_capacity_requires_hash_bound_normal_pi_access_and_exact_type(tmp_path):
    raw=tmp_path/"scheduler.txt"
    raw.write_text("PartitionName=gpu_devel AllowAccounts=ALL AllowQos=ALL\n"
                   "PartitionName=gpu_rtx6000 AllowAccounts=ALL AllowQos=normal\n"
                   "NodeName=normal_rtx_node Gres=gpu:rtx_pro_6000_blackwell:8 Partitions=gpu_rtx6000\n")
    receipt={"account":"pi_fl426","qos":"normal","gpu_type":"rtx_pro_6000_blackwell",
             "production_partitions":["gpu_devel","gpu_rtx6000"],"max_workers":8,
             "inspected_at_epoch":freeze.time.time(),"input_sha256":{str(raw):freeze.sha256(raw)}}
    path=tmp_path/"capacity.json";path.write_text(json.dumps(receipt))
    placement={"gpu_type":"rtx_pro_6000_blackwell"}
    assert freeze.verified_same_family_capacity(path,placement)==receipt
    with pytest.raises(ValueError,match="different qualified"):
        freeze.verified_same_family_capacity(path,{"gpu_type":"h200"})
    raw.write_text(raw.read_text().replace("AllowQos=normal","AllowQos=priority"))
    with pytest.raises(ValueError,match="changed"):
        freeze.verified_same_family_capacity(path,placement)
    receipt["input_sha256"][str(raw)]=freeze.sha256(raw);path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="normal PI access"):
        freeze.verified_same_family_capacity(path,placement)


def test_visible_census_contract_rejects_outcome_selection_and_changed_source(tmp_path):
    analyzer=tmp_path/"analyze_recoma_visible_instrument_census_v3.py";analyzer.write_text("fixture")
    pins={}
    for relative in freeze.CENSUS_SOURCE_FILES:
        p=tmp_path/"source"/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(relative)
        pins[relative]=freeze.sha256(p)
    contract={"version":"visible_instrument_census_v3", "analyzer_sha256":freeze.sha256(analyzer),
              "source_sha256":pins, "recognized_instrument_names":freeze.CENSUS_INSTRUMENTS,
              "checkpoint_rule":"first_classifiable_candidate_per_task",
              "selection_uses_visible_fields_only":True,"selection_uses_terminal_outcomes":False}
    path=tmp_path/"contract.json";path.write_text(json.dumps(contract))
    assert freeze.checked_visible_census_contract(analyzer,path,tmp_path/"source")==contract
    contract["selection_uses_terminal_outcomes"]=True;path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match="never outcomes"):
        freeze.checked_visible_census_contract(analyzer,path,tmp_path/"source")
    contract["selection_uses_terminal_outcomes"]=False;path.write_text(json.dumps(contract))
    (tmp_path/"source"/next(iter(pins))).write_text("changed")
    with pytest.raises(ValueError,match="source changed"):
        freeze.checked_visible_census_contract(analyzer,path,tmp_path/"source")
