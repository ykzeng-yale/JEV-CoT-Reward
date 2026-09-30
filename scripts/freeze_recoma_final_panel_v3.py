#!/usr/bin/env python3
"""Freeze the largest complete timing-only ReCoMA census after a strict GPU audit.

No model, simulator, outcome-rate selection, scheduler submission, or API call.
The sole accepted measurements are hash-bound records from a PASS qualification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

DEADLINE = 1790821618
SCHEDULER_DEADLINE = "2026-09-30T22:26:58"
PHASE_CAP = 16.0
SCENARIOS = ["Combinatorial Chemistry", "Archaeology Dating", "Plant Nutrients", "Reactor Lab",
             "Lost in Translation", "Space Sick", "Proteomics", "It's (not) Rocket Science!"]
DIFFICULTIES = ["Easy", "Normal", "Challenge"]
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ALLOWED_PARTITIONS = {"gpu_h200", "gpu_b200", "gpu_h100", "gpu_rtx6000", "gpu_devel"}
CENSUS_SOURCE_FILES = {"agents/recoma/prompts/react_prompt.txt", "agents/recoma/discoveryworld_promptlm.py",
                       "discoveryworld/UserInterface.py", "discoveryworld/objects/ScienceTools.py"}
CENSUS_INSTRUMENTS = ["microscope", "ph meter", "radiation meter", "spectrometer", "thermometer",
                      "densitometer", "proteomics meter"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value, name, *, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "invalid " + name)
    return float(value)


def hash_bound_rows(audit, kind):
    result = []
    pins = audit["input_sha256"][kind]
    require(isinstance(pins, dict) and pins, "missing independently audited " + kind)
    for filename, expected in pins.items():
        path = Path(filename)
        require(sha256(path) == expected, "changed qualification artifact: " + filename)
        result.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return result


def accounting_total(receipts, qualification):
    """Count terminal parent allocations only, including failed attempts/idle time."""
    total, seen, matched = 0.0, set(), False
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "DEADLINE"}
    for row in receipts:
        job = str(row.get("job_id", row.get("job", row.get("slurm_job_id", ""))))
        require(job and "." not in job and job not in seen, "unique parent job accounting required")
        seen.add(job)
        require(row.get("state") in terminal, "live/ambiguous phase allocation cannot be ignored")
        require(row.get("account") == "pi_fl426" and row.get("qos", "normal") == "normal",
                "phase accounting uses unauthorized account/tier")
        gpus = row.get("gpu_count", row.get("allocated_gpu_count"))
        require(type(gpus) is int and 0 <= gpus <= 8, "invalid allocated GPU count")
        seconds = number(row.get("elapsed_seconds"), "parent allocated elapsed seconds")
        total += seconds * gpus / 3600
        if row == qualification:
            matched = True
    require(matched, "qualification accounting must appear exactly in the phase receipts")
    require(total < PHASE_CAP, "phase allocated GPU-hour cap is exhausted")
    return total


def qualified_forecast(manifest, audit):
    """Ignore endpoint rates; extrapolate the full call/context envelope explicitly."""
    require(audit.get("audit_status") == "PASS" and audit.get("audit_scope") == "runtime_qualification",
            "independent real-GPU qualification PASS required")
    require(manifest.get("audit_scope") == "runtime_qualification" and manifest.get("qualification_only") is True,
            "qualification manifest must prohibit efficacy pooling")
    require(manifest.get("model_revision") == REVISION and manifest.get("max_llm_calls_per_episode") == 62
            and manifest.get("max_new_tokens_per_generation") == 400
            and manifest.get("qualified_context_token_ceiling") == 32768, "qualified runtime envelope differs")
    tasks, calls = hash_bound_rows(audit, "task_records"), hash_bound_rows(audit, "calls")
    require(len(tasks) == 2 and len({r["task_id"] for r in tasks}) == 2, "two durable qualification tasks required")
    require(all(c.get("status") == "ok" for c in calls) and calls, "qualification generation error/missing calls")
    by_task = {r["task_id"]: [] for r in tasks}
    for call in calls:
        require(call.get("task_id") in by_task, "unassigned qualification call")
        by_task[call["task_id"]].append(number(call.get("elapsed_seconds"), "call latency", minimum=1e-300))
    walls, overheads = [], []
    for row in tasks:
        wall = number(row.get("task_wall_seconds"), "end-to-end task wall", minimum=1e-300)
        service = sum(by_task[row["task_id"]])
        require(service <= wall + 0.1, "call service exceeds measured task wall")
        walls.append(wall)
        overheads.append(max(0.0, wall - service))
    stress_seconds = number(audit.get("context_stress_model_service_seconds"), "full context stress latency", minimum=1e-300)
    require(audit.get("context_stress_prompt_tokens") == 32768 and audit.get("context_stress_completion_tokens") == 400,
            "full hardware token envelope was not qualified")
    allocation = number(audit["slurm_accounting"].get("elapsed_seconds"), "qualification allocation", minimum=1e-300)
    require(sum(walls) + stress_seconds <= allocation + 1.0, "qualification timeline is inconsistent")
    setup = max(0.0, allocation - sum(walls) - stress_seconds)
    slow_call = max([stress_seconds] + [value for values in by_task.values() for value in values])
    return {"margin_multiplier": 1.25, "full_horizon_calls_per_task": 62,
            "measured_task_wall_seconds": walls, "measured_non_model_overhead_seconds": overheads,
            "measured_setup_and_allocation_overhead_seconds": setup,
            "measured_slow_call_seconds_including_full_context_stress": slow_call,
            "task_seconds_with_margin": 1.25 * max(max(walls), max(overheads) + 62 * slow_call),
            "setup_seconds_with_margin": 1.25 * setup, "cleanup_reserve_seconds": 60,
            "basis": "Maximum observed end-to-end and non-model overhead; 62 times slower of measured real-call/full32768+400stress; observed setup/allocated idle;25%margin. No endpoint-rate inputs.",
            "hardware_transport_boundary": "Forecast applies only to the exact measured GPU name and capacity. Other GPU families are unqualified and must be rejected before production model loading; even matching hardware does not guarantee unseen dynamic-scene throughput. Hard deadline/caps remain enforced."}


def choose_plan(forecast, spent, now, max_workers, allow48=False, allow24=False, partitions=None):
    require(type(max_workers) is int and 1 <= max_workers <= 8, "at most eight independent workers")
    if partitions and "gpu_devel" in partitions:
        require(max_workers <= 2, "gpu_devel permits at most two GPU workers per user")
    reserve = PHASE_CAP - number(spent, "phase allocated GPU hours")
    remaining = DEADLINE - now
    require(remaining > 0 and reserve > 0, "deadline or phase GPU-hour reserve exhausted")
    forecasts = []
    for seeds in ([0, 1, 2, 3, 4], [0, 1], [0]):
        permitted = len(seeds) == 5 or (len(seeds) == 2 and allow48) or (len(seeds) == 1 and allow24)
        for workers in range(1, max_workers + 1):
            per_worker_tasks = math.ceil(8 / workers) * 3 * len(seeds)
            predicted = forecast["setup_seconds_with_margin"] + per_worker_tasks * forecast["task_seconds_with_margin"]
            available_wall_minutes = math.floor(min(remaining, reserve * 3600 / workers) / 60)
            wall_minutes = math.ceil((predicted + forecast["cleanup_reserve_seconds"]) / 60)
            if partitions and "gpu_devel" in partitions:
                available_wall_minutes = min(available_wall_minutes, 360)
            fits = wall_minutes > 0 and wall_minutes <= available_wall_minutes
            item = {"episodes": 24 * len(seeds), "seeds": seeds, "workers": workers,
                    "maximum_worker_tasks": per_worker_tasks, "predicted_slowest_worker_seconds": predicted,
                    "requested_wall_minutes": wall_minutes, "available_wall_cap_minutes": available_wall_minutes,
                    "requested_gpu_hour_ceiling": workers * wall_minutes / 60,
                    "fits": fits, "fallback_permitted": permitted}
            forecasts.append(item)
            if fits and permitted:
                return item, forecasts
    raise ValueError("no complete prospectively permitted census fits measured throughput/deadline/reserve")


def task_assignment(seeds, worker_count):
    tasks, workers = [], [{"index": i, "task_ids": [], "scenario_prefixes": [], "seeds": seeds}
                          for i in range(worker_count)]
    for index, scenario in enumerate(SCENARIOS):
        worker = workers[index % worker_count]
        worker["scenario_prefixes"].append(scenario)
        for difficulty in DIFFICULTIES:
            for seed in seeds:
                row = {"scenario": scenario, "difficulty": difficulty, "seed": seed,
                       "task_id": f"{scenario}_{difficulty}_{seed}"}
                tasks.append(row)
                worker["task_ids"].append(row["task_id"])
    return tasks, workers


def render_batch(template, remote_root, workers, wall_minutes, partitions=None, gpu_type=None, node_list=None):
    require(re.fullmatch(r"/nfs/roberts/project/pi_fl426/yz2324/[A-Za-z0-9_./-]+", remote_root) is not None,
            "remote run root must be an explicit safe PI-project path")
    partitions = partitions or ["gpu_h200", "gpu_b200"]
    require(isinstance(partitions, list) and len(partitions) == len(set(partitions))
            and set(partitions) <= ALLOWED_PARTITIONS, "only qualified standard GPU partitions allowed")
    if "gpu_devel" in partitions:
        require(workers <= 2, "gpu_devel permits at most two GPU workers per user")
        require(wall_minutes <= 360, "gpu_devel wall request cannot exceed six hours")
    require(gpu_type is None or re.fullmatch(r"[A-Za-z0-9_]+", gpu_type), "invalid scheduler GPU type")
    require(node_list is None or re.fullmatch(r"[A-Za-z0-9_-]+", node_list), "one unambiguous qualified node required")
    values = {"__REMOTE_ROOT__": remote_root, "__GPU_COUNT__": str(workers), "__PARTITIONS__": ",".join(partitions),
              "__GPU_GRES__": f"gpu:{gpu_type}:{workers}" if gpu_type else f"gpu:{workers}",
              "__NODE_PLACEMENT__": f"#SBATCH --nodelist={node_list}" if node_list else "# Exact GPU name/capacity verified before any production model load.",
              "__CPU_COUNT__": str(4 * workers), "__MEMORY_GIB__": str(32 * workers),
              "__WALL_TIME__": f"{wall_minutes // 60:02d}:{wall_minutes % 60:02d}:00"}
    result = template
    for key, value in values.items():
        require(key in result, "batch template marker missing: " + key)
        result = result.replace(key, value)
    require("__" not in result, "unresolved batch template marker")
    return result


def qualified_gpu_placement(audit):
    """Use measured card identity/capacity; never infer A40/Ada memory or speed."""
    reports = audit.get("runtime")
    if isinstance(reports, dict):
        reports = [reports]
    require(isinstance(reports, list) and len(reports) == 1, "one real qualification runtime report required")
    report = reports[0]
    require(report.get("status") == "completed" and report.get("performed_generation") is True,
            "hardware qualification did not complete generation")
    name, total = report.get("cuda_device"), report.get("cuda_total_memory_bytes")
    require(isinstance(name, str) and name and type(total) is int and total > 0,
            "measured qualified GPU name and physical memory required")
    reserved = number(audit.get("peak_reserved_gpu_bytes"), "audited peak reserved GPU memory", minimum=1)
    allocated = number(audit.get("peak_allocated_gpu_bytes"), "audited peak allocated GPU memory", minimum=1)
    require(allocated <= reserved <= total, "qualified full-context stress peak exceeds measured physical GPU capacity")
    resource = audit["slurm_accounting"]
    # Only explicit scheduler metadata can introduce a typed GRES request.
    # Never route production generically after an ambiguous/mismatched mapping.
    # A valid scheduler type and the independent exact-card runtime guard are both required.
    raw = resource.get("node_gres", resource.get("gres", ""))
    types = set(re.findall(r"(?:^|[,\s])(?:gres/)?gpu:([A-Za-z0-9_]+):\d+", raw)) if isinstance(raw, str) else set()
    explicit_type = resource.get("gpu_type")
    if isinstance(explicit_type, str) and re.fullmatch(r"[A-Za-z0-9_]+", explicit_type):
        types.add(explicit_type)
    require(len(types) == 1, "one unambiguous audited scheduler GPU type required for production")
    gpu_type = next(iter(types))
    normalized_name = re.sub(r"[^a-z0-9]", "", name.casefold())
    require(re.sub(r"[^a-z0-9]", "", gpu_type.casefold()) in normalized_name,
            "audited scheduler GPU type does not match the measured qualification device")
    raw_node = resource.get("node_list", resource.get("node", report.get("hostname")))
    node = raw_node if isinstance(raw_node, str) and re.fullmatch(r"[A-Za-z0-9_-]+", raw_node) else None
    return {"name":name, "total_memory_bytes":total}, {"gpu_type":gpu_type, "node_list":node,
        "qualification_peak_allocated_gpu_bytes":allocated, "qualification_peak_reserved_gpu_bytes":reserved,
        "qualification_remaining_physical_memory_bytes":total-reserved,
        "policy":"No cross-card throughput/memory transport. Exact qualified GPU name and physical capacity must match before every worker loads or generates. An unambiguous receipted scheduler type matching the measured card is required and pinned as typed GRES; mismatched or ambiguous qualification metadata rejects the freeze."}


def measured_production_partitions(qualified, audit):
    requested = qualified["resources"]["partitions"]
    require(isinstance(requested, list) and requested and len(requested) == len(set(requested))
            and set(requested) <= ALLOWED_PARTITIONS, "qualified GPU partitions missing or unsupported")
    actual = audit["slurm_accounting"].get("partition")
    require(actual in requested and actual in ALLOWED_PARTITIONS,
            "actual audited qualification partition must belong to the frozen authorized request")
    # No queue-driven move onto an unmeasured partition or preemptible route.
    return [actual]


def verified_same_family_capacity(path, placement):
    """Optional hash-bound scheduler evidence expands placement, never model qualification."""
    receipt=json.loads(path.read_text())
    require(receipt.get("account")=="pi_fl426" and receipt.get("qos")=="normal",
            "capacity route must preserve the PI and standard QoS")
    require(receipt.get("gpu_type")==placement["gpu_type"], "capacity route has a different qualified GPU type")
    require(receipt.get("production_partitions")==["gpu_devel","gpu_rtx6000"]
            and placement["gpu_type"]=="rtx_pro_6000_blackwell" and receipt.get("max_workers")==8,
            "only inspected standard same-family RTX Pro routes may expand capacity")
    inspected=number(receipt.get("inspected_at_epoch"), "capacity inspection time", minimum=1)
    require(0 <= time.time()-inspected <= 1800, "capacity inspection must be recent and not future-dated")
    pins=receipt.get("input_sha256")
    require(isinstance(pins,dict) and pins, "raw hash-bound scheduler capacity observations required")
    texts=[]
    for filename,digest in pins.items():
        raw=Path(filename)
        require(sha256(raw)==digest,"scheduler capacity observation changed")
        texts.append(raw.read_text())
    combined="\n".join(texts)
    for partition in receipt["production_partitions"]:
        sections=re.split(r"(?=PartitionName=)",combined)
        records=[s for s in sections if s.startswith("PartitionName="+partition+" ") or s.startswith("PartitionName="+partition+"\n")]
        require(records,"raw scheduler partition observation missing: "+partition)
        def accepted(section,key,value):
            match=re.search(r"\b"+key+r"=([^\s]+)",section)
            return match and (match[1]=="ALL" or value in match[1].split(","))
        require(any(accepted(s,"AllowAccounts","pi_fl426") and accepted(s,"AllowQos","normal") for s in records),
                "normal PI access not established for capacity partition: "+partition)
    node_sections=re.split(r"(?=NodeName=)",combined)
    require(any(s.startswith("NodeName=") and re.search(r"\bGres=gpu:rtx_pro_6000_blackwell:[1-9]\d*",s)
                and re.search(r"\bPartitions=[^\s]*gpu_rtx6000",s) for s in node_sections),
            "raw normal-node typed GPU capacity observation required")
    return receipt


def checked_promotion_artifacts(args, revision):
    verification=json.loads(args.qualification_model_verification.read_text())
    require(args.model_verifier.name=="verify_model_snapshot_v3.py", "unexpected model verifier source name")
    require(verification.get("status")=="PASS_MODEL_SNAPSHOT_CONTENT"
            and verification.get("snapshot_revision")==revision
            and verification.get("checksum_file_sha256")==sha256(args.model_checksums)
            and verification.get("verifier_sha256")==sha256(args.model_verifier),
            "independent exact-weight content promotion receipt differs from frozen model/verifier")
    require(verification.get("file_count",0)>0 and verification.get("model_generations")==0,
            "exact-weight verification must be content-only and nonempty")
    elapsed=number(verification.get("elapsed_seconds"), "model-content verification elapsed seconds", minimum=0)
    require(args.analysis_script.name=="analyze_recoma_discoveryworld_v3.py"
            and args.analysis_contract.name=="recoma_baseline_analysis_contract_v3.md"
            and args.analysis_tests.name=="test_recoma_discoveryworld_v3_analysis.py",
            "prospective analysis script/contract/tests have unexpected artifact names")
    for path in (args.analysis_script,args.analysis_contract,args.analysis_tests):
        require(path.is_file() and path.stat().st_size>0, "prospective outcome analysis artifacts required")
    return verification,elapsed


def checked_visible_census_contract(analyzer_path, contract_path, source_root):
    """Bind the prospective visible-only rule without reading any episode outcome."""
    require(analyzer_path.name == "analyze_recoma_visible_instrument_census_v3.py",
            "unexpected visible census analyzer artifact name")
    contract = json.loads(contract_path.read_text())
    require(contract.get("version") == "visible_instrument_census_v3"
            and contract.get("analyzer_sha256") == sha256(analyzer_path),
            "prospective census analyzer/contract identity differs")
    require(contract.get("selection_uses_visible_fields_only") is True
            and contract.get("selection_uses_terminal_outcomes") is False,
            "census checkpoint selection must use visible fields only, never outcomes")
    require(contract.get("checkpoint_rule") == "first_classifiable_candidate_per_task"
            and contract.get("recognized_instrument_names") == CENSUS_INSTRUMENTS,
            "census checkpoint/instrument rule differs from the prospective contract")
    pins = contract.get("source_sha256", {})
    require(set(pins) == CENSUS_SOURCE_FILES, "all four visible census source pins required")
    for relative, expected in pins.items():
        require(sha256(source_root/relative) == expected, "visible census source changed: " + relative)
    return contract


def freeze(args):
    inputs = args.qualified_inputs.resolve()
    qualified_path = inputs / "qualification_manifest.json"
    qualified = json.loads(qualified_path.read_text())
    audit = json.loads(args.qualification_audit.read_text())
    require(audit.get("manifest_sha256") == sha256(qualified_path), "audit binds a different qualification manifest")
    require(audit.get("auditor_sha256") == sha256(args.qualification_auditor),
            "explicit independent qualification auditor source differs from PASS receipt")
    require(args.production_auditor.name=="audit_recoma_discoveryworld_v3.py"
            and args.qualification_auditor.name=="audit_recoma_discoveryworld_v3.py",
            "explicit independent auditor artifact names required")
    pins = qualified.get("artifact_sha256")
    require(isinstance(pins, dict) and pins, "qualification auxiliary artifact pins required")
    for name, expected in pins.items():
        require(sha256(inputs / name) == expected, "qualified auxiliary artifact changed: " + name)
    partitions = measured_production_partitions(qualified, audit)
    require(sha256(inputs / "source/SHA256SUMS") == qualified["source_manifest_sha256"], "qualified source manifest changed")
    for line in (inputs / "source/SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(sha256(inputs / "source" / relative) == expected, "qualified source changed: " + relative)
    census_contract = checked_visible_census_contract(args.visible_census_analyzer, args.visible_census_contract,
                                                     inputs/"source/discoveryworld")
    qualified_gpu, placement = qualified_gpu_placement(audit)
    verification,verify_seconds=checked_promotion_artifacts(args,qualified["model_revision"])
    capacity=verified_same_family_capacity(args.capacity_receipt,placement) if args.capacity_receipt else None
    if capacity:
        partitions=["gpu_rtx6000"]
        placement["node_list"]=None
        placement["capacity_receipt_sha256"]=sha256(args.capacity_receipt)
        placement["placement_scope"]="Inspected standard same-GPU-family nodes; exact qualified device name/capacity guard remains mandatory."
    forecast = qualified_forecast(qualified, audit)
    forecast["weight_verification_seconds_with_margin"]=2*1.25*verify_seconds
    forecast["setup_seconds_with_margin"]+=forecast["weight_verification_seconds_with_margin"]
    receipts = [json.loads(path.read_text()) for path in args.phase_resource_receipts]
    spent = accounting_total(receipts, audit["slurm_accounting"])
    now = int(time.time())
    require(type(args.max_workers) is int and 1 <= args.max_workers <= 8, "at most eight requested independent workers")
    effective_worker_cap = min(args.max_workers, 2) if "gpu_devel" in partitions else args.max_workers
    plan, considered = choose_plan(forecast, spent, now, effective_worker_cap,
                                   args.allow_48_fallback, args.allow_24_fallback, partitions)
    if capacity and plan["workers"]<=2:
        partitions=capacity["production_partitions"]
    require(not args.output.exists(), "refuse to overwrite a frozen production input directory")
    tasks, workers = task_assignment(plan["seeds"], plan["workers"])
    manifest = copy.deepcopy(qualified)
    for stale in ("qualification_only", "context_stress_input_tokens", "context_stress_output_tokens", "artifact_sha256",
                  "supersedes_unallocated_job", "submission_change_reason", "retry_of_failed_job"):
        manifest.pop(stale, None)
    manifest.update(version="v3-final-six-hour-production-census", audit_scope="full_panel" if len(plan["seeds"]) == 5 else "balanced_census",
        study="Frozen complete Qwen/ReCoMA stratum baseline and cost/trace census", seeds=plan["seeds"], task_instances=tasks, workers=workers,
        qualification_only=False, timing_based_fallback=len(plan["seeds"]) < 5, selection_without_outcome_rates=True,
        qualification_audit_sha256=sha256(args.qualification_audit),
        qualification_auditor_provenance={"original_frozen_auditor_sha256":sha256(inputs/"audit_recoma_discoveryworld_v3.py"),
            "effective_independent_qualification_auditor_sha256":sha256(args.qualification_auditor),
            "production_auditor_sha256":sha256(args.production_auditor),
            "immutable_qualification_input_pins_preserved":True,
            "audit_source_correction_boundary":"A separately versioned independent source can repair source-proven serialization accounting; no task, endpoint, outcome or controller mutation is permitted."},
        qualified_gpu=qualified_gpu, qualified_gpu_placement=placement,
        model_content_contract={"snapshot_revision":qualified["model_revision"],
            "checksums_sha256":sha256(args.model_checksums),"verifier_sha256":sha256(args.model_verifier),
            "qualification_verification_sha256":sha256(args.qualification_model_verification),
            "require_compute_allocation_pre_and_post_verification":True,
            "verification_latency_is_allocated_cost":True},
        prospective_outcome_analysis={"script_sha256":sha256(args.analysis_script),
            "contract_sha256":sha256(args.analysis_contract),"tests_sha256":sha256(args.analysis_tests),
            "frozen_before_production_generation":True},
        parent_qualification_job_id=str(audit["slurm_accounting"].get("job_id",
            audit["slurm_accounting"].get("job", audit["slurm_accounting"].get("slurm_job_id", "")))),
        submission_lineage="New separately frozen production request after audited qualification; no resurrection or supersession of an unallocated historical job.",
        seeds_source="Published DiscoveryWorld benchmark API seeds 0,1,2,3,4. The separately preregistered timing fallbacks retain all24strata at seeds0,1 or seed0; runtime-only seed5 is excluded.",
        reader_policy="Use exactly the newly frozen production API seeds from the official0–4panel; each worker applies the exact task-ID whitelist over its assigned scenarios and all3difficulties, preserving upstream reader order.",
        selection_rule="All24official theme/difficulty strata; select120episodes(seeds0–4), else48(seeds0–1), else24(seed0) only from independently qualified timing/memory and remaining deadline/GPU-hour reserve. Never use qualification/production endpoint rates or favorable task selection.",
        panel_evaluator="audit_recoma_discoveryworld_v3.py",
        visible_census_contract=census_contract, visible_census_contract_sha256=sha256(args.visible_census_contract),
        maximum_completion_tokens_panel=len(tasks)*62*400, deadline_epoch=DEADLINE,
        scheduler_deadline=SCHEDULER_DEADLINE, scheduler_deadline_timezone="America/New_York",
        primary_trace_output_cap_bytes=2147483648, total_evidence_cap_bytes=4294967296,
        primary_outcome="Equal-weighted mean official terminal scoreNormalized over all24theme/difficulty strata, first averaging the prospectively fixed seeds in each stratum.",
        claim_boundary="Complete fixed-stratum baseline/model adaptation; timing fallbacks are not the full five-seed official comparator. No Jev effect, intervention effect, OOD performance, or published GPT-4o score reproduction.",
        parallel_shards={"count": plan["workers"], "assignment": "Source scenario order modulo worker count; each assigned scenario retains all3difficulties and frozen seeds; exact-ID whitelist in upstream reader order."},
        resource_planning={"frozen_at_epoch": now, "phase_gpu_hours_cap": PHASE_CAP, "already_allocated_phase_gpu_hours": spent,
            "requested_worker_cap":args.max_workers, "effective_qualified_partition_worker_cap":effective_worker_cap,
            "qualification_receipt_sha256": sha256(args.qualification_audit), "measurement_forecast": forecast,
            "selected_plan": plan, "considered_plans": considered,
            "tier_selection": "Largest complete allowed tier; then smallest adequate independent GPU worker count; only timing/memory/qualified record evidence.",
            "selection_without_outcome_rates": True},
        resources={"account":"pi_fl426", "qos":"normal", "partitions":partitions, "nodes":1,
                   "gpu_count":plan["workers"], "cpus":4*plan["workers"], "memory_gib":32*plan["workers"],
                   "wall_minutes":plan["requested_wall_minutes"], "gpu_type":placement["gpu_type"],
                   "node_list":placement["node_list"]})
    if isinstance(manifest.get("execution_adapter"), dict):
        manifest["execution_adapter"]["note"] = (
            "Official ReCoMA reader convenience variations are1–5; the explicit override uses the frozen production API seeds from0–4. "
            "A48/24episode timing fallback is a balanced stratum census, not the complete five-seed comparator. "
            "Backend, counter, context-envelope and scientific-failure accounting adaptations are disclosed; no published GPT-4o score reproduction is claimed.")
    require(inputs.joinpath("wheelhouse").is_dir(), "offline qualified wheelhouse required")
    output = args.output
    output.mkdir(parents=True)
    for name in ("source", "wheelhouse"):
        shutil.copytree(inputs / name, output / name)
    for name in ("run_bounded_recoma_worker_v3.py", "recoma_discoveryworld_react_full_v1.jsonnet",
                 "recoma_discoveryworld_hf_v1.lock",
                 "final_six_hour_validation_protocol_20260930.md"):
        shutil.copyfile(inputs / name, output / name)
    shutil.copyfile(inputs/"audit_recoma_discoveryworld_v3.py",output/"qualification_original_frozen_auditor.py")
    shutil.copyfile(args.qualification_auditor,output/"qualification_independent_auditor.py")
    shutil.copyfile(args.production_auditor,output/"audit_recoma_discoveryworld_v3.py")
    shutil.copyfile(qualified_path,output/"qualification_original_manifest.json")
    shutil.copyfile(args.qualification_audit, output / "qualification_audit.json")
    shutil.copyfile(args.visible_census_analyzer, output/"analyze_recoma_visible_instrument_census_v3.py")
    shutil.copyfile(args.visible_census_contract, output/"recoma_visible_instrument_census_v3.json")
    for path,name in ((args.model_checksums,"model_snapshot_SHA256SUMS"),
                      (args.model_verifier,"verify_model_snapshot_v3.py"),
                      (args.qualification_model_verification,"qualification_model_verification.json"),
                      (args.analysis_script,"analyze_recoma_discoveryworld_v3.py"),
                      (args.analysis_contract,"recoma_baseline_analysis_contract_v3.md"),
                      (args.analysis_tests,"test_recoma_discoveryworld_v3_analysis.py")):
        shutil.copyfile(path,output/name)
    if capacity:
        shutil.copyfile(args.capacity_receipt,output/"capacity_receipt.json")
        capacity_raw=output/"capacity-observations";capacity_raw.mkdir()
        capacity_pins={}
        for index,(filename,digest) in enumerate(sorted(capacity["input_sha256"].items())):
            target=capacity_raw/f"scheduler-{index}.txt"
            shutil.copyfile(filename,target)
            require(sha256(target)==digest,"scheduler observation changed while freezing capacity")
            capacity_pins[str(target.relative_to(output))]=digest
        manifest["capacity_route"]={"receipt_sha256":sha256(args.capacity_receipt),
            "raw_scheduler_observations_sha256":capacity_pins,"qualified_hardware_type":placement["gpu_type"],
            "requested_partitions":partitions,"expands_placement_only":True,
            "no_cross_hardware_throughput_transport":True}
    receipt_dir = output / "phase-accounting"
    receipt_dir.mkdir()
    for index, path in enumerate(args.phase_resource_receipts):
        shutil.copyfile(path, receipt_dir / f"parent-{index}.json")
    manifest["artifact_sha256"] = {name: sha256(output/name) for name in
        ("run_bounded_recoma_worker_v3.py", "recoma_discoveryworld_react_full_v1.jsonnet",
         "recoma_discoveryworld_hf_v1.lock", "audit_recoma_discoveryworld_v3.py", "qualification_audit.json",
         "qualification_original_frozen_auditor.py", "qualification_independent_auditor.py", "qualification_original_manifest.json",
         "analyze_recoma_visible_instrument_census_v3.py", "recoma_visible_instrument_census_v3.json",
         "model_snapshot_SHA256SUMS", "verify_model_snapshot_v3.py", "qualification_model_verification.json",
         "analyze_recoma_discoveryworld_v3.py", "recoma_baseline_analysis_contract_v3.md",
         "test_recoma_discoveryworld_v3_analysis.py")}
    (output / "panel_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
    batch = render_batch(args.batch_template.read_text(), args.remote_root, plan["workers"], plan["requested_wall_minutes"],
                         partitions, placement["gpu_type"], placement["node_list"])
    (output / "recoma_final_panel_v3.sbatch").write_text(batch)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(output)}\n" for p in files))
    return {"status":"FROZEN_NOT_SUBMITTED", "episodes":len(tasks), "gpu_count":plan["workers"],
            "wall_minutes":plan["requested_wall_minutes"], "planned_gpu_hour_ceiling":plan["requested_gpu_hour_ceiling"],
            "manifest_sha256":sha256(output/"panel_manifest.json"), "inputs_sha256":sha256(output/"SHA256SUMS")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-inputs", type=Path, required=True)
    parser.add_argument("--qualification-audit", type=Path, required=True)
    parser.add_argument("--qualification-auditor", type=Path, required=True)
    parser.add_argument("--production-auditor", type=Path, required=True)
    parser.add_argument("--visible-census-analyzer", type=Path, required=True)
    parser.add_argument("--visible-census-contract", type=Path, required=True)
    parser.add_argument("--model-checksums", type=Path, required=True)
    parser.add_argument("--model-verifier", type=Path, required=True)
    parser.add_argument("--qualification-model-verification", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--analysis-tests", type=Path, required=True)
    parser.add_argument("--capacity-receipt", type=Path, help="Optional hash-bound inspected normal same-family GPU capacity; no model qualification substitution")
    parser.add_argument("--phase-resource-receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", required=True, help="New immutable production root, including /inputs below it")
    parser.add_argument("--batch-template", type=Path, default=Path(__file__).resolve().parents[1]/"cluster/bouchet/recoma_final_panel_v3.sbatch")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--allow-48-fallback", action="store_true")
    parser.add_argument("--allow-24-fallback", action="store_true")
    print(json.dumps(freeze(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
