# Receipt-grounded ReCoMA phase cost inventory v3

`scripts/build_recoma_phase_cost_inventory_v3.py` builds the terminal inventory consumed by the predeclared baseline analyzer. It reads scheduler/proof files and hashes failed call artifacts without opening task endpoints. Its purpose is actual-resource accounting, not a success-rate analysis, new model inference or new experiment authorization.

The prospective specification must list every receipt-discovered **parent** Slurm ID exactly once. Before production, the actual six qualification-related parents are27972098,27973858,27975076,27976130,27976321,27978394. Add every subsequently submitted CPU audit/production/deadline job when its submission identity is known. Do not hard-code a presumed number of submissions or count `.batch`, `.extern` or numeric steps as another allocation.

An independent local parse of the latest `qualification-phase-accounting.txt` establishes those six terminal allocation rows: three cancellations with zero allocation,113s failedH200,709s failedRTXPro6000Blackwell, and923s completedRTXPro6000Blackwell. Their allocation subtotal is **0.484722222222GPU-hours and1.938888888889 reservedCPU-hours**, before the separate CPU audit/production. These are allocation facts; no qualification or failed endpoint rates were read to compute them. A completed qualification is not a production scientific result.

## Scheduler receipt rules

Each job selects one authoritative pipe-delimited `sacct` file with explicit headers `JobID|State|ExitCode|ElapsedRaw|AllocTRES|Account|QOS|Partition`; Submit/Start/End are retained when available. Multiple exact parent rows, an absent parent, or a `.batch`/step identity are errors. Additional older receipts are archived/hash-bound but are not substituted for the selected terminal row.

The builder normalizes `CANCELLED by UID`, reconciles `gres/gpu` with the sum of typed GPU TRES, and counts the allocation once. Generic and typed TRES represent the same GPU allocation; they are not additive. CPU count and memory derive from AllocTRES. Account must be `pi_fl426` and actual/requested phase tier normal. Any supplied normalized resources file or passing audit's Slurm accounting must agree with the exact parent receipt.

Final costs require terminal records, including `DEADLINE`. An unallocated `CANCELLED`/`DEADLINE`, zero CPU/GPU/ElapsedRaw and no Start prove no allocation/generation. An allocated timeout/failure/deadline retains its resource time. A completed CPU-only audit uses `cpu_audit`, not `failed_attempt`; it contributes reserved CPU-hours and no invented GPU time.

`--snapshot` emits **`recoma_phase_cost_inventory_snapshot_v3`**, explicitly incompatible with the final analyzer's terminal schema. A running job's elapsed allocation is only an as-of lower bound; the snapshot never extrapolates future cost or opens that job's results. A final inventory requires completed independent audits/proofs for all model jobs; no snapshot is a substitute for missing evidence.

## Specification

Use absolute paths in a JSON object:

```json
{
  "schema": "recoma_phase_cost_spec_v3",
  "scope": "All receipt-discovered parents in the final six-hour phase",
  "expected_job_ids": ["JOB_ID"],
  "jobs": [
    {
      "job_id": "JOB_ID", "role": "failed_attempt",
      "accounting_file": "/absolute/latest_terminal_sacct.txt",
      "additional_accounting_files": ["/absolute/earlier_receipt.txt"],
      "resources_file": "/absolute/verified_resources.json",
      "generation_accounting": {
        "mode": "partial_logs", "all_attempts_recoverable": false,
        "call_files": ["/absolute/recovered_calls.jsonl"],
        "stress_files": [],
        "failure_explanation": "Recovered regular calls are a lower bound; failed/unlogged work remains unknown"
      }
    }
  ]
}
```

Optional `resources_file` supplies exact extra fields used by the existing auditor, but its core resource fields must match the scheduler receipt. For `mode=audit`, the builder uses the audit's exact `slurm_accounting` object so the final analyzer can bind identical resource metadata. Other mode definitions:

- **Passing GPU audit:** `role=qualification` or`production`; `mode=audit`, `audit_file`, `manifest_file`, explicit`expected_auditor_sha256`. The builder requires PASS and checks manifest identity/scheduler resources. The final analyzer additionally rechecks all bound trace/source/model-artifact hashes.
- **Import failure before generation:** `role=failed_attempt`; `mode=no_generation`, `runtime_report_file`, concrete`reason`, optional`additional_evidence_files`. The report must bind the job ID, `status=infrastructure_failure`, and`performed_generation=false`. Missing outputs alone are insufficient.
- **Unallocated scheduler termination:** `role=cancelled_unallocated`, or`failed_attempt` with`mode=no_generation`; scheduler proof must show zero CPU/GPU/elapsed and no Start. No nonexistent runtime report is required.
- **Failed run after generation started:** `role=failed_attempt`, `mode=partial_logs`, `all_attempts_recoverable=false`, explicit`call_files`/optional`stress_files` and`failure_explanation`. These files are bound as raw bytes here. The final analyzer counts recoverable call work as lower bounds and marks exact all-generation totals unknown; it never treats failed task rows as new scientific evidence.
- **CPU audit:** `role=cpu_audit`, `mode=no_generation`, `execution_receipt_file`, `batch_file`, `log_file`, concrete`reason`. The execution receipt must have matching`job_id`, status`completed` or`infrastructure_failure`, `performed_generation=false`, and integer-zero`model_generations`, `hosted_calls`, `jev_calls`. Allocated GPUs must be zero. Batch/log/receipt hashes preserve the no-generation proof. The audit's historical `model_calls` count is not a counter of new CPU-audit inference.

## Narrow qualification auditor correction

The original immutable qualification manifest pins the originally submitted auditor. Source-proven scalar/string and prompt-key-order serialization corrections may be evaluated afterward using a separately retained corrected auditor, without changing inference or outcomes. The backend hashes ordered role/content messages, while durable logging sorts nested keys; the corrected source-bound auditor reconstructs that original two-key order and still rejects unexpected message fields. Retain the original failed audits, including the separate CPU-audit failure. The production manifest instead prospectively pins the final corrected auditor and analysis hashes.

For qualification only, append this object to `generation_accounting`:

```json
{
  "qualification_auditor_correction": {
    "receipt_file": "/absolute/correction_receipt.json",
    "receipt_sha256": "EXACT_CORRECTION_RECEIPT_SHA256"
  }
}
```

Its hashed receipt has these exact fields:

```json
{
  "schema": "recoma_qualification_auditor_correction_v3",
  "original_manifest_sha256": "ORIGINAL_MANIFEST_SHA256",
  "original_auditor_sha256": "ORIGINAL_FROZEN_AUDITOR_SHA256",
  "effective_auditor_sha256": "CORRECTED_APPROVED_AUDITOR_SHA256",
  "correction_scope": ["numeric_scalar_json_serialization", "prompt_message_key_order_serialization"],
  "reason": "Source proves upstream scalar-string/numeric equivalence and role/content ordering before sort_keys logging; both official ratios and exact model messages remain unchanged",
  "generation_rerun": false,
  "task_assignment_changed": false,
  "inference_source_changed": false,
  "raw_inputs_modified": false,
  "scientific_endpoint_changed": false
}
```

The final analyzer rejects this exception for production, a changed endpoint/input/source/task assignment, a generation rerun, wrong manifest/auditor identity, unrecognized/partial correction scopes or an unbound correction receipt. Numerical string equivalence does not excuse Booleans, nonfinite/malformed numbers or incorrect official ratios. The combined string `numeric_predicted_and_prompt_key_order_serialization` is an accepted equivalent provenance tag; the original numeric-only tag remains for historical single-correction records.

## Execution

```text
python scripts/build_recoma_phase_cost_inventory_v3.py \
  --spec /absolute/final_phase_cost_spec.json \
  --output /absolute/new_terminal_phase_cost_inventory.json
```

Add`--snapshot` only for a non-final live allocation snapshot. Outputs refuse overwrite and record specification/builder/evidence hashes. Do not infer complete historical project or monetary costs from this phase inventory; the previous Mac compute/accounting limitations remain.

## Investment interpretation

This accounting repairs denominators and resource transparency; it is not evidence that Jev works. The scientifically defensible positive signal remains the audited cheap meter's useful conductivity measurement. Prior completed Jev comparisons have not established an incremental gain, and the meter's development-panel ceiling leaves no observed accuracy headroom there. A baseline execution or code/weight qualification cannot justify an open-ended compute extension by itself. Further investment requires an audited, nontrivial observable action/cost question and a fresh contrast against strong cheap/published alternatives. Missing or negative evidence must remain explicit at the phase deadline.
