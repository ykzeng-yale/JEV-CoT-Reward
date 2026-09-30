# Prospective complete-baseline analysis contract v3

Prepared while the separately frozen runtime qualification is running, before inspecting any production outcome. This contract describes analysis of the final six-hour ReCoMA/DiscoveryWorld adaptation; it does not approve another experiment, change task assignment, or establish Jev value. Freeze this document, `scripts/analyze_recoma_discoveryworld_v3.py`, its tests and the approved v3 auditor hashes in the production manifest before generation. The analyzer refuses a production manifest lacking its exact script hash.

## Gate and primary estimand

Analyze only a terminal `PASS` independent audit of the complete prospective `full_panel` or `balanced_census`: all 24 scenario/difficulty strata, each with exactly the frozen seeds. The allowed tiers are 120 episodes/seeds0–4, 48/seeds0–1 or24/seed0. Timing-only fallback selection is frozen separately. An incomplete or infrastructure-aborted panel has no efficacy rates here. Runtime-only seed5 qualification outcomes and context stress are never pooled into the production denominator.

Before reading endpoints, recheck the approved auditor SHA, frozen manifest SHA, every audit-bound record/call/event/stress/runtime-receipt hash, all source hashes and the original runtime source inventory's exact recorded path/hash. For the production model-content contract, recheck the exact original checksum/verifier/before/after paths against their approved artifact hashes; weight verification establishes frozen bytes and applicable HF blob identity, not an authenticated remote commit tree or scientific efficacy. Retokenization, source semantics and failure/output classification remain the independent auditor's responsibility; this analysis additionally recomputes raw `score/maxScore`, task coverage, per-task and total endpoint/cost arithmetic. Upstream `all_data.predicted` serializes a numeric scalar as a string while durable task records parse it to a number: independently require each value to equal the raw official ratio, rejecting Booleans, malformed/nonfinite values and incorrect scores. Metadata must match exactly; serialization equivalence does not excuse an endpoint discrepancy. Remote receipt paths may be relocated with an explicit absolute-prefix map. Never guess a file by its basename.

The primary descriptive endpoint is mean official raw normalized progress: calculate the mean over assigned seeds within each scenario/difficulty stratum, then equally average all24 strata. Balanced rows imply the same whole-panel mean. This is a finite benchmark description of the stated model/backend/controller/budget adaptation; it is neither the published larger-model result nor a treatment comparison.

Always report separately:

- Official partial progress, preserving actual partial progress even after a scientific format failure.
- Official `completed` and `completedSuccessfully` counts, as distinct scorecard fields.
- Failure-adjusted success: official success **and** no typed scientific format failure.
- All four declared format-failure counts, unsuccessful terminal rows, and all costs in the full assigned denominator. No conditional-on-survival rate replaces the full-panel rate.

Only `missing_action_json`, `invalid_action_json`, `non_object_action_json` and `invalid_submit_arguments` are format failures under this contract. A valid action object denied by the environment is not silently converted into a format failure. Generated text-format evidence is not evidence of hidden CoT access or intervention usefulness.

## Cap diagnostics and grouping

Display environment-action-cap reached, model-call-cap reached, and uncompleted/no-format-failure at either frozen cap. The latter is an operational budget-exhaustion diagnostic. Durable metadata contains no explicit exclusive termination reason, so reaching a cap does not establish that it was the sole cause of termination. Do not label `scientific_task_status=completed` as successful scientific completion: it means the policy ended without a typed format error.

Provide complete tables by all24 strata, eight scenario families, three difficulty levels and each assigned seed. Seed tables describe the same fixed panel; they are not independent studies. Sum environment actions, every model call, full input/prefill tokens, all generated output-token IDs, final malformed-output tokens, model-service seconds and task-wall seconds. Tokens discarded by stop handling or unusable final generations remain charged through the full output-ID count. A separate exact useful/discarded-token partition is not inferred from text or regenerated tokenization.

Task wall-time sum and model-service-time sum are parallel-work totals, not user-facing end-to-end latency. Parent Slurm elapsed allocation includes loading, idle workers, environment work and failures; do not replace reserved GPU-hours with the model-service-time fraction.

## Descriptive variability, not population inference

The complete finite census has exact observed descriptive means. There is no paired treatment effect, causal benefit, Jev increment, OOD/transfer conclusion or population confidence interval. Five seeds share a template and must not be counted as independent populations.

An explicitly labeled sensitivity display reweights the **eight observed scenario families**, retaining every difficulty and seed together:20,000 draws with replacement of eight family means, RNG seed20260930, middle95% range. This is descriptive sensitivity to the observed family mix. It is not uncertainty for the observed census, new templates, future tasks or published-model replication. The output field says `population_confidence_interval=false`. Resampling24 difficulty strata as independent families would understate their shared-template dependence; this analysis does not do that.

## Final-phase total cost inventory

Provide `--phase-cost-inventory` to report final-phase totals, including production, passing qualification/stress, failed attempts and unallocated cancellations. Without that inventory, the output explicitly reports production-only allocation and marks total allocation/generation completeness false. The phase ledger must enumerate every receipt-discovered parent Slurm job identity exactly once; never count `.batch` or `.extern` as another allocation. Include every actual submitted qualification, superseded unallocated request and failed attempt; reconcile exact receipt-discovered parent IDs.

For every parent job, provide terminal `state`, `exit_code`, actual allocated `gpu_count`, `cpu_count`, `elapsed_seconds`, PI account and requested/actual tier, bound to archived `sacct`/submission/process receipt hashes. Compute actual GPU-hours as elapsed×allocatedGPUs/3600 and reserved CPU-hours similarly. Requested pending GPUs are not allocated GPU-hours. An unallocated cancellation or `DEADLINE` has zero allocation and requires evidence of no generation; any allocated time before `DEADLINE` remains charged. CPU-only audit jobs retain honest `cpu_audit` roles, zero GPUs, their reserved CPU-hours and hash-bound batch/log/execution proof of no new generation. No claim is made about all historical Mac compute, electricity or monetary costs.

Use one of three generation-accounting modes:

- `audit`: a hash-bound passing independent audit and its exact manifest. Qualification costs include the separately logged hardware stress; their outcomes remain excluded. The production ledger entry must bind the same audit being analyzed.
- `no_generation`: hash-bound logs/receipts, explicit `generation_proved_not_started=true`, and a concrete reason. Missing output alone is insufficient evidence that no generation occurred.
- `partial_logs`: a failed attempt's hash-bound available call/stress logs, `all_attempts_recoverable=false`, and a concrete failure explanation. Account recovered calls/tokens/service as lower bounds and flag unrecoverable costs. Never substitute zero for unknown work. A stress call that fails before returning any output may have submitted32,768 input tokens without completing prefill: record those attempted tokens separately, not as certified processed-token lower bounds. Any such job makes exact all-generation totals unavailable (`generation_costs_exact_if_complete=null`).

The analyzer validates file hashes, job uniqueness, normalized resource arithmetic and matching passing-audit resource records. The lead remains responsible for independently verifying that normalized resource fields agree with the archived scheduler receipts and for discovering the phase inventory exhaustively. A hash alone is not a second scheduler query or an independent source of resource semantics.

Minimal ledger structure (replace illustrative values with verified records):

```json
{
  "schema": "recoma_phase_cost_inventory_v3",
  "scope": "Receipt-enumerated final six-hour jobs",
  "expected_job_ids": ["PRODUCTION_ID", "QUALIFICATION_ID", "CANCELLED_ID"],
  "jobs": [
    {
      "job_id": "PRODUCTION_ID",
      "role": "production",
      "resources": {
        "job_id": "PRODUCTION_ID", "state": "COMPLETED", "exit_code": "0:0",
        "account": "pi_fl426", "requested_qos": "normal", "qos": "normal",
        "actual_qos": "normal", "elapsed_seconds": 600, "gpu_count": 2, "cpu_count": 8
      },
      "accounting_receipt_sha256": {"/absolute/sacct.txt": "RECEIPT_SHA256"},
      "generation_accounting": {
        "mode": "audit", "audit_file": "/absolute/production_audit.json",
        "audit_sha256": "AUDIT_SHA256", "manifest_file": "/absolute/manifest.json"
      }
    }
  ]
}
```

The full ledger includes a job object for each expected identity; the abbreviated example intentionally cannot pass as written. `role` is one of `production`, `qualification`, `failed_attempt`, `cancelled_unallocated`, `cpu_audit`. For a qualification audited with a different approved v3 auditor revision, supply its explicit `expected_auditor_sha256` and `qualification_auditor_correction={receipt_file,receipt_sha256}` inside that job's generation-accounting object. The correction receipt must bind original/effective auditor and original manifest hashes, a source-proven reason, and all five unchanged-science flags false: generation rerun, task assignment change, inference source change, raw input modification, scientific endpoint change. The final combined `correction_scope` is the exact two-element list `[numeric_scalar_json_serialization,prompt_message_key_order_serialization]`: upstream scalar/string representation and logger key sorting do not alter model inputs or scientific endpoints. The earlier numeric-only receipt tag is retained for historical single-correction records. This exception is qualification-only; a production auditor must match its prospectively frozen manifest. Preserve the original failed audits. `no_generation` uses `evidence_sha256` (path→hash) and `reason`; `partial_logs` uses `call_sha256`, optional `stress_sha256`, and `failure_explanation`.

## Execution and interpretation

Run remotely in the pinned CPU runtime after strict audit or locally after explicit prefix relocation; no model loads or API calls occur:

```text
python scripts/analyze_recoma_discoveryworld_v3.py \
  --manifest /absolute/frozen_manifest.json \
  --audit /absolute/passing_independent_audit.json \
  --source-root /absolute/source/discoveryworld \
  --expected-auditor-sha256 APPROVED_FROZEN_SHA256 \
  --phase-cost-inventory /absolute/phase_cost_inventory.json \
  --output /absolute/new_complete_baseline_analysis.json
```

Optional `--path-map /absolute/path_map.json` is a JSON object of absolute remote-prefix→local-prefix mappings. Outputs refuse overwrite and carry script, manifest, independent-audit, source/input and inventory hashes. Rates and costs do not authorize adaptive prompts, task selection, extrapolated success claims or another experiment beyond the phase boundary.

This baseline can supply a missing published-controller adaptation, complete failure/cost census and visible trace inventory. Strong scores or frequent thought fields alone establish neither a new useful intervention nor Jev value. An unsuccessful or malformed baseline establishes behavior under the frozen budget/adaptation; it does not refute every research direction. The final investment decision must incorporate the separate audited meter utility, previous negative/imprecise Jev comparisons, baseline completeness and remaining causal/novelty requirements.
