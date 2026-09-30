# ReCoMA scientific failure and durable output contract v3

The v3 adapter fixes evaluation attrition when the frozen ReAct controller cannot parse its generated action. It records that task as a scientific failure without another generation, repair, fallback action, or resampling. It preserves the upstream JSON extractor and all valid object actions, prompts, model settings, stopping limits, simulator code, official scorecards, and ordinary environment-denied actions. Runtime qualification is separate from this CPU validation; these changes establish no model capability or Jev benefit.

## Failure classification

Only four declared model-output conditions end a task: no JSON found by the upstream extractor (`missing_action_json`), invalid extracted JSON (`invalid_action_json`), a parsed non-object (`non_object_action_json`), or `SUBMIT` with a non-string/missing answer or non-string thought (`invalid_submit_arguments`). The record binds the exact raw controller output with SHA256 and sets `retry_or_resample=false`. A scientific failure fetches the official terminal scorecard once after stopping the policy. It retains every already-recorded model call and token count.

Unexpected internal types, simulator errors, CUDA failures, imports, filesystem errors, output-cap breaches, and other unknown exceptions abort the worker as infrastructure failures. They cannot become scientific outcomes. Earlier durable ended-task rows remain available; the incomplete task has an `infrastructure_abort` event and no fabricated score. Existing upstream handling of `RecursionError` is unchanged.

## Durable records and endpoint semantics

`task_records.jsonl` is the per-task durable authority. Each row is flushed and fsynced before the next inference, with exclusive ledger creation, no resume/overwrite, unique task identity, and a byte cap. `task_events.jsonl` records starts, ends, and infrastructure aborts. Defaults are 32 MiB and 4 MiB, respectively, and may be bounded explicitly through `RECOMA_TASK_RECORDS_MAX_BYTES` and `RECOMA_TASK_EVENTS_MAX_BYTES`.

Each ended-task row contains `task_id`, `scientific_task_status`, `task_wall_seconds`, the executed `runtime_contract` and module SHA256, `predicted`, and unmodified terminal metadata including `final_scorecard`, `num_steps`, and model usage. Raw official normalized partial progress is retained. `failure_adjusted_completed_successfully` requires both official completion success and absence of a scientific failure; a malformed policy output cannot inherit task success. Analyses must display raw progress, scientific failures, and failure-adjusted success separately. Original `all_data.jsonl` is still produced on normal worker completion for compatibility.

Per-call audit rows retain full prompt/output/token information and add the resolved pinned `model_revision`, `controller_output_sha256`, synchronized generation elapsed time, and CUDA peak allocated/reserved bytes. The controller receives `output_text.lstrip()`; its hash is distinct from the raw decoded `output_text` when leading whitespace exists. The independent auditor must verify the final malformed call and declared failure code, not accept arbitrary bad intermediate calls.

## Frozen delivery and validation

The builder reads `runs/bouchet-recoma-discoveryworld-full-v2-control/artifacts/inputs/source/` and refuses existing output/patch destinations. The canonical delivery is `runs/recoma-runtime-v3-delivery-control/source-v3/`, its `SHA256SUMS`, `build-receipt.json`, and reproducible `source-v3.tar.gz`. Prior packaging attempts remain preserved separately.

- Patch: `patches/recoma_scientific_failure_accounting_v3.patch`; SHA256 `d5670d775f00766606b332a4de966c09b3a7f496e9bab59a1e35c343067b8b91`.
- Source manifest SHA256: `6232e7cde251d10e788b1cacdc6f17c1d3d33c32168edb73277a0d7b30e5d906`.
- Archive SHA256: `336f8033ae3549c62c24f963d3ea148de11aa6d91cd2142b565cca0ade90b903`.
- Executed accounting module SHA256: `4a2b2d47972fa908053d7c8513c1133479a8f30b911e34a5677ab1fa97e1b068`.

Fault-injection tests execute the actual rendered controller/search/accounting source. They cover normal success, malformed JSON, typed failure binding, unchanged valid actions and denials, one-call failure termination, token retention, official partial scores, durable ordering, infrastructure aborts, caps, overwrite refusal, and actual patch application with all changed source bytes compared. CPU tests do not qualify full-context CUDA memory, throughput, simulator task execution, or outcome validity. Those require the separately frozen one-GPU qualification and independent record/endpoint audit before any bounded production scaling.
