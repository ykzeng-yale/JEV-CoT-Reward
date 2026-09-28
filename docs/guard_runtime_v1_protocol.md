# Segment-boundary GUARD runtime qualification v1

Specified September 27, 2026 before generating any qualification tasks or examining v3 outcomes. This is an instrumentation study, not a performance comparison or faithful original GUARD reproduction.

Run `scripts/run_guard_qualification.py` only after the active v3a worker terminates and its evidence is audited. Use the existing pinned local Qwen3-4B-Instruct-2507 snapshot, one worker, cooperative 2,400-second limit, external 2,550-second limit and 8 GiB sampled RSS bound. No Jev requests or downloads. Preserve failed attempts rather than overwriting their receipts.

`configs/guard_runtime_v1.json` fixes four fresh development tasks (seed 991027), one repeat and three policies: standard continuation, segmented sham and segment-boundary GUARD adaptation. Preserve all ineligible cases without replacement. All policies start from the same natural online checkpoint and receive a total 2,048-token generation allowance including initial and discarded work. This tiny study qualifies execution and accounting; it cannot establish superiority.

The sham follows the same 64-token pause/resume schedule, final instruction and sampling as the adaptation, with branching disabled. This separates segmentation artifacts from the extra branch mechanism. Standard continuation preserves the existing rollout interface and final instruction; its different finalization is recorded rather than assumed equivalent. Any later main comparison must decide and freeze a shared finalization contract.

The adaptation uses last-emitted-token entropy, a current-inclusive 0.90 quantile, and heterogeneous candidates. This differs from original GUARD's predictive boundary trigger. All three candidates consume budget regardless of selection, and candidate instructions remain in the selected context. Prompt insertion is counted through actual processed prompt lengths, not as model-generated output. Sampler and logging context restore even on exceptions. EOS/timeout ends a path; the text marker FINAL: alone does not imply a complete answer and does not prematurely stop generation.

## Required audit and decision

CPU tests check trigger opportunity, temperature restoration, ineligible preservation, exact token-prefix reconstruction, discarded-work charging and rejection of altered seeds/budgets/choices. `scripts/audit_guard_episode.py` reconstructs segmented episodes independently from their saved calls. The complete run additionally needs source/tokenizer identity checks, initial checkpoint audit, association with durable generation events, original final-verifier recomputation and timestamp ordering across calls/decisions. Do not label the full run audited based solely on the episode checker.

Report all scheduled tasks, actual trigger counts, failures, cap utilization, service time and memory. A run with zero triggers can qualify some runtime paths but cannot qualify real-model branch execution; add a separately labeled forced-trigger instrumentation test if needed, never report it as naturally selected efficacy. Freeze that qualification before execution. Only after runtime and complete-record audit pass should the adaptation enter a fresh outcome study. No pending v3 result is used to tune these settings.

## Complete-run auditor prepared

`scripts/audit_guard_qualification.py` now composes the episode checker with frozen task/rollout source contracts, tokenizer metadata identity, checkpoint reconstruction, durable call matching, selector decision chronology and independent outcome labels. It refuses incomplete runs, missing/extra calls and duplicate outcomes. The weights' saved revision digest is checked; this is not fresh inference or a fresh weight-file rehash. After the future runtime job ends, write a new report and retain the original evidence:

```sh
.venv/bin/python scripts/audit_guard_qualification.py RUN_DIRECTORY \
  --tokenizer PINNED_LOCAL_SNAPSHOT --output results/guard_runtime_v1_audit.json
```

This auditor has CPU component tests; its first complete real-run exercise is still pending. Do not claim the runtime qualified until that audit succeeds.
