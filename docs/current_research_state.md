# Research continuation state

Updated September 27, 2026, after launching v3a. Broad curriculum remains open; no duplicate goal or automatic completion.

## Authoritative live dependency

`runs/branch-replication-v3a-20260927` is the active 24-problem/four-repeat development replication. Receipt: `runs/branch-replication-v3a-control/process.json`. Worker PID **57951**, start **Sun Sep 27 19:35:23 2026** local. Supervisor session 18896. Verify PID AND start time against the receipt; never infer termination from a stale log or restart a live job. Cooperative limit 14,400 seconds, external 14,550, sampled RSS 8 GiB. One local MLX model job only.

The first launch (`runs/branch-replication-v3-20260927`, control without `a`, PID 57919) failed before task scheduling/model inference/API dispatch because a relative config path was not resolved before source freezing. Exit 1 was verified. Preserve that failed directory/receipt. The tested path fix is commit 1dbfb8a. v3a uses the same frozen scientific config; it is not a replacement for observed outcomes.

Protocol: `docs/branch_replication_v3_protocol.md`; config: `configs/branch_replication_v3.json`. Same pools and downstream outcomes compare continue, uniform, likelihood, entropy, local semantic and Jev semantic. At most 24 hosted calls, central $1 cumulative development cap/$25 project hard cap. First call succeeded; last observed ledger: 56 successful requests total, $0.002324574. Query ledger for current spend; do not print responses or credentials. All selector decisions precede downstream continuations. Do not inspect partial success rates to alter the run.

At launch memory pressure reported 55% free and 44 GiB disk available. Hardware is sufficient for this bounded worker. Current limitations are evidence precision, causal intervention/controller qualification and complete published-baseline compatibility—not a missing theory proof or more memory.

## Completed evidence

Repair v2: 80 outcomes, 104 calls, zero audit disagreements; continue 11/16, segment repair 12/16 with wide uncertainty. Branch v2: 64 distinct outcomes, 126 calls, zero disagreements; continue 9/16, uniform 10/16, likelihood/entropy/local 11/16 each. No semantic increment or Jev efficacy established. Published reports and terminal releases are on GitHub. v1 prospective test remains closed to model selection/training.

Manuscript `paper/main.tex` includes both v2 diagnostics, standard conditional theory and pending v3 status; built-in compilation passed. Not arXiv-ready and no upload authorized.

## Terminal-run procedure

1. Check the recorded v3a process identity and terminal receipt/manifest. Do not restart a live or merely quiet run.
2. Run `scripts/audit_branch_qualification.py` with the pinned local tokenizer and a NEW `results/branch_replication_v3_audit.json`. It supports v2 and v3; hosted prompt hashes, choice parsing, chronology, labels and cost fields are checked. Raw hosted records remain private.
3. Only after audit passes, run `scripts/analyze_branch_qualification.py` with that report and a NEW `results/branch_replication_v3_analysis.json`. The analyzer adds Jev-vs-likelihood/local contrasts, acquisition cost and disjoint two-repeat half diagnostics. It was updated during collection without looking at partial outcomes, following the frozen protocol.
4. Independently review all outcome counts, failed/ineligible cases and selector disagreement. Reconcile actual ledger charges versus conservative per-episode acquisition costs. Publish an allowlisted release, update the paper, compile and push.
5. Advance the curriculum based on complete evidence; no automatic broad-project closure from either a small positive or negative result.

## Independent work completed / next qualification

The v2 terminal export reproduces all labels. Choice API validation and ordering/cost tests pass. GUARD's pinned trigger/ranking semantics now have CPU qualification tests (`docs/guard_qualification_contract.md`); this is NOT a full published comparator. Full integration requires predictive-logit/boundary handling and a separately qualified all-work runner, and cannot be claimed from common-pool entropy ranking. Qualify it before a prospective full-controller claim. Separate-model local judging, strong learned cheap representations, fresh rate-matched deployment and sequential/transfer tests remain in the curriculum.

## Recurring authorization

The existing 30-minute `jev-experiment-completion-check` heartbeat authorizes active research, not monitor-only work. Keep it enabled. Notify on meaningful findings/failure/resource needs. Goal mode is paused and cannot be programmatically reactivated; do not edit internal goal state. This does not block authorized research. While the run is live, avoid competing inference, changing its code/config, or making adaptive decisions from partial results.

## September 27, 19:56 heartbeat work

Verified worker 57951 with unchanged start time; still live. Without inspecting partial success rates, strengthened the independent hosted-record auditor with malformed/tampered-record tests (request leakage, temporal ordering, usage, distributions, cached accounting, failure fallback). Added missing-usage reporting so failed API usage is not silently treated as zero. Public releases may include usage/cost/time scalars but never raw hosted responses.

Implemented and CPU-tested `guard_adaptation.py`: a separately labeled segment-boundary variant, not the original GUARD method. It requires a backend/runtime qualification and independent call-ledger audit before deployment comparisons. Do not add it retroactively to v3 or compete with the live model job. Next qualification can begin after the current worker terminates and its result is audited.

## September 27, 20:26 heartbeat work

Worker 57951 still matched its recorded start time; progress reached seven completed problems without examining partial success rates. Added a frozen four-task runtime qualification config/runner (`guard_runtime_v1.json`, `run_guard_qualification.py`) with continue, segmented sham and GUARD adaptation. Do not launch concurrently with v3a. The sampler bridge restores state on failure; an independent segmented-episode auditor reconstructs prefixes, decisions and all-work costs. Full-run provenance/event/checkpoint/label auditing remains required beyond that helper. Protocol: `docs/guard_runtime_v1_protocol.md`.

Fixed the prospective adaptation's premature stopping on an unfinished FINAL: marker before any real-model qualification; this does not alter the running v3 code. Added regression tests for that case, changed call metadata to preserve exact requests and temperatures, and tested the complete runner with CPU fixtures. No real-model baseline outcome is claimed.
