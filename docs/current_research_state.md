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

## September 27, 20:56 heartbeat work

Verified worker 57951 at its original start time; eleven of 24 problems complete, with no interim success-rate inspection. Added `audit_guard_qualification.py` for the forthcoming four-task baseline job. It binds segmented calls and decisions to the durable generation ledger, checks decision-before-continuation chronology, frozen task/source/tokenizer identities, exact prefixes, costs and final labels. CPU tamper tests cover late/early decisions, delayed call recording and changed durable copies. Full real-run validation remains pending and is a prerequisite to calling the baseline qualified.

Also recorded a dated secondary v3 analysis amendment before completion: expected-uniform candidate selection, obtained by averaging all candidate outcomes rather than taking their maximum. The sampled-uniform baseline and primary comparisons remain unchanged. Added the standard conditional averaging derivation to the existing manuscript; native compilation passed. Do not conflate this secondary amendment with the original frozen primary protocol.

## Bouchet track authorized and submitted, September 27 at 21:21

User explicitly requested Yale GPU acceleration. Job **27713897** is submitted (not test-only), initially PENDING: one GPU across H100/B200/H200 eligible partitions, pi_fl426 normal QOS, 4 CPUs, 32 GiB RAM, 30 minutes. Local receipt `runs/bouchet-cuda-qualification-v1-control/submission.json` records remote directory and source hashes. Working route is `ssh -T mac-aux` followed by authenticated `ssh -T -o BatchMode=yes bouchet`; direct lead SSH currently requires interactive authentication. Read yale-bouchet-research skill before remote work. No further user action needed while the auxiliary connection persists.

On EACH research continuation inspect BOTH local worker 57951/start identity and Slurm job 27713897 through squeue/sacct. Do not resubmit a pending job or claim it ran merely from submission success. On termination retrieve the remote outputs/logs, verify resume/long-context/batch checks and actual GPU/runtime, then qualify a separately frozen CUDA research backend. The job's BF16 results cannot be pooled with local MLX 4-bit evidence. Details: `docs/bouchet_execution.md`. No Jev key was copied to Bouchet.

## September 27, 21:26 heartbeat and recurring-task update

Local worker 57951 remains live (15/24 problems at inspection). Bouchet original job 27713897 is terminal FAILED, 132:0, 23 seconds on B200; native hf_xet download SIGILL occurred before model inference. Retrieved logs and manifest are preserved. Corrected retry **27714091** is submitted with Xet disabled and otherwise unchanged 30-minute one-GPU resources. Its authoritative receipt is `runs/bouchet-cuda-qualification-v1b-control/submission.json`; monitor this job alongside the Mac worker, not the failed original. Use `audit_cuda_qualification.py` after retrieving complete outputs and successful sacct accounting. Do not infer success from allocation or imports alone.

At the user's explicit request, updated existing automation `jev-experiment-completion-check` in place to “Jev research — Mac and Bouchet”, ACTIVE at the same 30-minute cadence. The prompt now manages both tracks: queues, bounded failure recovery, correct PI/standard-tier use, result retrieval, independent audits, GPU-hour accounting and qualification-gated scale-up. It preserves quiet-on-unchanged notification behavior, the $25 Jev cap, frozen studies and the full research objective. No duplicate automation was created.

## Latest Bouchet identity: September 27, approximately 21:43

Monitor **27714861** and local worker 57951. Bouchet 27714091 is FAILED 132:0 after 1:42: download succeeded, then native tokenizer BPE deserialization raised SIGILL. The next attempt uses a Python Qwen2Tokenizer that passed six exact text/chat token fixtures on the login host before submission. Same pinned BF16 model, standard-tier account and 30-minute one-GPU envelope; no scientific rollout yet. Receipt: `runs/bouchet-cuda-qualification-v1c-control/submission.json`. Frozen fixtures accompany the job and are rechecked before GPU inference. The earlier failures remain preserved. Automation must discover this newest receipt rather than continue polling only 27714091. The user's instruction remains to continue after successful qualification into justified research jobs, not call infrastructure success project completion.


## Latest state: CUDA research-loop submission

Bouchet qualification **27714861 COMPLETED 0:0** on H200 in 3:35. Retrieved record audit passed (`results/bouchet_cuda_qualification_v1c_audit.json`): exact greedy resume, 8k context, batch generation; peak allocation 9.31 GiB, measured batch throughput 5.78 tokens/s. This establishes compatibility, not acceleration or scientific benefit.

Submitted next job **27717762** (actual submission), receipt `runs/bouchet-cuda-guard-runtime-v1-control/submission.json`, remote `runs/cuda-guard-runtime-v1-20260927`. One GPU, pi_fl426 normal tier, 4 CPUs, 32 GiB, 2 hours; fresh two-problem CUDA continue/segmented-sham/GUARD-adaptation qualification. No Jev credential. Frozen protocol `docs/cuda_guard_runtime_v1_protocol.md`. All 371 CPU tests pass. On completion retrieve complete records and run the independent guard qualification auditor before analysis/scaling. Local worker 57951 remains the separate frozen MLX study; do not pool BF16 and MLX results. Monitor this newest job, not completed or failed qualification attempts.


## September 27, 22:26 continuation

Job 27717762 COMPLETED 0:0, H200, 14:30 (0.2417 GPU-hours). Retrieved under `runs/bouchet-cuda-guard-runtime-v1-retrieved`; independent audit `results/cuda_guard_runtime_v1_audit.json` passed: two eligible tasks, six outcomes, 75 calls, two natural branch triggers, zero label disagreements. All three policies solved the same first task and failed the second; no benefit established. Initial retrieval requested nonexistent top-level requirements.lock, but the actual frozen requirements.lock.txt is preserved inside outputs/source; all audit-required records were retrieved and verified.

Next actual submitted job **27719471**, receipt `runs/bouchet-cuda-guard-development-v2-control/submission.json`, remote `runs/cuda-guard-development-v2-20260927`: twelve fresh development tasks/36 potential episodes, one GPU, four CPUs,32 GiB,four hours, pi_fl426 normal. Protocol `docs/cuda_guard_development_v2_protocol.md` frozen before generation. Monitor this job and local worker 57951 (original start identity verified). No pooling across backends. Audit terminal records before analysis; investigate timeouts and enrollment along with outcomes. 371 CPU tests passed.


## September 27, 22:56 audited replication and next local run

V3a worker 57951 exited 0 at 02:33:59 UTC. Full audit passed: 24/24 eligible problems,384 unique outcomes,677 calls,zero label disagreements. Public reports `results/branch_replication_v3_{audit,analysis}.json`; allowlisted release `results/branch_replication_v3_release` independently reproduced all labels. Continue48/96,uniform44/96,likelihood43/96,entropy44/96,local42/96,Jev42/96. Jev-minus-likelihood −1.04pp (exploratory problem-bootstrap −5.21 to +3.13pp); no efficacy/equivalence claim. Two disjoint repeat-selection diagnostics yielded no gain over uniform. Do not scale this fixed selector unchanged simply to pursue a positive result.

Next already-frozen local qualification launched: **PID14363, start Sun Sep27 22:57:12 2026**, receipt `runs/guard-runtime-v1-control/process.json`, outputs `runs/guard-runtime-v1-20260927`. Four fresh tasks,continue/segmented-sham/GUARD,2400s cooperative/2550s external,8GiB sampledRSS,no Jev. Audit with `audit_guard_qualification.py` after termination. CUDA job **27719471 is RUNNING on H100**, separate twelve-task development study. Monitor both new identities. Broad curriculum remains open; state-dependent intervention timing and fair baselines remain untested beyond these qualifications.
