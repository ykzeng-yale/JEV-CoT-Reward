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


## September 27, 23:26 completed audits and randomized baseline

Local14363 completed0 in455s; audit reconciled12outcomes/153calls/3triggers: all policies3/4. CUDA27719471 completed0 in23:56 (0.3989GPUh); audit reconciled36outcomes/430calls/3triggers: continue5/12,sham6/12,GUARD6/12. No advantage over sham. Reports `results/guard_runtime_v1_audit.json`, `results/cuda_guard_development_v2_audit.json` and corresponding summaries. Audit initially rejected harmless mean-entropy reduction roundoff; fixed by independently validating the stored mean then using it for exact runtime decision reconstruction. Tamper/roundoff tests pass. Original records unchanged. Missing AppleDouble frozen-source files were recovered byte-for-byte over SSH; all frozen hashes passed.

New actual job **27724738**, receipt `runs/bouchet-cuda-guard-random-v3-control/submission.json`, outputs remote `runs/cuda-guard-random-v3-20260927`. Twelve fresh tasks/four policies add random branching with probability3/55 per eligible boundary, frozen from completed v2 opportunities. Same budget, ranking, generation and finalization as GUARD. This matches opportunity rate in expectation, not realized costs/counts. Protocol `docs/cuda_guard_random_v3_protocol.md`. One standard GPU,4CPU,32GiB,4h. 375CPUtests passed. No live local model now; previous jobs terminal, never restart. Next: full audit and paired timing comparison after terminal, then choose controller study based on complete results.

## September 27, 23:56 analysis preparation

Job27724738 verified RUNNING on H100 (24:29 elapsed); no partial outcomes examined or duplicate submitted. No local model worker is live. Added audit-bound `analyze_guard_qualification.py`, tested stale-audit rejection and shared-prefix accounting. Completed v2 analysis records one continuation timeout versus none for sham/GUARD; the latter both6/12. GUARD processed227210 prompt tokens vs sham209329. A continuation advantage is not a branching benefit and is confounded by different per-call timeout exposure. Retain historical records; freeze a uniform episode/runtime contract before prospective efficacy claims. Running random-vs-GUARD shares segmentation and remains useful. See `docs/paper_evidence_checklist.md` for missing paper evidence and conditional next decision. After terminal audit, run the new analyzer with a fresh report path.


## September28,00:26 randomized result and precision stage

Job27724738 completed0 in26:17 (0.4381GPUh). Retrieved `runs/bouchet-cuda-guard-random-v3-retrieved`; full audit passed48outcomes/567calls,zero disagreements. Audit field natural_triggers totals both policies:2GUARD and2random. GUARD/random/sham each8/12;continue10/12 with one call timeout. GUARD-minus-random0pp,descriptive paired interval−25 to+25pp,two discordant problems. Reports `results/cuda_guard_random_v3_{audit,analysis}.json`. No efficacy/equivalence.

Actual submitted job **27727778**, receipt `runs/bouchet-cuda-guard-precision-v4-control/submission.json`, remote `runs/cuda-guard-precision-v4-20260928`:96fresh tasks/384 potential policy episodes, unchanged actions,random3/55,primaryGUARD-vs-random. Protocol `docs/cuda_guard_precision_v4_protocol.md` frozen before generation. Planning halfwidth8.2pp under observed discordance,not3pp power. Measured linear cost3.50GPUh; bounded1GPU/4CPU/32GiB/8h,pi_fl426 normal. 377tests passed. No local model live. Audit and analyze on terminal; do not adapt from partial rates. This is substantive precision development,not another backend qualification or final test.


## September28,00:56 independent analysis/theory work

27727778 verified RUNNING on B200,26:09 elapsed; no interim outcomes inspected. Local model workers remain terminal. Added tested episode-level timeout flags and worst-case completion-sensitivity envelopes to the guard analyzer. The envelope assumes non-timeout outcomes unchanged under hypothetical completion; it is not a confidence interval or a correction for unequal deadlines. Applied only to completed audited v3 as a separately named secondary report; original reports preserved. Added proof and assumption boundary to paper/main.tex. This does not change v4 primary endpoint/design; apply as secondary diagnostic after terminal audit.

## September28,01:26 provenance hardening

27727778 still RUNNING on B200,56:08 elapsed. No partial success rates read, no duplicate jobs or local inference started. Added audit/analysis dependency hashes so imported helper changes (including the prior floating-point checker correction) are identifiable, rather than hashing only the entry script. Sixteen targeted tests passed. Added `docs/guard_result_acceptance.md` for terminal source/receipt/accounting checks and frozen-primary versus secondary sensitivity reporting. Existing reports/remote sources remain unchanged; use fresh reports at completion.

## September28,01:56 uncertainty safeguard

27727778 remains RUNNING on B200,1:26:09 elapsed; no interim outcomes inspected. Added simultaneous two-sided Hoeffding intervals across reported contrasts to supplement potentially degenerate small-sample bootstrap intervals. Assumptions and conditional enrollment target are explicit; original primary contrast remains unchanged. Five analyzer tests pass, including zero-discordance uncertainty. No new model job is justified before the current precision-stage result; existing theory/provenance/cost preparation is complete for its next decision.


## September28,12:34 completed precision study and repair replication

27727778 COMPLETED0:0 in1:33:37 (1.5603GPUh). Retrieved `runs/bouchet-cuda-guard-precision-v4-retrieved`; audit passed96eligible/384outcomes/4583calls/zero label disagreements. Continue54/96,sham66/96,GUARD61/96,random59/96. GUARD-minus-random+2.08pp (descriptive−6.25 to+10.42);GUARD-minus-sham−5.21pp (−12.5 to+2.08). GUARD37branches,random14; not realized-rate matched. One continuation timeout. Reports `results/cuda_guard_precision_v4_{audit,analysis}.json`. No semantic efficacy, no equivalence, no positive branching claim. Stop scaling this exact timing policy unchanged.

Launched fresh repair replication **PID79898,start Mon Sep28 12:34:41 2026**, receipt `runs/action-replication-v3-control/process.json`, output `runs/action-replication-v3-20260928`. Config `action_replication_v3.json`,protocol `docs/action_replication_v3_protocol.md`:24freshproblems,4repeats,5qualifiedactions,up to480outcomes,4h local bound,no Jev. Primary segment-repair-vs-sham and disjoint-repeat action preference diagnostic. No cluster job remains live. 380tests passed before launch. Auditor extended to the new frozen design; never modify generation source in the run directory. Next terminal audit then action replicability, not noisy winner training.

## September28,14:10 repair analysis preparation

PID79898 verified live with original12:34:41 start,elapsed1:36:18. No partial outcome rates read. Prepared frozen-protocol segment-repair-minus-sham analysis and disjoint two-repeat action selection/evaluation diagnostics. Tests prove evaluation outcomes cannot select the action, including ties (frozen action order). This is an outcome-informed diagnostic,not deployable controller or oracle bound; reversed splits overlap. No extra API or model work. Cluster precision run remains terminal. Analyze only after complete audit; next decision depends on repeatable action usefulness.

## September28,16:11 bounded-run progress and recovery preparation

PID79898 still matches MonSep28 12:34:41 start,elapsed3:37:19. Inventory420/480 outcomes,21problems with records; no partial success inspected. Keep original4h cooperative/14550s external bound. Prepared `docs/action_replication_v3_recovery.md`: complete auditor must reject incomplete data, preserve unmatched-call costs, missing-arm recovery needs exact checkpoint/action/seed and independent merge audit; never silently promote complete cases or restart live worker. Next check terminal receipt before any recovery. No new Jev expense or concurrent model job.
