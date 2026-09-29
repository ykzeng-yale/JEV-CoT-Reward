# Research continuation state

## September 29, 04:35 EDT — current handoff

**Highest-impact bottleneck:** no methods-level gap is currently established. *Calibration Is Not Control* directly defines action-conditioned intervention advantage, same-prefix branching, and prefix-feature action-conditioned policies; *CausalFlow* performs outcome-changing counterfactual step repair. *interwhen* and adaptive verifier allocation further cover online process verification and compute control. The manuscript has been corrected to acknowledge direct duplication of our prior proposed estimand/protocol. Only a possible Jev-versus-local judgment-source empirical comparison remains, and that is not yet a validated novelty claim. See `docs/prior_art_gap_audit_2026-09-29.md`.

**Completed local study:** H8 process PID 56403, start Tue Sep 29 02:18:25 local, receipt `runs/arithmetic-horizon-opportunity-v1-control/process.json`, terminal exit 0, 714.876 seconds, sampled RSS peak 528.5 MiB. Independent report `results/arithmetic_horizon_opportunity_v1_audit_v2.json` passed 120/120 record/task/witness checks. Eligible 120; explicit first-addition claims 41; incorrect 15 (36.6%, Wilson 95% 23.6–51.9%); incorrect among all eligible 12.5% (7.7–19.6%). The frozen continuation gate was 20 errors; it failed. There were 120 generation calls, zero terminal task outcomes, and zero Jev calls. Secondary post-hoc parser sensitivity found 12 standalone-equation errors across 143 claims in 11 checkpoints; it does not replace or pass the primary gate. Arithmetic-construction correction efficacy will not be run under this design.

**Baseline feasibility:** checked *interwhen* at upstream commit `2bc515f0c13a7a81b3f72a93958280c60231573c` (MIT). Its official task examples use `microsoft/VISION_LANGUAGE` Maze and `nlile/24-game`, with an OpenAI-compatible vLLM server. The lead host has the pinned MLX model but no local `vllm` or `datasets`; the local HF dataset cache is only 7.4 MB. Bouchet has Apptainer but no vLLM module; its live `squeue` contained no `jev`, `interwhen`, or `maze` jobs. Thus the published baseline is identified and its source inspected, but is **not yet executed or runtime-qualified**. A reimplementation using our current MLX runner must be labeled an adaptation. Bouchet reports 19 TB free at filesystem level and 126 GB user quota; a model download still requires project-path quota/cache and runtime qualification before submission.

**Jev constraint:** current TypeSafe MCA §2.3(b) restricts use of Service/Output to train imitation models or develop/facilitate a similar or competing service. This potentially covers the planned Jev-output-conditioned controller. We have made no new Jev requests. Hold that dependent training/product experiment and any release of such a controller pending written clarification; this is an operational contract constraint, not legal advice. H8 used no API call and booked no Jev spend.

**Next step:** no model process or JEV-owned Bouchet job is live. No new inference is justified until (1) a systematic comparison confirms a defensible incremental empirical contribution despite *Calibration Is Not Control* and *CausalFlow*, (2) TypeSafe clarifies whether Jev outputs may be used for this academic controller evaluation and what may be published, and (3) we freeze a compatible task/action/budget/sample-size protocol. The official *interwhen* path is a useful comparator but is not runtime-qualified on this Mac or Bouchet yet; any port must be called an adaptation. Closed v1 data stay untouched. Existing paper source compiled successfully in the built-in editor; 22 focused theory/baseline/task regression tests passed. No positive efficacy, Jev increment, OOD transfer, or discovery claim is supported.

## September 29, 02:20 local — current handoff

The six-number arithmetic-opportunity screen terminated successfully: local PID 50440, start Tue Sep 29 00:18:17, exit 0, 726s elapsed, sampled RSS peak 547.8 MiB. Independent audit `results/arithmetic_opportunity_v1_audit_v2.json` passed 120/120 exact task/call/prefix records, with zero ineligible tasks, zero judge calls and zero terminal outcome evaluations. It found 60 explicit first-addition claims, 50 correct and 10 incorrect: 16.7% among explicit claims (Wilson 95% interval 9.3–28.0%) and 8.3% among all eligible checkpoints (4.6–14.7%). The frozen advancement gate was 20 incorrect claims; it was not met. No correction rollouts were run.

Following that gate, a separate fresh longer-horizon feasibility screen is now running: PID **56403**, start **Tue Sep 29 02:18:25 2026** local, receipt `runs/arithmetic-horizon-opportunity-v1-control/process.json`, output `runs/arithmetic-horizon-opportunity-v1-20260929`. It samples 120 eight-operand expression tasks under the same Qwen3-4B-Instruct-2507 MLX 4-bit checkpoint and prefix budget. The arithmetic witness is excluded from the prompt and checked independently by the existing AST/Fraction verifier. The synthetic task generator passed CPU validation on all 120 unique tasks; every witness reached the target, and none leaked into its prompt. No outcomes or Jev calls are collected. Protocol/config: `docs/arithmetic_horizon_opportunity_v1_protocol.md`, `configs/arithmetic_horizon_opportunity_v1.json`; external bound 7,350s, sampled RSS 8GiB, logs 32MiB. This is a targeted horizon redesign because the simpler task produced too few eligible errors. On terminal status, run the exact-prefix and witness audit before applying the predeclared 20-error gate. Do not duplicate the live process or inspect interim event rates.

Paper includes the audited six-number feasibility result and the built-in editor compile passed. There is no active Bouchet job; this bounded local task fits the qualified model and avoids unnecessary cluster queue/storage overhead.

## September 29, 00:20 local live study

After the independent action-availability diagnosis, a fresh 120-task feasibility screen is running locally. Worker PID **50440**, start **Tue Sep 29 00:18:17 2026** local, process receipt `runs/arithmetic-opportunity-v1-control/process.json`, output `runs/arithmetic-opportunity-v1-20260929`. It uses the pinned local Qwen3-4B-Instruct-2507 MLX 4-bit snapshot, one 256-token paragraph checkpoint per fresh arithmetic-construction task, maximum 384 prefix tokens, no Jev calls and no terminal outcome collection. Frozen protocol/config: `docs/arithmetic_opportunity_v1_protocol.md`, `configs/arithmetic_opportunity_v1.json`. Cooperative 7,200s, external 7,350s, sampled RSS cap 8GiB, log cap 32MiB. Verify PID and start identity; do not restart or inspect partial event counts. The isolated rerun attempt collided with the existing immutable control directory and exited before spawning a child; the original PID remains authoritative.

The independent auditor is prepared at `scripts/audit_arithmetic_opportunity_screen.py`; after terminal completion verify every task, call, exact retained prefix and equation parser, then calculate a Wilson interval over eligible task checkpoints and bounds including ineligible tasks. The preregistered gate is at least 20 incorrect explicit equations among eligible checkpoints. Crossing it permits only a fresh, matched correction/sham development study. Falling short means redesign the task/checkpoint mechanism. This is a feasibility screen, not evidence of judge quality or policy efficacy. No competing local inference or new Bouchet job is needed while this appropriately bounded screen runs.

## September 29, 00:16 local decision update

The targeted correction experiment and its independent audit are complete (see September 28, 22:15 entry). A new frozen-checkpoint availability audit, `results/path_cost_opportunities_v2.json`, shows only 19 explicit path-and-sum claims in 36 eligible weighted-path checkpoints; four claims disagreed with graph/arithmetic (one arithmetic error, three graph/edge mismatches). This is a limited text-pattern prevalence audit, not an estimate for the task population. It makes the next decision clear: the current path-cost correction mechanism is too sparse to scale as-is. The next action qualification should target task families with frequent, consequential, executable intermediate obligations, while sampling natural checkpoints independent of Jev scores. Before running a controller study, measure event prevalence and verify that the proposed corrective information can alter a subsequent decision without leaking the final answer. Then freeze continue, neutral-feedback, tool-verified correction and a matched-cost protocol; advance only if development data show repeatable outcome value.

No local model worker or Bouchet Slurm allocation is active at this checkpoint (verified latest process receipts and process list). Thus no compute is being held. I did not submit a new workload: existing evidence shows the chosen intervention is rare and low precision, so another large run on the same task/actions lacks scientific justification. Next work is benchmark/action design around frequent intermediate obligations, followed by a fresh bounded development evaluation and only then Jev-versus-local/cheap selection tests.

## September 28, 22:15 local handoff (current)

Targeted arithmetic-correction worker PID 36977 ended with supervisor exit 0 after 1,211.7 seconds; `runs/targeted-arithmetic-v1-control/process.json` and output manifest are terminal. Independent audit `results/targeted_arithmetic_v1_audit.json` passed all 25 checkpoints, 50 outcomes and 53 calls, with zero outcome disagreements or timeouts. Analysis `results/targeted_arithmetic_v1_analysis_v2.json`: correction 42/50, prior matched sham 40/50; five versus three discordant paired episodes; problem-weighted +4 points, descriptive cluster-bootstrap [-6,+16]. Correction processed 32,534 prompt tokens versus sham 29,187; generated 50,653 versus 50,379. Nonconcurrent measured model-service seconds 1,205.8 versus 1,221.7, excluding correction calculation. Twenty checkpoints were weighted-path and five arithmetic-construction. No reliable outcome benefit or Jev-routing result established. Paper updated and built-in compilation passed.

The decisive bottleneck is action usefulness, not hardware access or a missing general theorem. Do not scale this narrow correction unchanged or train a checkpoint-specific controller on 25 selected states. Develop a fresh treatment whose information is genuinely relevant to final verified success, with matched neutral feedback, executable verification and a frozen runtime contract. Only if it yields repeatable action-effect heterogeneity should Jev semantic value be tested against strong cheap/local features. No local model or Bouchet job is active at this handoff; historical IDs below are terminal.

## September 28, 20:21 local handoff (current)

The active local targeted-arithmetic worker is PID **36977**, start **Mon Sep 28 20:19:07 2026**, receipt `runs/targeted-arithmetic-v1-control/process.json`, output `runs/targeted-arithmetic-v1-20260928`. It was verified live with matching PID and start. Its frozen schedule selects 25 naturally erroneous explicit addition checkpoints from completed development runs; 50 new continuations compare against saved same-seed sham outcomes. Do not inspect partial outcomes or launch a competing local model. On terminal status, independently audit all records with `scripts/audit_targeted_arithmetic.py`, then analyze with `scripts/analyze_targeted_arithmetic.py`. A verified arithmetic correction is extra information and this design does not test Jev routing.

The fresh 48-problem sham replication completed and passed independent audit: 192 outcomes, 296 calls, zero label disagreement. Sham 47/96 versus continue 46/96; paired difference +1.04 percentage points, descriptive problem-bootstrap interval [-7.29,+9.38]. Exploratory equal-problem synthesis with repair v3 gives +2.43 points over 72 problems, interval [-4.17,+9.03]. No reliable sham benefit established. Reports `results/sham_replication_v4_{audit,analysis}.json`, `results/pooled_static_sham_development.json`.

The narrow Jev natural-addition sensor study completed 48/48 judgments on 23 correct and 25 incorrect natural claims, with 14 cache hits. Independent audit AUROC 0.639; at 0.5 threshold sensitivity 23/23, specificity 1/25. Source `results/jev_natural_addition_v1_audit.json`; descriptive bootstrap `results/jev_natural_addition_v1_precision.json`. This favors executable arithmetic checks but does not answer broader semantic intervention value. A separate paired c-to-c+1 probe has only 14/16 complete pairs after HTTP503; 13/14 pair score differences positive, but all corrupted claims scored above 0.5. Do not treat partial probe as completed study or retry blindly. Cumulative accounted Jev ledger $0.015710698 includes $0.01 unresolved reservation, under the $25 cap. Paper updated with these narrow findings and built-in compiler passed.

The next decision is whether actionable local arithmetic correction improves final verified outcomes versus sham after all 50 new episodes pass audit, including timeouts and total costs. If it does, separately test whether a Jev feature can select checkpoints/actions beyond executable checks and strong cheap/local controls. If it does not, redesign treatment or move to another well-defined failure mode rather than scale the same action.

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


## Repair v3 terminal and static control replication

PID79898 absent; supervisor receipt remains stale running,so exit code not confirmed. Scientific manifest/summary complete480outcomes;independent audit passed631calls/24eligible/zero label disagreements. Last segment-repair outcome has timeout at4h bound;retain operational result and disclose. Continue63/96,sham68/96,suffix57/96,recheck56/96,segment54/96. Segment-minus-sham−14.58pp (descriptive−27.08 to−2.08). Split diagnostics selected68.75% versus continue64.58/66.67%,not evidence of beating best static sham. Reports results/action_replication_v3_{audit,analysis}.json.

New local worker21169,startMon Sep 28 18:13:31 2026,receipt runs/sham-replication-v4-control/process.json,outputs runs/sham-replication-v4-20260928.48fresh tasks,two repeats,continue/sham only,192outcomes,7200scooperative/7350external,8GiB sampledRSS,no Jev. Protocol docs/sham_replication_v4_protocol.md.382tests passed. No cluster job active. Next full audit and static-comparator estimate before adaptive controller claim.


## Continued session: cheap-controller attribution diagnostic

Sham21169 verified live original18:13:31 start,elapsed49:35. No interim outcomes inspected. Implemented audited-data-only leave-one-problem-out TF-IDF/Ridge(alpha10) diagnostic on completed repairv3; vocabulary,model and best static chosen using training problems only. All24held-out decisions selected sham;cheap-minus-training-static0,cheap-minus-continue+5.21pp. No personalized-control evidence. Source/results scripts/diagnose_repair_controller.py and results/repair_controller_loo_v1.json. Descriptive bootstrap does not account for fold dependence and is not a prospective CI. Constant-target/unseen-word test passed. Keep frozen sham study unchanged; complete it before adding semantic acquisition.
