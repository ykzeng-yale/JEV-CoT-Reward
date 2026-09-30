# Final six-hour validation and investment decision — September 30, 2026

## Authorization, scope and stopping boundary

The user requested a final bounded six-hour research session with at most three iterations. The exact window is **September 30, 16:26:58–22:26:58 EDT** (20:26:58 UTC through October 1, 02:26:58 UTC). This document is a prospective decision protocol, not a claim that a new experiment is running or that all remaining curriculum stages can finish in six hours. Current live job identities and terminal receipts remain authoritative in `docs/current_research_state.md`.

The previously submitted eight-GPU ReCoMA job **27880628 was cancelled at 16:13:06 EDT with no allocation and zero GPU-hours**. Preserve its input tree, original receipt and cancellation history. A new qualification/full-panel run requires a new immutable directory, manifest and receipt; it is not a silent resurrection of that submission.

No new hosted-model or Jev call is permitted in this phase while the recorded TypeSafe terms gate remains unresolved. Keep the existing cumulative $25 Jev ledger unchanged. Do not move credentials to Bouchet, use a different PI account, request paid priority, touch unrelated jobs, reopen the closed v1 test, or start competing local model inference. Qwen3-4B-Instruct-2507 BF16 is the existing qualified generation backend; DiscoveryWorld/ReCoMA end-to-end controller execution still needs qualification. The model revision is `cdbee75f17c01a7cc42f958dc650907174af0554`.

At the deadline, stop this phase's generation, cancel its still-pending requests, preserve completed/partial records and give a final evidence/investment decision. An incomplete panel is incomplete evidence, not a reason to extend the window. Only task-owned jobs from this phase are subject to its deadline. The broad project is not scientifically complete merely because this validation phase ends.

## Scientific decision and minimum necessary new experiment

The useful-action question now has an audited answer for a specified cheap conductivity decoder: measurement **150/150** versus masked measurement **70/150**, +53.33 percentage points, 80 paired wins, zero losses, all six substance groups positive, and zero policy failures. Full strict replay accepts 600 episodes, 450 census resets and three integration traces. This is strong development-panel information-use evidence; it is not Jev or learned-controller value.

The cheap meter has 100% accuracy on that panel, leaving no observed accuracy headroom for Jev. Do not use the remaining hours to repeat this measurement proof, train an outcome-selected controller, enlarge the previously negative repairs/branches, or invent a new toy tradeoff solely to obtain a positive result. Its 32 padded calls per episode also cannot establish an efficient policy: intrinsic work is 18 actions for measurement versus six for continuation.

The minimum meaningful remaining model experiment is a **complete, independently audited, bounded ReCoMA-ReAct/DiscoveryWorld baseline and visible-trace/cost census**, following an end-to-end single-GPU qualification. A compatible published-controller baseline, scientific-failure accounting, observable action/checkpoint availability and actual trajectory costs are missing from the paper evidence. This experiment addresses those prerequisites. It tests no Jev increment, intervention effect, hidden CoT, general OOD performance or novel method.

The preferred scientific panel is **120 total episodes = 24 official theme/difficulty strata × five seeds (0–4)**. It is not 120 strata or 600 episodes. The exact published-controller adaptation, patched simulator, prompt, model/runtime identity and 30-environment-action / 62-model-call / 400-completion-token ceilings remain explicitly disclosed. A smaller panel, if forced by measured throughput, is a newly frozen stratum census and cannot be described as completing the official full-panel comparator.

## Hard resource and output limits

Use only Bouchet account `pi_fl426`, normal tier, and a compatible GPU partition already supported by the pinned runtime. The hard limit is **16 newly allocated GPU-hours in total**, including qualification, failed attempts, idle worker time and all scaling. There may be at most **eight concurrent GPUs**, and concurrency above one requires successful qualification of the identical worker/backend/controller route. Prior project costs are separate from this new limit.

Iteration 1 permits one GPU, four CPUs, 32 GiB RAM and **45 minutes maximum**, hence at most 0.75 GPU-hours for the initial qualification. Every cause-specific retry counts against the same 16-hour ledger and the same iteration; stop after two attempts with the same cause. No duplicate pending request is allowed. Pending jobs consume zero actual GPU-hours, but requested allocation ceilings must be reserved when deciding whether another submission fits the phase cap.

For a qualified parallel run, allow at most four CPUs and 32 GiB RAM per GPU worker (maximum 32 CPUs/256 GiB for eight). Freeze a per-job wall limit no longer than both the remaining deadline and the remaining GPU-hour reserve divided by the requested GPU count. For example, after a 45-minute single-GPU qualification, an eight-GPU request may reserve at most 15.25/8 = 1.90625 hours; a two-hour eight-GPU request would exceed the cumulative cap. A full eight-GPU/48-hour allocation is prohibited in this phase.

Use the existing cached pinned weights and offline locked dependencies; no large new model download. The full 120-episode completion ceiling is **2,976,000 tokens** (120 × 62 × 400), not an expected token count or a throughput forecast. Freeze a 2-GiB primary trace/output cap, a 4-GiB total evidence-bundle cap and bounded logs per job. Exceeding an output cap is an infrastructure stop with preserved records. Report every generated/discarded token, full prefill, environment action, failure, hosted/local judging call and actual wall/allocation cost. Simulator score and completion fields must retain the published adapter's explicitly documented visibility boundaries.

The batch program must enforce the absolute deadline before generation and during the bounded worker execution. Scheduler submission timing alone does not ensure termination by the deadline. No work may continue after 22:26:58 EDT merely because a request started late.

## At most three iterations

### Iteration 1 — repair failure accounting and qualify one complete worker

Before inference, fix and test the discovered malformed-JSON to uncaught-`TypeError` route and verify that malformed scientific actions cannot silently abort the remainder of a shard or disappear from the denominator. Preserve each full request/response and explicit failure reason. A transport/device/runtime failure is infrastructure failure; a malformed or unsupported generated action is a scientific controller failure under the frozen action contract. Do not silently reprompt or grant extra calls unless that behavior is separately frozen, charged and disclosed as an adaptation.

Freeze **two representative runtime-only episodes** selected from the already available official task inventory by source/runtime characteristics, not endpoint outcomes. Record their exact task identities and why they exercise different task/action/context paths in the new manifest before generation. They are technical qualification units, not a two-task efficacy panel. Use the same 30/62/400 limits and source/model/runtime intended for scaling. Source/parser regression tests are necessary but do not replace this allocated execution.

Qualification passes only if both scheduled episodes are retained with explicit terminal success/failure records, all call/action caps hold, independent parsing/token/request/outcome checks accept the records, scientific malformed-output handling is exercised by deterministic tests, GPU peak memory fits the allocation, and measured end-to-end per-episode/context throughput is available. Endpoint success is **not** a qualification gate. A correctly recorded unsuccessful agent episode can pass infrastructure qualification.

The existing full-panel auditor hard-requires 120 tasks/eight GPUs and parses every model output as a valid action. Consequently a runner-only exception fix is insufficient: a separately frozen auditor must accept and charge explicitly logged scientific parse failures while rejecting missing/inconsistent traces. Qualification and timing-fallback panels need their own exact expected-task/resource contracts fixed before generation; do not loosen the original 120-row/eight-GPU auditor after seeing outcomes.

The final qualification additionally performs one separately logged **32,768-input-token / 400-forced-output-token synthetic hardware stress** through the already loaded BF16/SDPA model; it loads no second model and is not a scientific episode. Ordinary EOS and stop-string termination are disabled only for that synthetic call, and all input/output IDs, tokens, memory and synchronized latency are recorded at `worker-0/context_stress.json`. Two early agent parse failures cannot substitute for this full context/completion stress when forecasting longer episodes. The source's 10,000-character history restriction alone does not bound injected observations/dialogs. Therefore the new v3 worker also checks each real prompt with the pinned tokenizer before generation and **aborts as infrastructure failure above 32,768 input tokens**, without truncation, resampling or extra inference. Such an abort preserves records and blocks full-panel rate analysis. This is an explicit bounded-runtime adaptation, not a proven global DiscoveryWorld prompt bound.

Inspect terminal `sacct`, exit/resources, logs, model/runtime identity and checksums before promotion. Retain allocation and tokens even when qualification fails. No endpoint-rate-based tuning or sample-size decision is allowed. A repeated same-cause failure exhausts that route rather than triggering another serial tiny pilot.

### Iteration 2 — freeze and run the largest complete defensible census

Use only qualified timing/memory/correctness evidence and remaining phase resources to choose one tier, before opening any baseline success rates:

| Tier | Frozen panel | Claim available after a complete audit |
| --- | --- | --- |
| Preferred | All 24 strata, seeds 0–4: 120 episodes | Full official-panel Qwen/ReCoMA adaptation and cost/trace census |
| Timing fallback | All 24 strata, seeds 0–1: 48 episodes | Balanced two-seed stratum census; incomplete official comparator |
| Final timing fallback | All 24 strata, seed 0: 24 episodes | One-seed stratum census only; strong precision limitations |

All three tier rules are fixed now. Do not select favorable tasks or omit difficult strata. Qualification task results may be excluded from scientific inference or regenerated under the separately frozen census; either choice must be recorded prospectively and all repeated collection charged. Technical qualification outcomes cannot choose the tier.

Forecast full-worker completion from measured end-to-end episode time, including environment/history growth and observed setup time, with an explicit conservative allowance recorded in the receipt. Verify that every planned shard fits the remaining deadline and that aggregate requested GPU-hours fit the remaining 16-hour reserve. Maximum-worker memory and actual allocated idle time matter. Generation tokens/second alone are inadequate. Choose the largest complete tier whose forecast fits both limits; if no complete tier fits, do not launch an arbitrary partial efficacy panel.

Assign independent tasks to worker shards deterministically before launch. Run a single compatible Slurm request rather than duplicates across GPU types. Do not inspect partial success rates, change prompts/controllers, resize the panel from outcomes, pool MLX/BF16 data or extend after a disappointing result. No uncontrolled hyperparameter grid is allowed.

All rows must remain in the intended denominator, including scientific generated-action failures, unsuccessful terminal episodes and budget exhaustion. Missing traces, unconsumed calls, private evaluator-field leakage, invalid token accounting, unexplained process failure or incomplete planned task coverage block scientific rate analysis. A validated scientific failure is not an infrastructure excuse for exclusion. Report task completion, independently verified success and normalized official score separately.

### Iteration 3 — audit, scientific synthesis and investment decision

Retrieve scripts/configuration, source/model/runtime manifests, full traces, terminal endpoints, logs, checksums and parent-job accounting. Independently verify the selected complete tier before calculating success rates. Aggregate first within theme/difficulty strata, then equally over strata. Uncertainty is descriptive for this finite benchmark; seeds sharing a template are not independent populations. Report parse/action failures, task/environment failures, eligibility under any declared visible-only checkpoint rule, intervention counts (zero for this baseline), full tokens/actions and actual GPU/CPU-hours.

Define any visible-trace checkpoint rule before applying it to the completed data, and never use terminal success to select checkpoints. An observation that a baseline reasons or scores well cannot identify intervention value. If a concrete useful information action or state-dependent cost opportunity is visible, document its observable trigger, strong cheap alternative and needed fresh-data contrast. This preparation is not a newly validated controller and does not authorize an unregistered fourth iteration.

The final report must separate: audited measurement utility; absent/negative Jev evidence; correctness fixes; new baseline/cost/trace findings; novelty/theory limits; unresolved E2–E7 requirements; and a clear bounded continue/pivot/stop recommendation. If queueing or infrastructure prevents a complete new census, state that the decisive evidence is still missing. Never manufacture efficacy or scientific novelty to fill the six-hour window.

## Sample-size and comparator limits

The complete 120-episode census has only 24 template/difficulty strata and is a baseline description, not a randomized treatment comparison. Even optimistically treating episodes as independent, the checked paired-normal planning function with 30% discordance, two-sided alpha 0.05 and 80% power gives approximate detectable paired effects of **13.87 pp for n=120**, **21.61 pp for n=48**, and **29.81 pp for n=24**. Clustering/transport can worsen these boundaries. These numbers are scenario calculations for a hypothetical binary paired contrast, not power claims for this one-arm score census.

Under the same assumptions a 3-pp contrast requires **2,614 independent tasks**, and a 5-pp contrast **940**. Repeated generations, action calls and five seeds cannot be counted as thousands of independent tasks. The six-hour phase is therefore not a valid small-Jev-increment confirmation. No effect-based sample-size adaptation or repeated tiny study is permitted.

Necessary paper comparison requirements remain:

- Strong cheap/static baseline: the complete visible meter policy is now qualified; continuation prior and masked/random classification controls are retained, with intrinsic and padded costs separated.
- Strong local and Jev increment: earlier completed selectors show no positive increment. No new Jev comparison is available while terms remain unresolved; do not relabel the meter or ReCoMA result as Jev evidence.
- Published comparator: the new ReCoMA run is an open-weight backend/model adaptation. InterWhen is also an adaptation; neither reproduces the original larger-model reported scores. LW2S-style cheap action-safety control remains a required comparison for any future skip/control novelty claim.
- Causal action attribution: a future intervention requires fresh eligible states, an action useful beyond cheap static behavior, one frozen primary paired contrast, fair sham/randomization and complete costs. Baseline throughput or natural thought prevalence does not satisfy this requirement.
- Sequential/transfer claims: still absent. Conditional identification/value-of-information/regret statements cannot guarantee that deployed Jev judgments are accurate or useful; the existing theory is standard and assumes the relevant conditions rather than empirically proving them.

## Independent existing-evidence checks performed for this protocol

An independent offline recomputation checked the original conductivity result SHA, all 600 unique variation/policy keys, each 150-row denominator, task-success counts, all action totals and three paired win/loss/tie counts against the safe aggregate. All checks agree: measurement150/150, each control70/150, 4,800 actual padded actions per policy, and 80/0/70 paired outcomes for each measurement contrast. This check does not replace the already completed strict runtime replay; it verifies summary arithmetic independently.

The receipted Slurm inventory has 27 unique parent jobs. Recomputing `elapsed_seconds × allocated_gpu_count / 3600` and its CPU counterpart exactly reproduces **3.683333 GPU-hours** and **17.741944 reserved CPU-hours**. These are documented receipt-discovered subtotals, not all-time Mac device time, electricity cost or a complete monetary invoice. Batch/extern steps must not be counted again. Pending/cancelled-unallocated jobs contribute zero allocation. The arithmetic review also reproduced the sample-size boundaries above using the existing checked `scripts/paired_binary_power.py`.

Frozen references at protocol preparation:

| Artifact | SHA-256 |
| --- | --- |
| `data/discoveryworld_e1a_v2/recoma_react_full_panel_v2.json` | `84175595f412277b614d2c06773412994fdb9e26c3730ded851ff486f04ade8e` |
| `data/discoveryworld_e1a_v2/tasks.json` | `c1537b883f764de838d118394bbf89286a2a31e204a97a9d7bb0acff312803f3` |
| `results/scienceworld_conductivity_action_v1_analysis_20260930.json` | `bfd37bbf8578be01be90b9c9fd60a68764e67a6919fcc80185006bf0b8f6ebcd` |
| `results/investment_review_receipted_slurm_costs_20260930.json` | `dfb2633271f51bcf2914da14fc32135c1ad1a55dc7bdf2550a04ff4cdd62edce` |
| `scripts/paired_binary_power.py` | `f60d5e5656481f324449af6b9835bd616c1f8689bec8f8ce6adbc6c0a0ac7010` |

Freeze the actual newly corrected source, qualification identities, selected tier, wall/output/resource limits and this protocol's own hash in the new submission receipt before scientific generation. A needed engineering correction receives a separately versioned input bundle and explicit justification; it does not overwrite historical runs or permit outcome-selected redesign.

## Final decision rule

The default recommendation remains **no broad scale-up of the original Jev-CoT-Reward claim**. The final phase can justify a further bounded investment only if its audited evidence exposes a nontrivial observable action/cost question, a credible strong cheap/published comparator and fresh evaluation units capable of answering it. A complete baseline with no such opportunity supports stopping or pivoting this specific proposed contribution. A malformed/low-success baseline alone is not proof that all research in this area fails; it is evidence about this frozen adaptation. A queue/infrastructure failure leaves the scientific question unresolved and supplies no reason to promise another open-ended compute cycle.

Positive meter utility, successful code tests, a commit, GPU access and conditional standard theory are insufficient by themselves to establish Jev innovation. The final conclusion must follow audited outcomes and defensible scope, including negative or incomplete results.
