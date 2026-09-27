# Executed results and research decision

Date: September 27, 2026. These are development/instrumentation results, not a completed main study or a demonstration that Jev improves reasoning. The repository and local working directory were empty when work began; prior chat download links and numerical claims were not treated as existing artifacts.

## The most consequential result is the prior-art correction

The proposed conceptual contribution—intervention advantage learned from identical-prefix counterfactual branches—is already explicit in [Calibration Is Not Control](https://arxiv.org/html/2606.21399v1). [DIAL](https://arxiv.org/html/2605.06908v1) is another direct learned-intervention precedent. [Jev-Mem](https://arxiv.org/html/2609.23986v1) already uses Jev for agent memory decisions and allocation. The [audit](literature_audit.md) covers 25 papers and selected source files with pinned references.

The defensible project is a replication and extension testing **incremental typed-feature value, transfer under a fixed adaptation budget, and selective acquisition measured by verified outcomes**. These are hypotheses, not established novelty or performance claims. A generic Jev-for-PRM substitution is insufficient.

## Actual local model and API execution

The local host is an Apple M5 MacBook Pro with 32 GiB unified memory and approximately 52 GiB free storage at inventory. Qwen3-4B-Instruct-2507 was already cached. The project uses MLX with in-memory 4-bit affine quantization, group size 64; no new model weights were downloaded.

The short quantized smoke generated 117 tokens in 2.716 seconds, about 43.1 tokens/second, with 3.814 GB MLX peak memory. A greedy exact-token pause/resume matched the uninterrupted sequence. This is one instrumentation check, not a universal stochastic replay guarantee. The cached checkpoint is an instruction model with explicit observable explanations, not the original Qwen3 thinking variant. [Model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

The live Jev adapter pins `jev-1.13.0`, asks seven independent Noul questions together, and stores usage/cost in one locked persistent ledger. One hand-written smoke plus six natural checkpoints used **5,327 input tokens across seven billable calls**. At the verified $0.042/million input-token price, the recorded input cost is **$0.000223734**. This is calculated from provider usage and published pricing, not an invoice reconciliation. The [official pricing page](https://docs.typesafe.ai/models) was checked during this work. Failed-call reservations were zero.

The six checkpoint queries had median round-trip latency **0.311 seconds**. An independently prompted version of the same local 4B model produced valid seven-field JSON for all six states, with median generation time approximately **1.952 seconds**. These are small-workload latency observations; no human semantic labels were collected, so they do not establish either judge's accuracy or calibration. The local judge never saw Jev outputs.

## Phase 0: 36 real continuations

Configuration: six procedural problems, one eligible constructed checkpoint each, three actions, two separate stochastic continuations per action. The prefix protocol generates a 128-token block, screens that block for completion/answer phase, and rewinds to a paragraph boundary. Its all-arm effect concerns that constructed-state distribution, not online stopping at the retained paragraph. Half the problems require a minimum-weight directed path; half require constructing a six-number arithmetic expression. Exact path validation and safe rational AST evaluation determine success independently of either judge.

Each hypothetical episode has a **512-token** generation allowance, including the 128 tokens generated to obtain its checkpoint and all losing branch candidates. Finalization reserves 96 tokens within that allowance. Branch selection uses mean generator log probability; it is a weak fixed selector, not a correctness verifier. Repair rolls back a bounded suffix and adds a fixed correction instruction. Generation settings are held fixed across arms.

| Action | Verified successes / continuations | Descriptive rate |
|---|---:|---:|
| Continue | 2 / 12 | 16.7% |
| Repair | 2 / 12 | 16.7% |
| Branch | 1 / 12 | 8.3% |

All six problems were eligible. The independent statistical unit is the original problem, **n = 6**, not the 36 replay episodes. Three problems had no successes under any tested action. Twenty-four continuations ended at their token cap; twelve stopped naturally. No timeout occurred. The continuation calls emitted **12,795 tokens in 313.48 seconds**, plus 768 shared prefix-generation tokens. The derived aggregate of 40.8 emitted tokens per continuation wall-second includes these calls' prefills; it is not isolated kernel decode throughput.

This is **all-arm data collection**, not deployment of a Jev-controlled policy. It cannot estimate a Jev-versus-no-Jev improvement. Isolated rescue examples are insufficient to establish stable treatment-effect heterogeneity. A high local-validity probability followed by terminal failure also does not prove a bad judgment: the question concerns the current segment, not success after a restricted future budget.

The completed-run audit found no generated-token budget violations, no initial-prefix hash mismatches for continue/branch, and no episode affected by the finalization bug discovered during review. Five checkpoints discarded a later generated tail; their whole-initial-generation entropy/logprob means must be excluded from predictive features. Future runs save retained-token statistics directly. Original outcomes are preserved unchanged.

Evidence: [summary](../results/phase0_summary.json), [implementation audit](../results/phase0_review_audit.json), and raw `runs/phase0-20260927/{manifest.json,checkpoints.jsonl,outcomes.jsonl,local_judge.jsonl}`. The original source was reconstructed by exactly reversing the recorded review patch; this provenance limitation is stated in its manifest. Later runs automatically save source snapshots and hashes.

## Development budget calibration

A separate completed run reuses the first four development problems with a 1,024-token allowance and one replay per action. It uses the corrected final-reserve implementation and the same frozen model/rubric. All four Jev states were cache hits, so no additional billable requests occurred. This is a budget diagnostic, **not an independent replication or held-out test**.

| Action | Verified successes / continuations |
|---|---:|
| Continue | 0 / 4 |
| Repair | 2 / 4 |
| Branch | 1 / 4 |

The twelve continuations emitted 10,298 tokens in 244.34 seconds, plus 512 shared prefix tokens. These are the same development problems and partly the same seeds as the first run; do not pool them as new independent problems or infer a positive repair effect from 2/4. Increasing the budget alone did not produce consistently successful continuations. The final audit found no generation-budget violations or initial-prefix mismatches. [Summary](../results/budget1024_summary.json), [analysis](../results/budget1024_analysis.json), [audit](../results/budget1024_audit.json).

## Theory and implementation validation

The [theory appendix](theory.md) gives eight propositions with proofs: checkpoint identification, free-information value, action-value regret, conditional sequential improvement, held-out concentration, randomized pruning audits, a conditional transfer bound, and cost-inclusive feature acquisition. These specialize established theory; none supplies a guarantee from Jev confidence alone.

[Executed synthetic validation](../results/theory_validation.json) now passes fourteen checks, including 200 finite MDPs and 500 greedy-regret trials. The largest performance-difference identity residual was approximately 1.30e-16. An explicitly constructed sensor example has cheap/always-query/selective-query net values of 0.725/0.765/0.780. Those numbers are mathematical examples with stipulated parameters, **not measured Jev results or predicted effect sizes**. The [independent review](theory_review.md) adds exact counterexamples for adaptively stopped replay means, dependence across cross-fit contributions, and noisy-oracle optimism. Fixed-contribution bootstrap summaries of learned cross-fit policies have no generic nominal-95% coverage.

The tested implementation includes concurrent budget reservations, persistent cache behavior, ambiguous failure charges, malformed API rejection, exact outcome validators, losing-branch accounting, partial-final-answer completion, prefix preservation, and exclusion of future-token features. **93 unit tests passed before launching development v1**, including the new online-boundary and baseline/sham tests. Additional synthetic functional checks exercised the grouped cross-fitting path on 24 constructed problems; no learned controller was fitted to the six real development problems.

## Decision: improve the experiment before scaling it

Do not launch the originally proposed multi-million-token campaign from these results. Several early checkpoints mostly restate task inputs; two graph checkpoints end at “Given edges:” after only 41/45 retained tokens. Error localization, contradiction, and repeated-failure signals consequently have little opportunity to vary. Short budgets and weak branch selection further limit the conclusions.

**Development v1 is running:** 12 new problems, two seeds, baseline and sham-resumed generation (48 episodes), 1,024 generated tokens each. [Frozen settings](../configs/development_v1.json). The concise prompt discourages input repetition; checkpoints stop online at the first newline at or after 256 tokens, capped at 384, without seeing later text. A newline is a formatting boundary, not a semantic-quality guarantee. Both arms use the same final-answer reserve protocol. No Jev scores or outcomes select checkpoints, and this run makes no Jev calls. Finished/ineligible tasks remain recorded. If arithmetic remains at floor, consider a separate four-number development condition while retaining six-number tasks as a later difficulty shift. Until the run finishes, no outcome claim is available from this stage.

Once those settings and the selector are frozen, collect **24 fresh problems × 3 actions × 4 repeats = 288 continuations**, plus uninterrupted baselines. Expand to 48 only if state/outcome variation justifies it. Compare identical learners using cheap text/statistical features, independently constructed local-judge features, and Jev features; include best-constant, correctness-threshold, and rate-matched random controls. Fresh full-episode evaluation remains necessary before a policy-effect claim. The 24-problem threshold merely avoids absurdly small fits; it does not make a small study statistically powerful.

## Resources needed now

No additional API key or GPU purchase is needed for development. The supplied key is configured privately outside the repository. Local inference and controller fitting are sufficient. Keep one model process at a time; existing system swap use means nominal 32 GiB should not be treated as entirely free memory. Reserve about 5–10 GiB additional working storage for dependencies and text/token artifacts; do not duplicate weights or save full vocabulary logits/KV caches.

For 48 checkpoints × three actions × four repeats at 1,024 tokens with 128-token prefixes, the ceiling is about **0.522 million generated tokens including shared prefixes**. At an assumed 40 emitted tokens/second, that is **3.63 device-hours**, before costs absent from that throughput scenario. The separate 12-checkpoint × three-action × 12-repeat mechanism subset adds about 0.387 million continuation tokens. Allow roughly **5–8 local hours** for the combined screen and replications, then revise using actual longer-context profiling. [Calculator](../scripts/resource_estimator.py)

The same 48 unique states at an assumed 1,200 Jev input tokens each cost about **$0.00242**, excluding other rubrics/policies and retries. Compute and experimental design are the present constraints, not the $25 allocation. Spending caps are $0.25 for instrumentation, $1 for the mechanism screen, $5 for a gated pilot, $20 for justified extensions, with $5 held in reserve and **$25 absolute total through this client**.

The initial Mac mini SSH refusal was resolved by another authorized Codex setup workflow on 2026-09-27. This project subsequently verified SSH and SFTP through `mac-mini`, using the enrolled client key and pinned server identity. The mini has an Apple M4, 16 GiB unified memory, and approximately 14.8 GiB disk free at this check. Its inference runtime/model suitability and throughput still require qualification before moving experiments. No scientific run was launched or modified during the connectivity check. The current host can complete the next development experiment; a 48–80 GB NVIDIA worker remains optional for a later justified multi-model study.
