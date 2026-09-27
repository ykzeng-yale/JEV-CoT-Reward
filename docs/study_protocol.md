# Study protocol: when should a judge intervene?

Version: 2026-09-27. This is a staged prospective protocol with the active Phase 0 configuration recorded below. It does not report study outcomes, reproduce the numerical results pasted in the project request, or claim a proven novelty gap. Actual run results and deviations belong in separate manifests.

## 1. Decision and falsifiable hypotheses

Start with a frozen local open-weight generator and one checkpoint per episode. Estimate the effect of **continue, local repair, and local branch** by replaying the same checkpoint. Train only a small controller. Jev is one optional feature source; a local open-weight judge and a strong cheap-feature controller are essential comparators.

The primary question is whether Jev features add sufficient decision value to improve independently verified success under declared resource constraints. Separate three hypotheses:

| Hypothesis | Required evidence | Result that weakens it |
|---|---|---|
| H1: intervention effects differ by checkpoint | Independently replicated variation that a policy can exploit beyond the best constant action | No stable differences; one action dominates |
| H2: Jev adds useful information | Frozen Jev-feature policy exceeds the same learner with cheap features on fresh problems | Only in-sample fit improves, or local judge matches it at lower total cost |
| H3: useful control survives deployment | Full episodes improve with all costs counted, then survive repeated interventions and named shifts | Gains vanish with query cost, repeated control, or modest shift |

**The earlier proposed novelty is already covered.** [Calibration Is Not Control](https://arxiv.org/html/2606.21399v1) explicitly studies intervention advantage, same-prefix counterfactual branching, and action-conditioned prediction. [DIAL](https://arxiv.org/html/2605.06908v1) also uses counterfactual checkpoint exploration to learn intervention direction, and [AERA](https://arxiv.org/html/2608.27964v1) learns the residual value of further inference. These are required direct comparators and replication foundations; “learn intervention value rather than correctness” is not our invention.

The narrower project is a **replication and feature-acquisition extension**: measure Jev's incremental semantic decision information versus cheap and local open-weight alternatives, including whether a cheap pre-query gate can acquire that information selectively under budget, latency, and specified distribution shifts. Query selection is standard [metareasoning](https://arxiv.org/abs/1207.5879), so this still needs evidence and a targeted novelty audit. Observable hypothesis/test/retraction roles may be a useful prespecified subgroup, not an assumed unique mechanism. The accompanying [theory](theory.md) states identification, acquisition-value, and safety limitations.

## 2. Prerequisites before spending on Jev

Record actual host, chip/GPU, memory, available disk, backend, and measured throughput. Freeze model repository plus revision, quantization, tokenizer revision, chat template, sampler parameters, package versions, source commit, and benchmark generator seeds. A quantized Apple Silicon run is a valid local condition; it must not be described as a BF16 NVIDIA replication.

The resource inventory reports a local Apple M5 with 32 GiB unified memory, about 56 GiB free disk, and cached Qwen3-4B-Instruct-2507 weights. The active Phase 0 uses that model in MLX with affine 4-bit quantization, group size 64, applied in memory. Keep the exact revision and conversion configuration in the run manifest. A short engineering profile reported about 43 decode tokens/s and 3.8 GB peak memory; this is a short-run measurement, not whole-study throughput. It is an instruction model: the experiment operates on **explicit, observable reasoning text**, not inaccessible private CoT. Do not download multiple large models solely to satisfy a model list. A second receiver or larger model is conditional on positive results and available memory. A 1–2B-class local typed judge can run sequentially with the generator to reduce resident memory; the same 4B model can provide an initial separately prompted local-judge diagnostic. If it cannot follow the schema, measure that failure rather than silently using a stronger proprietary model.

Four instrumentation gates precede scientific data collection:

1. **Resume correctness.** Token IDs, prefix length, special tokens, and chat template are preserved. Compare uninterrupted generation to sham pause/resume under controlled sampling; deterministic or seed reproducibility is tested at the backend's actual guarantees.
2. **Budget correctness.** Generated-but-discarded drafts, replacement tokens, candidates, selector inference, prompt processing, and Jev usage all enter their appropriate ledgers. Verify exhaustion and crash recovery.
3. **Outcome integrity.** A separate exact solver or audited verifier scores the final artifact. Gold answers never enter generator/judge prompts or branch selection. Parsing failures are failures under the fixed specification.
4. **Isolation and logging.** Replays restore tools/files/frontier and use separate output locations. Data records retain API failures, timeouts, early finishes, and truncation. No secret appears in artifacts or git.

If exact prefix continuation is unavailable, use an explicitly named **message-level intervention** experiment with a sham reformatting control. Do not claim it estimates exact-token continuation effects.

## 3. Benchmarks and checkpoint sampling

The first runnable harness uses procedurally generated **weighted directed acyclic graph minimum-path** tasks and **six-number arithmetic construction** tasks. The arithmetic verifier evaluates a restricted expression AST with exact fractions and checks number usage; the graph verifier checks feasibility and independently computed optimum. Those are engineering diagnostics, not external benchmark evidence. Include multiple correct artifacts when appropriate and accept all valid solutions according to the task definition.

The research pilot should move to audited procedural families in [Reasoning Gym](https://github.com/open-thought/reasoning-gym), with a pinned revision and an independently checked sample. Candidate families are arithmetic construction, graph/path planning, and propositional logic; exact available names, difficulty parameters, and scorer semantics must be verified in the installed revision. Strict success means the fully correct endpoint, not positive partial credit. External math or coding tests are later extensions.

Freeze task difficulty on development examples to avoid both floor and ceiling effects. Preserve every sampled problem. If a problem finishes before its target checkpoint, record “no eligible checkpoint”; it remains in episode-level evaluation and contributes no forced replay. Never extend a completed trace merely to create a treatment opportunity.

Historical Phase 0 generated an initial block of at most 128 tokens, screened the whole block for completion/answer phase, then retained a paragraph boundary after at least 32 tokens. Generated tails remained charged. This is a generate-then-rewind constructed-state distribution, not online stopping at the retained paragraph; earlier all-arm effects are scoped to that protocol.

Development v1 instead stops **online** at the first emitted newline at or after 256 tokens, capped at 384. It never uses later text to choose a checkpoint. Twelve fresh problems, two seeds, baseline generation and sham-resume make 48 episodes; each has 1,024 generated tokens including a shared 96-token final-answer reserve. The baseline is uninterrupted up to that reserve; both conditions use the same finalization rule. Resume preserves token IDs but uses a fresh recorded RNG seed, so this is not a claim of stochastic bitwise equivalence. See `configs/development_v1.json` and `scripts/run_development.py`.

The gated mechanism v1 adopts that online boundary, with the last retained paragraph as the judge's latest segment and prior paragraphs as history. A newline/paragraph is a formatting unit, not proof of semantic completeness; a surface calculation marker is a diagnostic, not a correctness label. Its fixed schedule covers 24 new problems and four continuations per action, using distinct task seeds from development. Model weights/tokenizer/config, source bytes, rubric, schedule, package versions, failures and incomplete calls are recorded. See `configs/mechanism_v1.json` and `scripts/run_mechanism.py`. A preparation or interrupted-call failure is not scientific futility. Later checkpoint/horizon variants need a new frozen protocol and fresh tasks.

Sample natural checkpoints without using Jev scores or outcomes. Stratify task families and prespecified difficulty bins. Induced errors, edited traces, or adversarial prefixes form a separate diagnostic dataset and cannot substitute for natural-prefix results.

## 4. Three treatment contracts

The active Phase 0 uses a **512-token** total generator allowance including initial prefix and discarded generation, with **96 tokens reserved within it for finalization**. A fixed finalization marker is identical across all arms. It tests six problems, three actions, and two independent replays per arm. The 24–48-problem screen may profile 768 or 1,024 tokens on development tasks if 512 leaves no usable variation; freeze the chosen allowance before collecting that screen. Expand to 2,048 only when supported by measured task behavior and throughput. Context caps are independent constraints and must be saved in the actual run configuration.

| Action | Contract | Accounting |
|---|---|---|
| Continue | Resume exact saved prefix with baseline sampling and no additional advice | Uses remaining allowance |
| Local repair | Roll back to a paragraph boundary if one occurs within the final 128 tokens; otherwise remove that 128-token suffix. Insert a frozen self-correction instruction and regenerate | Rolled-back generation is not refunded; new tokens and extra prefill are charged |
| Local branch | Active Phase 0 samples two next-segment candidates, each at most 64 tokens, from the checkpoint; a fixed selector chooses one, then baseline continuation resumes | Both candidates and continuation are charged; selected candidate tokens are not charged twice |

Use one frozen selector across all controller comparisons. Active Phase 0 selects by the candidate's **mean base-model token log probability**, without another judge-generation call. This is a weak instrumentation baseline, not evidence that the branch implementation is strong or that likely text is correct. Specify tie behavior. For Phase 1, compare a stronger audited local semantic selector on development tasks and freeze it before data collection. It receives only the checkpoint and candidate texts, never gold outcomes; randomized candidate order and choice-only output can control position and verbosity effects. If that selector generates tokens using the same model, debit those tokens from the same allowance, and record its extra prefill. A random-selector ablation separates branching diversity from selector quality.

For near-exhausted checkpoints, action feasibility is determined before selection. Either require enough budget for all three actions or set a declared feasible action set; never give repair/branch extra resources. The same complete checkpoint, budget counters, model settings, and available evidence initialize every arm. Randomize execution order and use fresh seeds.

Repair necessarily changes context and instructions. Its estimated effect belongs to that specific bundled intervention. A separate sham pause/resume and, where needed, an instruction-only sham distinguish scheduling and formatting artifacts. There is no claim that the experiment isolates an abstract “repair” independent of prompt design.

## 5. Feature sources and crossed comparisons

The inexpensive baseline includes depth, elapsed generation tokens, remaining budget, repeated n-gram counts, task-visible metadata, selected-token entropy/log-probability summaries when available, and train-only TF-IDF/SVD of the prefix. Do not give one condition hidden logits unavailable to another without reporting a separate access regime.

The implemented `semantic-v1` schema asks the same seven binary question meanings of Jev and the local judge. A richer categorical epistemic-role schema is a future version, not an output of this run:

| Signal | Target |
|---|---|
| Tentative hypothesis | Explicit tentative hypothesis or assumption to be tested, rather than an established assertion |
| Explicit contradiction | Mutually incompatible asserted claims under the same stated assumptions |
| Unsupported dependency | An unavailable observation, source or tool result is treated as already obtained |
| Repeated failure | A previously failed approach repeats without addressing its failure |
| Error localization | A specific potentially repairable error is identified |
| Testability | A concrete available check distinguishes active hypotheses |
| Local validity | Asserted conclusions are supported by the given premises/history, treating hypotheses as tentative |

Use closed-set questions, a frozen rubric version, and one batched Jev request per unique checkpoint when the verified API supports it. Preserve raw response distributions, missingness, model version, exact rubric, normalized state hash, usage, elapsed time, and failure status. Cache by all model/question/state parameters that affect the response. Features are collected before replay outcomes are inspected. Never ask Jev for the known final answer or add the answer to its state.

One exploratory conjecture being unproved is not automatically an incorrect assertion. Jev confidence is not a bound on action-value error. The initial local judge uses the same cached 4B weights under a separate frozen prompt, with no Jev outputs. It is independently prompted, not an independent backbone or a model trained to imitate Jev. Its generated numeric probabilities are not assumed calibrated; invalid JSON/fields remain failures and recorded missing features.

Use a small regularized per-action Bernoulli outcome model, initially pooled logistic regression with action interactions; a shallow boosted-tree learner is a secondary development choice. Keep learner class and tuning budget the same across feature sets. Fit every observed replay outcome, not the maximum noisy action mean as a classification label. Weight problems equally so long traces or extra replications do not dominate silently. Features must be available at action-selection time.

The supplied exploratory analysis implements a simpler per-action Ridge model on individual Bernoulli outcomes, with predictions clipped to [0,1], train-fold TF-IDF, and numeric imputation/scaling. It is a reproducible screening baseline, not a calibrated logistic model or the required conservative random-forest reproduction. It refuses learned-policy analysis below 24 complete independent problems. The six-problem instrumentation run therefore supplies no trained-policy result.

## 6. Staged execution and spending gates

The user-authorized Jev ceiling is **$25 total**. It is a ceiling, not a target. The expected first useful test should cost far less than one dollar at currently verified input-token rates; compute, data quality, and small-sample uncertainty are likely the constraints. A provider-side cap is preferable when available. Local enforcement should stop new requests before a conservative liability reserve reaches the cap.

| Stage | Design | Jev spending ceiling | Gate |
|---|---|---:|---|
| 0: instrument | Completed: 6 tasks × 3 arms × 2 repeats; sham/resume, validator, budget, cache and failure tests | $0.25 cumulative | Correct accounting and usable unfinished prefixes |
| 1: mechanism screen | Frozen v1: 24 new problems; at most one checkpoint; 3 actions × 4 replays. Expansion to 48 requires a separate review | $1.00 cumulative | Nondegenerate outcomes and evidence of actionable differences worth more data |
| 1b: replication | Prespecified 12 eligible checkpoints; 3 actions × 12 **new** replays | Within Stage 1 cap | Check persistence and winner's bias independently |
| 2: gated pilot | 120 development + 60 tuning + 120 untouched test problems; at most 2 checkpoints per development problem, 3 actions × 3 replays; 4–6 frozen policies on test | $5.00 cumulative | Positive held-out value/cost evidence, not a Jev-score increase |
| 3: targeted extension | Extra replication or one transfer/sequential test chosen from the unresolved question | $20.00 cumulative | Written result review selects the next informative test |
| Reserve | Reconciliation/retries and later validation | $5 held back; **$25 absolute total** | No automatic escalation to larger campaign |

These are accounting gates, not repeated hypothesis tests. Stage 1 is exploratory; it cannot establish a small superiority effect or definitive absence of value. Changing prompts or difficulty after Stage 1 creates a new development version and requires untouched evaluation problems. Stage 3 should be planned from the observed uncertainty, not opened merely because funds remain.

At Stage 1, 48 checkpoints × 3 actions × 4 replays = at most 576 continuations. The separate replication subset adds 432. With 256 charged prefix tokens and a 1,024-token full allowance, each continuation can consume no more than the remaining 768 new generator tokens; actual workloads depend on early stopping and action allocation. The maximum for those 1,008 continuations is about 0.774M generated tokens plus initial prefixes and prefill. At a 2,048-token allowance and 512 charged prefix tokens, the analogous ceiling is 1.55M. A same-model generated selector must fit within this remaining allowance. A different local-judge model needs a separate recorded cost ledger; it cannot disappear from deployment cost. The full screen is conditional on the actual Phase 0 throughput.

Time is measured before scaling: 1M generated tokens at 20 aggregate tokens/s is 13.9 decode-hours; at 50 tokens/s, 5.6 hours. These are arithmetic scenarios, not claims about this Mac or model. Add measured prefill, judge, selector, I/O, and orchestration time. Prefer sequential model loading over swapping or memory pressure.

Jev estimates use provider-billed input tokens and the live verified price, not generator-token counts. Reserve before dispatch using a conservative request-size upper bound and configurable headroom; reconcile actual usage after response. On ambiguous timeout, retain the reservation until reconciled because the provider may have processed it. Retries count as possible new charges. Concurrent workers share one locked ledger. Abort on unknown price, missing usage without a conservative fallback, or insufficient reservation. Cached responses are reused without another billable call. These safeguards support a conservative cap; they cannot undo unreported provider charges.

## 7. Analysis of the mechanism screen

Report counts of sampled problems, finished-before-checkpoint problems, eligible checkpoints, completed/failed replays, and all costs. Plot per-action outcome means with binomial uncertainty; display individual checkpoint intervals without asserting the noisy winning arm is the true optimum.

Three analyses answer distinct questions:

1. **Average action effects.** Compare repair and branch with continue using per-problem differences, not independent-checkpoint tests. Report family/difficulty strata descriptively.
2. **Action heterogeneity.** Use a pooled outcome model with prespecified action-feature interactions. Compare against the best constant policy selected on development. On the replication subset, use the first replays only to select an action and the new replays only to evaluate it; independently estimate the comparator as well. Report uncertainty and acknowledge low power.
3. **Representation signal.** Train cheap versus cheap-plus-Jev versus cheap-plus-local models using grouped cross-validation. Fit vectorizers and all preprocessing inside each training fold. Report held-out log loss/Brier score and estimated selected-policy value. These screen estimates choose whether to invest; they are not confirmatory deployment results.

If only 24 eligible natural checkpoints remain, the next step is more data or a clearly reported futility decision, not a highly flexible learner. Do not use a nominally tiny p-value created by treating replay seeds as independent problems. Do not call a “hindsight oracle” evaluated on the same noisy means a realistic upper bound; that is selection-biased.

## 8. Gated pilot: the comparisons that matter

Reserve 120 fresh test problems, stratified by family and frozen difficulty; test seeds and outputs remain untouched until policies are frozen. At minimum evaluate these four policies with two complete-episode seeds per problem:

| Condition | Policy |
|---|---|
| P0 | Uninterrupted baseline; sham-resume is a separate instrumentation control |
| P1 | Best constant feasible action chosen on development |
| P2 | Outcome-trained policy with cheap features |
| P3 | Same learner plus Jev typed features |

Add P4 (same learner plus independent local typed judge) and P5 (frozen correctness/uncertainty heuristic) once instrumentation supports them. They are required before a strong Jev-specific paper claim. A compatible published control baseline is required before a methods superiority claim; its exact choice depends on whether the final experiment uses step routing, local intervention, or tree search. Do not force unrelated PRM models onto non-math tasks without documenting domain mismatch.

For the narrow acquisition extension, add **P6: selective Jev acquisition** only after P2/P3 are stable. A small gate observes cheap features and predicts the net value of querying Jev, including lost budget or deadline time. Compare always-Jev, never-Jev, random acquisition at the same measured query rate, selective-Jev, and selective-local-judge. Train the gate on independent/nested cross-fitted policy outcomes; it must not inspect Jev features before deciding to buy them. The primary acquisition plot is verified success versus query count, latency, and total cost. Save enough untouched problems to evaluate this extension independently instead of repeatedly tuning on the pilot test.

The pilot primary contrast is P3 minus P2 in independently verified episode success, with equal original-problem weighting. Use problem-level paired differences and problem-level bootstrap, stratified by frozen families. All controller decisions are executed prospectively; scoring a replay table alone is insufficient for sequential claims. Report P3 minus P4 and other comparisons as secondary with adjustment if inferential claims are made.

A 120-problem test is an effect-size and feasibility pilot, not a powered +3 percentage-point trial. Report intervals, discordant-problem counts, and effect sizes. Use the pilot to plan a later test, with a separate held-out set. For a single Bernoulli episode per condition, a rough paired normal approximation is \(n\approx(z_{.975}+z_{.8})^2d/\delta^2\), with discordance \(d\). At \(d=.30,\delta=.03\), this is about 2,617 independent problems. Averaged seeds, stratification, costs, and multiple tests alter the calculation; simulate the actual planned design using pilot variance rather than recycling this number as a guarantee.

## 9. Resource fairness and outcomes

The primary token-constrained experiment fixes the generator's total new-token allowance and context cap, then separately reports judge, selector, and prefill costs. Call it **generator-token matched**, not total-compute matched. Report success against multiple budgets to construct a utility-cost frontier.

A second experiment uses a common wall-clock envelope on the same host/configuration, counting all components and pending calls; or an explicit, measured common resource envelope that includes local inference and hosted spend. Hosted inference hardware is unobserved, so claims of identical FLOPs across Jev/local conditions are not defensible. Report both latency and dollar cost. API waiting may count against a deadline even though it does not consume the local GPU.

Every run records generated tokens by component, resident input tokens, discarded tokens, model-loading overhead, prefill/decode duration where measurable, judge request count/input usage/cost, tool calls, wall time, cache status, and peak memory. Training-data acquisition and model fitting costs are reported separately from per-deployment costs. Offline Jev caching reduces research spend but deployment projections must price one fresh query per required state; a cache hit in the research run is not free general deployment.

Primary endpoint: strict independently verified success. Secondary endpoints: action rates, intervention-caused failures, final-answer extraction failures, p50/p95 latency, success per wall-clock/dollar envelope, action-value calibration, and performance by difficulty. “Innovation,” alignment, and correctness of private internal reasoning are not endpoints of this first study.

## 10. Sequential and transfer extensions

Only after one-checkpoint value is demonstrated, allow at most three interventions and evaluate the complete frozen controller. Acquire a new dataset from states the learned controller actually visits; retain an untouched sequential test set. A one-checkpoint replay model does not validate repeated deployment by itself.

Test shifts separately: new procedural instances; larger structural horizon at fixed task semantics; held-out task family; a different generator; and controller-induced prefix shift. Record both resource allowance and difficulty because longer context or larger budgets change the estimand. Cross-family tests require a family-level holdout. Generalization within one procedural template cannot establish broad OOD reasoning.

Context trimming, stopping, model routing, audit/revisit, and generator RL are separate later interventions. In particular, trimming changes available history, whereas pruning changes allocation across alternatives. Any extension requires new treatment contracts and cost accounting. A semantic redundancy score does not prove that deleted context was causally dispensable.

## 11. Decision rules and deliverables

Advance when instrumentation passes, usable outcome variation exists, replicated interventions offer plausible gains beyond a constant policy, and held-out predictive/control evidence is worth further compute. There is no minimum p-value for the exploratory screen. If gains are uncertain, allocate the next experiment to the uncertainty that could change the decision: more action replications, more independent problems, selector quality, or cost profiling.

Stop or reframe when performance is at floor/ceiling, actions are equivalent at useful budgets, a constant policy dominates, Jev adds no useful representation signal, or gains disappear after cost. A cheap/local controller may still be publishable even if Jev is unnecessary; that claim must follow evidence.

Deliverables are a frozen protocol and deviation log; pinned environment/model manifests; documented intervention and validator implementations; de-identified public/synthetic checkpoint and rollout records with release rights checked; feature/rubric versions; budget ledger; honest grouped analysis; and a results report distinguishing completed runs from plans. Full-model RL, huge benchmark sweeps, and external GPU acquisition are not prerequisites for the first evidence.
