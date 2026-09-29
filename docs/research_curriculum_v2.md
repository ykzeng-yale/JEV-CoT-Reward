# Research curriculum v2: reopen the scientific program

## September 29 design boundary

The full remaining-stage ledger, arm definitions, stop/go rules, cost contract,
and power plan now live in `docs/remaining_experiment_program_v1.md`. Use it as
the operational source of truth for the remaining experiments; it is not a
claim that every conditional stage is currently eligible to run. Its paired
binary planning calculation is implemented in `scripts/paired_binary_power.py`
and checked by `tests/test_paired_binary_power.py`.

The prior-art audit in `docs/prior_art_gap_audit_2026-09-29.md` finds direct overlap with our core prior proposal: *Calibration Is Not Control* already defines intervention advantage, performs same-prefix branching, and evaluates action-conditioned controllers; *CausalFlow* already performs step-level counterfactual repair. *interwhen* and adaptive verifier-call allocation add further overlap. **No methods-level novelty gap is currently established.** Do not proceed with a generic “Jev judge controls CoT” or “learn action value by branching” claim. Only a source-comparison empirical study (Jev versus cheap/local judges on a genuinely new transfer or horizon setting) might remain, and this is not yet a validated contribution.

The eight-operand arithmetic screen did not pass its frozen event-count gate (15 errors versus 20 required), so do not run correction rollouts or repeat arithmetic screens with the same checkpoint mechanism. Next, qualify a compatible published baseline and a fresh action/domain contract. *interwhen* is a strong candidate, but its official path uses vLLM and public Maze/Game24 datasets not currently cached; the local MLX runtime is not a faithful upstream reproduction. Name any port an adaptation, independently test the validator, and set the budget/sample-size before inference. A cluster model download requires a fresh quota/path check and actual runtime qualification.

The TypeSafe Master Customer Agreement current §2.3(b) may constrain training a controller from Jev outputs or developing a similar/competing service. Hold that Jev-dependent step pending written clarification. Continue local-only baseline, action-value and theory work without spending the Jev ledger.

**September 29 counterfactual-credit update:** primary-source review of *Policy-Conditioned Counterfactual Credit for Verifiable RL* (PCCC; arXiv:2606.05263) shows direct overlap with counterfactual deletion/substitution/tool perturbations, terminal verified outcomes under frozen continuation, and training from action-specific credit. *Verify, Repair, Repeat, or Stop?* (arXiv:2607.17641) further preempts broad noisy-verifier repair/stop theory. The new dated comparison is in `docs/prior_art_gap_audit_2026-09-29.md`. Do not describe our general intervention-value estimand, counterfactual training, or repair/stop propositions as new.

The task/action screen in `docs/task_action_candidate_matrix_2026-09-29.md` leaves CLI coding as a conditional candidate only: the available failure corpus is annotation-only, without raw traces, and interventions have not been specified. Math/Game24, long-context QA, web/tool work, and scientific discovery either fail the event/independence gates or have direct method/application overlap. **No candidate currently passes all gates; do not launch another inference or GPU task.** Next work is to qualify a fresh-trajectory CLI action contract and independently pin its test evaluator/baselines. Jev remains held pending written terms clarification.

The September 29 Terminal-Bench source check improves only evaluator feasibility: official materials describe task-specific test scripts and a continuous tagged dataset registry. Pinning a release and auditing its tests, checkpoint visibility, action semantics, and prior-art overlap are still required. This check does not promote CLI work to an efficacy study.

**September 29 direct-Jev update:** *REFLEX with Jev* (arXiv:2609.26532v1) already evaluates Jev as an in-trajectory bounded-action layer with confidence-gated fallback and a cheap generative cascade comparator. *Jev vs. LLMs as Rubric Judges* (arXiv:2609.29769v1) studies judge-label agreement and correlated errors. These findings close off broad “first Jev supervisor,” confidence-routing, or judge-quality claims; neither evaluates counterfactual interventions on visible reasoning prefixes with terminal outcome utility. Any remaining candidate is a narrow empirical comparison in that setting, and must survive *Calibration Is Not Control*/*CausalFlow* overlap, a compatible baseline audit, and contract clarification. Full dated details are in `docs/prior_art_gap_audit_2026-09-29.md`.

**September 29 second direct-Jev update:** *Jev for Scientific Decisions* (arXiv:2609.24965), *JEV-Star* (arXiv:2609.27331), and *KITE* (arXiv:2609.27535; official MIT code) further preempt broad claims about Jev scientific semantic decisions, Jev plus planning for long-horizon agent control, and Jev in intervention studies with downstream population outcomes. They do not study actions on a visible LLM reasoning prefix with independent terminal-task verification. The remaining candidate is this precise empirical comparison, not “Jev for science/exploration/long-horizon control”; no methods novelty is established. See the dated prior-art audit for detailed evidence and limits.

**September 29 interwhen source audit:** the pinned interwhen TTS Game24 monitor already checks visible reasoning prefixes using periodic same-model expression-extraction side streams, executable validity/dead-end checks, repair feedback, and early stop. This directly overlaps the original real-time CoT judge/repair/stop proposal. Its side-stream is additional generation and its final score reuses the monitor's checker; the checker also evaluates unrestricted Python expressions. A faithful outcome comparison therefore needs an independently implemented exact terminal verifier and cost-matched extraction/sham arms. Added `src/jev_control/game24_exact.py` with an AST/Fraction evaluator and regression tests; 21 targeted tests pass. The same audit found no active Bouchet jobs, `vllm` is not installed as a module, and all owned local Macs have competing workloads or tight disk. Do not launch a Game24 reproduction just because compute is available. Full source/runtime analysis: `docs/interwhen_game24_baseline_audit_2026-09-29.md`.

User authorized September 27, 2026. The original broad research question remains open. The completed v1 branch is a small negative/inconclusive diagnostic, not completion of the whole program. Preserve its records and test designation. The broad research objective supersedes the previous branch-completion framing; scheduled authorization continues work even while app goal mode is paused.

## First baseline/design audit

The v1 implementation supports valid within-learner feature comparisons for its fixed actions. It does not supply all comparators needed for a general reasoning-control claim:

| Comparison | Existing evidence | Required next-stage treatment |
|---|---|---|
| Uninterrupted versus pause/resume | Separate small development diagnostic | Repeat after any backend, template or checkpoint change |
| Static continue/repair/branch | All-arm mechanism replay; only continue deployed prospectively | Include all three in fresh action qualification and best-static selection using development only |
| Random/rate-matched intervention | Restricted permutations in exploratory replay | Fresh randomized deployment, with rates frozen using development data |
| Cheap versus semantic representation | TF-IDF/numeric crossed with Ridge, forest and sparse gates in replay; Ridge prospective | Retain strong cheap text baseline; add a fixed local embedding comparator when runtime qualified |
| Jev versus local judge | Same rubric; local judge shares generator backbone | Retain matched rubric and explicitly measure acquisition cost; qualify a separate local judge before claiming cross-model robustness |
| Published control method | Forest/DIAL-inspired adaptations only | Choose a compatible public method, pin its code/config, test faithful action semantics, and label adaptations rather than claiming reproduction |
| Search/selection quality | Two 64-token candidates ranked by mean base likelihood | Compare random and likelihood selectors to an independently prompted local verifier under the same candidate pool and charged budget |
| Repair quality | At most 128-token suffix rollback, not necessarily a complete reasoning segment | Compare continuation, sham instruction and whole-segment/check-targeted repair on natural development states |
| Compute fairness | Generator tokens matched; prefill and judging separately measured | Report success/cost curves, total processed tokens, actual latency, service/model identity; never equate remote/local FLOPs from token counts |

RSD-style model routing requires a draft/target distinction; it is not a fair primary baseline for a fixed single-generator action experiment. Math-specific PRMs need a math-compatible domain and model qualification. Choose the compatible controller after the action/domain contract is fixed, before test collection. Re-check original papers and pinned code when implementing that baseline.

## Curriculum and stage transitions

1. **Design diagnosis (now).** Audit exact action semantics, task difficulty, truncation, false outcome labels, information access and cost. Estimate whether apparent action preferences repeat using only designated mechanism development data. No new Jev expense needed. Separate lack of information, weak interventions, measurement error and small samples.
2. **Action qualification.** Freeze a small development-only grid of natural checkpoints, budget levels and stronger action definitions. Include static and sham controls. Tasks should span independently verified intermediate difficulty, not be selected by favorable Jev scores. Preserve all sampled tasks and failed/ineligible cases. Synthetic known-error states may test instrument sensitivity separately, never stand in for naturally occurring benefit. Profile local runtimes before setting the bounded rollout envelope.
3. **Representation and controller qualification.** Once useful action differences are reproducible, cross the same learners/actions with cheap text/logit features, local semantic features and Jev. Include frozen-rate random interventions, best static, a compatible published baseline and direct semantic heuristics. Use independent repeats to select and assess noisy action preferences. Do not train on a sample maximum as certain ground truth.
4. **Frozen prospective evaluation.** Determine sample size from development estimates of problem-level variance and a declared practically relevant effect, within available runtime. Freeze hypotheses, policies, seeds, multiplicity treatment and costs before fresh tasks. A null result with wide uncertainty leads to an explicit precision/design decision, not automatic abandonment of the broad research question.
5. **Sequential/horizon/transfer curriculum.** Qualify one-checkpoint → repeated intervention → longer dependency horizon → held-out task family → receiver model transfer. Each changes the estimand/distribution and needs fresh independent evaluation. Preserve a fallback and measure harmful interventions. Budget and judge calibration are assessed on visited states, not assumed from pooled offline calibration.
6. **Training and manuscript.** Only after appropriate evidence, test controller-guided rollout allocation with independent outcome rewards; account for sampling/selection bias. Maintain theory, counterexamples and an arXiv-ready source draft alongside work. A complete manuscript may report negative results, but must meet evidence/novelty standards and distinguish proposed extensions from executed experiments. Do not upload to arXiv without explicit publication authorization.

A failed branch triggers a recorded choice between replication for precision, changing a scientifically motivated mechanism, or narrowing a claim. It does not automatically close the active broad goal. Conversely, broad scope is not authorization to run a large grid without a defensible question.

## September 29, 2026 — queued published-baseline qualification

The failed task/action opportunity gate remains in force. In parallel, a bounded
runtime and development-outcome qualification is now submitted for the published
InterWhen K-stable Game24 early-stop comparator: 96 frozen public training tasks,
continuation versus `k=2` stopping, and independent AST/Fraction terminal
evaluation. See `docs/interwhen_kstable_game24_v1_protocol.md` and Bouchet job
27857096. The receiver checkpoint differs from InterWhen's original model, so
this is an adaptation. Its outcomes cannot establish a Jev benefit, a new method,
or a surviving application gap; that still requires natural opportunities in a
held-out task family, a stronger compatible control, and TypeSafe's written
terms clarification before Jev-dependent work.

## First executed diagnostic

`results/curriculum_action_replicability_v1.json` uses only the 24-problem mechanism development release. For every fixed split of four repeats into two selection and two evaluation repeats, select the best observed action per checkpoint (ties favor continue), then evaluate it on the other two repeats. Across six overlapping splits the average paired advantage over continue is −4.51 percentage points; no split is positive. These splits are correlated and provide no confidence interval. The selector sees development outcomes and is not deployable; with two selection repeats it is noisy and cannot rule out true heterogeneity.

This finding supports investigating action quality and replication depth before attributing the result to Jev feature inadequacy. It does not justify searching the completed prospective test for favorable subgroups. That test stays closed to training and model selection.

## Resources and evidence discipline

Maintain the existing single Jev spending ledger and $25 cumulative ceiling. Keep the $1 cumulative development-stage cap for now; larger existing caps are conditional, not automatic. Prefer the qualified local MLX backend. The two remote Macs support SSH/CPU tasks; remote inference still needs runtime, memory and disk qualification. No new model downloads or services are implied by a connection check.

Do not overwrite v1 frozen configs, model source snapshots or original reports. New actions, data, budgets and policies receive new versioned manifests. Update this curriculum with measured transitions and GitHub artifacts; the active goal remains open until the broader staged program reaches a justified final scope.
