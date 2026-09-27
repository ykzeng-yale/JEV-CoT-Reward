# Research curriculum v2: reopen the scientific program

User authorized September 27, 2026. The original broad research question remains open. The completed v1 branch is a small negative/inconclusive diagnostic, not completion of the whole program. Preserve its records and test designation. A new active goal supersedes the previous branch-completion framing.

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

## First executed diagnostic

`results/curriculum_action_replicability_v1.json` uses only the 24-problem mechanism development release. For every fixed split of four repeats into two selection and two evaluation repeats, select the best observed action per checkpoint (ties favor continue), then evaluate it on the other two repeats. Across six overlapping splits the average paired advantage over continue is −4.51 percentage points; no split is positive. These splits are correlated and provide no confidence interval. The selector sees development outcomes and is not deployable; with two selection repeats it is noisy and cannot rule out true heterogeneity.

This finding supports investigating action quality and replication depth before attributing the result to Jev feature inadequacy. It does not justify searching the completed prospective test for favorable subgroups. That test stays closed to training and model selection.

## Resources and evidence discipline

Maintain the existing single Jev spending ledger and $25 cumulative ceiling. Keep the $1 cumulative development-stage cap for now; larger existing caps are conditional, not automatic. Prefer the qualified local MLX backend. The two remote Macs support SSH/CPU tasks; remote inference still needs runtime, memory and disk qualification. No new model downloads or services are implied by a connection check.

Do not overwrite v1 frozen configs, model source snapshots or original reports. New actions, data, budgets and policies receive new versioned manifests. Update this curriculum with measured transitions and GitHub artifacts; the active goal remains open until the broader staged program reaches a justified final scope.
