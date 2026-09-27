# Mechanism v1: completed exploratory screen

2026-09-27. **Do not promote automatically to the larger pilot.** Instrumentation passed, but the prespecified requirement for useful replicated action differences and preliminary decision value has not passed. This is a negative/inconclusive screen for this model, task distribution, 1,024-token allowance and fixed interventions, not evidence that all semantic control is ineffective.

All 24 scheduled problems reached online checkpoints. All 288 continuations completed. Independent reconstruction checked 773 generation calls and recomputed every final outcome: zero disagreements, zero unknown-work calls. Both Jev and the separately prompted local 4B judge supplied all seven features at every checkpoint. The local judge shares the generator backbone and is not an independent truth oracle.

| Action | Verified successes | Rate |
|---|---:|---:|
| Continue | 47/96 | 48.96% |
| Repair | 40/96 | 41.67% |
| Branch | 37/96 | 38.54% |

Original-problem-weighted repair minus continue was −7.29 percentage points; branch minus continue was −10.42. The corresponding descriptive family-stratified bootstrap ranges were [−14.58, 0.00] and [−17.71, −3.13] points. Four repeats within each problem are not independent problems. These exploratory comparisons do not establish broad superiority or absence of useful checkpoint-specific effects.

Grouped cross-fit Ridge selected-policy replay success was 46.88% with cheap features, 45.83% with added Jev and 45.83% with added local features. The training-selected constant and family-constant policies both achieved 48.96%. Jev minus cheap was −1.04 points, with a descriptive range [−4.17, +2.08]. Overlapping training folds and analysis refinements during collection prevent interpreting these ranges as general nominal-coverage intervals.

The prespecified forest adaptations, sparse binary gates, entropy heuristic and validity heuristics did not exceed always-continue on this replay screen. The full aggregate table is in [the machine-readable summary](../results/mechanism_v1_summary.json); no best-looking learner is promoted by selecting among these results. This is replay evaluation, not a fresh prospective deployment trial.

Generation consumed 201,006 tokens and 4,899.32 seconds of recorded model service; run wall time was 4,912.38 seconds. Local feature acquisition added 1,636 generated tokens and 50.75 service seconds (56.07 wall seconds). Cumulative recorded Jev expenditure across the project is $0.00124614, far below the $25 ceiling. Generator-token matching does not make these configurations total-compute or latency matched.

## Remaining decision

No scaling, sequential-control, transfer, innovation or Jev-efficacy claim is warranted. Inspect the completed family-stratified evidence and uncertainty before selecting any fixed-count replication. A prospective policy experiment, if pursued to test this negative finding, must freeze the learner and policies first and use new problems; it cannot be presented as a passed positive-signal gate. Fixable analysis or export issues are engineering work, not scientific futility. Public local-trace export and the final project report remain outstanding; this gate document does not declare the full project complete.
