# Repair qualification v2: audited results

Completed September 27, 2026 in 2,440.77 seconds under the external supervisor. All eight scheduled development problems reached a checkpoint; all 80 continuations completed. Independent audit reconstructed 104 calls, exact treatment/prefix changes and final outcomes with zero disagreements. This stage made no Jev requests.

| Action | Successes / continuations | Graph | Arithmetic | Mean episode generated tokens |
|---|---:|---:|---:|---:|
| Continue | 11/16 | 6/8 | 5/8 | 1,371.75 |
| Neutral continuation prompt | 10/16 | 6/8 | 4/8 | 1,490.81 |
| Suffix repair | 11/16 | 6/8 | 5/8 | 1,334.31 |
| Recheck without deletion | 11/16 | 7/8 | 4/8 | 1,461.81 |
| Full-segment reconstruction | 12/16 | 8/8 | 4/8 | 1,433.69 |

There are **eight original problems**, four per family, with two continuations each. Full-segment reconstruction minus continue is +6.25 percentage points, with a descriptive problem-stratified bootstrap range of −18.75 to +31.25 points. This is not evidence of a positive effect or a precise absence of benefit. The graph/arithmetic contrast is exploratory and far too small to establish effect heterogeneity. Historical v1 used different tasks and a different budget; do not attribute the new success rates to increasing tokens.

The experiment establishes executable treatment distinctions and gives concrete failure modes for further development. Last-call length termination occurred in 3/16 continue, 4/16 sham, 2/16 suffix-repair, 1/16 recheck and 2/16 segment-repair episodes. Better completion rates alone are not correctness. Full generation, prefill, deletion and service-time statistics are in the [analysis](../results/action_qualification_v2_analysis.json); [audit](../results/action_qualification_v2_audit.json) records input hashes and scope. RSS is not full unified-memory usage.

## Next scientific action

Proceed with the separately specified common-pool branch-selection qualification. It addresses candidate quality versus ranking, which the repair experiment cannot answer. Do not promote the slightly higher repair count as a selected winning policy. After both qualifications, specify a fixed independent replication with adequate repeated continuations and fresh problems to distinguish potentially useful state dependence from rollout noise. Strong compatible published-controller comparison remains required before broad method claims.

The original research goal remains open. No Jev benefit, sequential policy gain, transfer result or training result has been shown by this stage.

## Public terminal reproduction

`artifacts/action-qualification-v2-terminal` contains original tasks, terminal texts, labels and costs. Run `scripts/check_control_release.py` on that directory to reproduce all 80 labels without model inference or API access. This terminal-only release does not reproduce full prefix/treatment auditing; the original detailed logs remain preserved locally. Labels were not regenerated or edited for release.
