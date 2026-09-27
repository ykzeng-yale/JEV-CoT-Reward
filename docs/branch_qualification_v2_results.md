# Audited branch qualification v2

Eight fresh development problems, three candidates per checkpoint, two shadow continuations per candidate and default continuation. The worker completed with exit 0 after 1,611.29 seconds. All 126 generation calls, 24 candidates and 64 distinct outcomes passed independent audit with zero label disagreements. Public terminal artifacts reproduce the counts without inference.

| Frozen policy | Verified success | Mean generator tokens | Extra local judge output tokens |
|---|---:|---:|---:|
| Continue | 9/16 | 1413.50 | 0 |
| Uniform candidate | 10/16 | 1560.69 | 0 |
| Likelihood | 11/16 | 1579.13 | 0 |
| Entropy ranking | 11/16 | 1566.25 | 0 |
| Local semantic | 11/16 | 1520.31 | 7 |

Arithmetic counts are 1/8, 3/8, 3/8, 3/8, 3/8 respectively; graph counts are 8/8, 7/8, 8/8, 8/8, 8/8. Local parsing failures: zero. No Jev calls were made.

The three informed selectors each exceed continue by a descriptive 12.5 percentage points, with a problem-stratified bootstrap range [0, 25] points. This is eight independent problems, multiple exploratory comparisons, and a potentially degenerate bootstrap—not confirmatory evidence. Local semantics show no aggregate increment over cheap ranking. Both split-repeat outcome-informed diagnostics achieve 62.5%, equal to uniform; these are noisy correlated diagnostics, not an oracle or deployable strategy.

The common pool and reused continuations deliberately couple policy comparisons. Every hypothetical branch pays the full pool, including discarded candidates; local inference adds cost. Generator allowance is matched, total compute is not. Entropy ranking is only a GUARD-inspired component. These findings justify a frozen fresh development replication, not a claim of Jev efficacy, transfer or scientific innovation.

Artifacts: `results/branch_qualification_v2_audit.json`, `results/branch_qualification_v2_analysis.json`, `artifacts/branch-qualification-v2-terminal`. Next design: `docs/branch_replication_v3_protocol.md`.
