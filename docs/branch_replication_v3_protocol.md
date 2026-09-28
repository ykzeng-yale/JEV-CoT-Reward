# Fresh common-pool replication v3 — frozen development protocol

September 27, 2026; fixed before generating the new tasks or outcomes. Configuration: `configs/branch_replication_v3.json`. This is development replication, not a confirmatory test or evidence of Jev benefit.

## Motivation and estimand

The eight-problem v2 qualification yielded continue 9/16, uniform 10/16, and likelihood, entropy and local semantic selection each 11/16. This motivates replication of branching while directly testing whether semantic selection adds anything to cheap ranking. It does not establish that a semantic judge is useful. All v2 selectors remain in the comparison; no observed winner is promoted. The v1 prospective test stays closed.

Enroll 24 fresh problems (seed 891027), retaining all ineligible checkpoints without replacement. At each eligible checkpoint generate a common pool of three 128-token candidate segments. Freeze uniform, likelihood, entropy, local-semantic and Jev-semantic choices before generating any downstream outcome. Execute four continuations for each candidate and four default continuations: at most 384 actual outcomes. All use the same remaining generator allowance; each hypothetical branch policy pays for the entire candidate pool. Repeats are nested within the original problem, not independent sampling units.

The generator, local selector, checkpoint and budget configuration match v2: pinned local Qwen3-4B-Instruct-2507, 4-bit MLX, checkpoint target 256/cap 384, 2,048 total generated tokens and 128-token final reserve. Local selection is greedy and limited to 128 tokens. Jev receives the same task, retained text and candidate text, using a Choice question with index options. Neither selector sees gold answers, verifier outputs, later continuations, or model likelihood statistics. These direct selectors are not outcome-trained controllers.

## Comparisons and accounting

Primary exploratory contrasts: Jev minus likelihood and Jev minus local semantic. Report every policy against continue and uniform, absolute rates, original-problem paired differences and descriptive family-stratified bootstrap intervals. No confirmatory p-value, winner promotion or equivalence claim from degenerate intervals. Report family-specific results without selecting favorable families for the main conclusion. Use first two repeats to choose an outcome-informed candidate and last two to assess it, then reverse halves; these correlated diagnostics are not deployable policies or oracle bounds.

Record generator tokens, processed prompt tokens, local judge tokens/service time, Jev input usage, acquisition time and USD separately. Shared offline collection cost differs from hypothetical deployment cost. This protocol matches generator caps, not total FLOPs, total dollars or isolated wall-clock latency. Hosted calls may overlap no inference here, and hardware contention invalidates precise speedup claims.

Jev version is pinned to jev-1.13.0 with the existing single central ledger: $1 cumulative development cap and $25 hard project cap. At most one new request per eligible checkpoint, no retries; at most $0.24 conservative reservations. Failed or invalid choices fall back to likelihood and remain reported. Raw hosted responses remain private. Validate the full returned probability distribution; confidence is not a causal-action-value bound.

## Operational bounds and next decision

Cooperative wall limit 14,400 seconds; external supervisor 14,550 seconds; sampled worker RSS cap 8 GiB (not a complete GPU/shared-memory hard bound), combined log cap 32 MiB. Launch only one qualified model worker. Preserve partial data on failure and do not silently relaunch or replace failed subjects.

Require independent token-chain, selector chronology, outcome-label, full-pool cost and hosted-request audit before analyzing success. Frozen source and model identities are saved with the run. If semantic advantage remains absent, investigate candidate signal and replication precision rather than asserting Jev failure in all domains. If an advantage appears, freeze an independent prospective evaluation with strong cheap/local and compatible published baselines before claiming benefit. The complete published-controller, sequential, transfer and training curriculum remains outstanding.

## Secondary analysis amendment, September 27 at 20:56 local

Added while v3a remained live, without inspecting partial success rates. Preserve every frozen primary contrast and the sampled-uniform policy. Additionally integrate uniform candidate choice by averaging all three candidate continuation means within each problem, with the same full-pool cost. Report each policy versus this expected-uniform value using original-problem clusters. This reduces candidate-choice Monte Carlo variability; it is not an oracle, a trained selector or a replacement primary endpoint. It uses no outcome maximization. The conditional averaging proof and limitations are stated in the manuscript. The original v2 report is not overwritten or silently reinterpreted.
