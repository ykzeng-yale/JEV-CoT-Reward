# Completed prospective diagnostic and scale decision

On September 27, 2026, four policies frozen before test collection were evaluated on 24 fresh procedural problems (12 arithmetic constructions, 12 weighted paths). The frozen Qwen3-4B-Instruct-2507 generator used local MLX 4-bit inference, a 1,024 generated-token allowance and at most one intervention. This is explicit explanation control in a non-thinking instruction model, not a study of inaccessible internal reasoning.

| Policy | Verified successes | Continue / repair / branch | Mean generator tokens | Mean acquisition time |
|---|---:|---|---:|---:|
| Always continue | 11/24 | 24 / 0 / 0 | 947.9 | 0 s |
| Cheap TF-IDF + numeric features | 11/24 | 17 / 3 / 4 | 948.8 | 0 s |
| Same learner + Jev | 11/24 | 14 / 5 / 5 | 949.1 | 0.215 s |
| Same learner + local judge | 12/24 | 12 / 8 / 4 | 965.0 | 2.146 s |

The Jev versus cheap paired outcome difference is zero, with no discordant problems. The empirical bootstrap therefore degenerates to [0,0]; **this is not an equivalence interval or zero population uncertainty**. The local judge difference is one success (4.17 percentage points), with three discordant problems and a descriptive problem-stratified bootstrap range of −8.33 to +16.67 points. No benefit is established. Arithmetic success is 3/12 in every condition; the local difference occurs on graphs (9/12 versus 8/12).

Identical selected actions share one recorded continuation per problem, with a common continuation seed. There are 42 actual selected-action outcomes representing 96 hypothetical policy episodes, not 96 independently generated trajectories. Decisions were recorded before those outcomes. The fitted Ridge policies use the original alpha=10 setting and the preceding 24-problem mechanism screen; no test-result policy selection was performed.

## Cost and integrity

Actual collection generated 36,762 tokens across 142 durable calls, using 883.16 summed model-service seconds. The bounded supervisor recorded approximately 895.67 seconds. Per-policy costs include the work each policy would incur, including shared-prefix, discarded generation and its own feature acquisition; actual collection exploits sharing and is a different quantity. Local judging adds an average 68.46 generated tokens per problem. Jev averaged $0.0000425985 per problem. Generator allowances are matched; total compute and wall-clock latency are not.

The complete prospective audit reconstructed token/prompt chains, replayed the independent terminal verifier, recomputed frozen predictions, checked decision ordering and resource accounting, and found zero outcome disagreements. Two audit-only compatibility fixes were needed: explicitly requesting token IDs from Transformers, and restoring original dictionary insertion order when reconstructing the judge prompt from sorted durable JSON. Original experimental records and labels were unchanged. Regression tests cover both fixes.

[Analysis](../results/prospective_v1_analysis_v2.json) and [integrity report](../results/prospective_v1_integrity_v2.json) include source/input hashes. Earlier v1 local reports are retained privately. The assignment sub-report describes only its narrow layer; the outer audit performs the additional token/prediction/outcome checks. This is recorded-contract verification, not fresh model inference replay or externally attested timestamps.

## Decision

**Do not promote this branch to the large pilot, sequential control, transfer, or generator RL.** The preceding mechanism screen also found no Jev increment (cross-fit Ridge 45.83% versus cheap 46.88%; always continue 48.96%). These two small stages support stopping expansion of this specific configuration, not concluding that Jev or intervention control is universally ineffective.

A future study needs a separately justified development change—such as interventions that show reproducible action-value heterogeneity or a task setting requiring evidence acquisition—followed by a newly frozen independent test. Searching more test seeds for a favorable result is not that justification. Strong published-controller reproduction, broad OOD claims, trace trimming and scientific discovery remain untested, gated extensions.

Cumulative recorded Jev usage across the project is $0.002268504 for 55 successful requests, under the $25 authorization. No additional API allocation, model downloads or compute purchase is needed to finish this branch. The theory remains conditional, standard results specialized to the experiment, not evidence of a practical Jev gain.

## Public reproduction

Terminal texts, original tasks, labels, selected actions and costs are released for both stages, without raw hosted outputs or private paths. Run:

```sh
.venv/bin/python scripts/check_control_release.py artifacts/mechanism-v1-terminal
.venv/bin/python scripts/check_control_release.py artifacts/prospective-v1-terminal
```

These commands require neither a model nor an API key and reproduce 288/288 mechanism and 42/42 prospective terminal labels. This public endpoint release does not reproduce private feature fitting or the full token ledger; those were audited locally. The repository provides the acquisition, fitting, running and auditing code for a new fully instrumented experiment. Hosted feature records and the fitted private pickle are intentionally excluded; public endpoint reproducibility must not be described as full feature-level reproducibility.
