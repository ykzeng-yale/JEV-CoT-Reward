# ScienceWorld unknown-plant genetics candidate audit

**Date:** September 29, 2026, 21:10 EDT
**Scope:** static inspection of the pinned ScienceWorld 1.3.0 source plus the existing audited split/replay record. No model inference, Jev request, gold-path execution, hidden score, or held-out outcome was used.

**Integrity verification:** on September 29, 21:18 EDT, all four listed member SHA-256 digests and the archive digest were recomputed directly from the frozen tarball and matched this report and the structured JSON record. The `PythonInterface.scala` member hash was corrected before publication of this audit result.

## Finding and decision (revised September 30, 05:31 EDT)

The earlier promotion of `mendelian-genetics-unknown-plant` as the leading information-to-classification candidate is **suspended** by source-level validity concerns. The broader static screen is documented in `docs/scienceworld_candidate_source_leak_audit_2026-09-30.md`. Static review of the pinned task generator indicates it may be source-decodable: the same `domOrRec` variable chooses whether the prompt uses a fixed `valueDominant` or `valueRecessive` phenotype and which answer box is correct. The phenotype pairs appear fixed by public plant/trait source definitions. The first dynamic job `27937039` failed before importing the simulator because `scienceworld/version.py` is generated from the JAR manifest; no variation was loaded. Its logs and accounting are preserved. The retry `27937234` also failed before environment import because its SBATCH root still selected the original v1 input tree, repeating the missing generated-version exception. Both failures are preserved; no variation loaded and no third same-cause retry was submitted. There is no dynamic observation result. Do not use this candidate for action efficacy or Jev comparisons. If the visible query confirms the source rule, reject the current task/action estimand or create and separately validate a randomized mapping task before any inference study.

The prior statement that the variation ID determines the hidden trait mapping is retained as a source/runtime property, but it does not establish that the *answer to the visible classification question* is hidden. This distinction is central: hidden randomized plant state and a source-fixed prompt-to-answer relation can coexist. The dynamic audit tests visibility of the latter; it does not measure model accuracy.

The corrected audit code writes only aggregate train/dev parse and source-rule agreement counts plus hashes. The independent auditor enforces 360/360 row coverage, exact source/protocol digests, no test IDs, no stored raw observations or score/outcome reads, and no inference, Jev, network, or gold-path calls. Focused parser/firewall/auditor tests currently pass 13/13. Detailed first-attempt/retry receipts are `runs/bouchet-scienceworld-mendelian-semantic-cue-v1-control/submission.json` and `runs/bouchet-scienceworld-mendelian-semantic-cue-v1-control/retry1/submission.json`.

The intervention opportunity is weaker than the prompt wording alone suggests: the required `GoalSequence` terminates on the correct answer box. Plant-growth and life-stage goals are in the unordered/optional goals. Thus simulator aggregate score can reward optional cultivation without improving classification. Any outcome study must score final classification correctness independently as primary; report official progress/score only as decomposed secondary outcomes. An optional growth/offspring observation is a candidate information action, not an already-qualified treatment.

## Source and split evidence

The audit uses the source archive frozen for the ScienceWorld prefix-replay run: ScienceWorld commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52`; archive SHA-256 `b5977fdf71fecaa3cc3985caa9a2f295569db96031ed453041f0e3eef3a6d41b`. Relevant member hashes:

| Source file | SHA-256 |
|---|---|
| `TaskMendelianGenetics2.scala` | `9e1dd24a24900475887a60b9a1c6df022517825d3972f175de1c81b01a542890` |
| `PythonInterface.scala` | `86ecec51c9e37fb7881927afb54d18afca452cea92bbfe7d6c718989165fd9d77` |
| `TaskMaker1.scala` | `cad10396e6734bad6984001020a13e1ce613ea7b2b43bebfb5ec0b3184d56416` |
| `UnknownPlants.scala` | `43b3bab8eed059e6646ad2a6560b1749d498f9028454b1f4abbfd947846497bd` |

`TaskMendelianGenetics2` constructs four unknown plants; for each, it enumerates four traits and both dominance states. The combination product then adds five flower-pot-name variants and three answer-box-color pairs: `4 × 4 × 2 × 5 × 3 = 480` variations. The official general split groups these as 240 train, 120 development, and 120 test variations (test IDs 360–479). During `PythonInterface.reset`, the simulator seeds Scala `Random` with `variationIdx` before rebuilding `TaskMaker1`; the hidden trait map is therefore deterministic for a given official variation ID and varies across IDs. Keep that ID evaluator-side: verify the actual prompt/API wrapper never gives it to the controller, and describe the public deterministic source as a leakage surface rather than as cryptographic secrecy.

The completed structural preflights inspected only three test variation IDs for this task—360, 420, and 479—and used no task labels or agent outputs. Conservatively exclude those IDs from untouched confirmatory analysis; 117 nominal test variations remain uninspected for outcomes. These are all one task template/plant family, so they can support at most a within-template held-out-variation result, not task-family transfer or general scientific discovery. The audited 25-step prefix replay is simulator feasibility only and does not qualify model-generated prefixes or action effects.

The split-size calculation follows the pinned source, not an outcome file. The gold-path archive was not extracted or executed. Reproduce the evidence from the pinned archive and files above plus `PythonInterface.getSets`; the previous run receipt and no-label structural exclusions are `runs/scienceworld-prefix-replay-v1-control/submission.json` and `docs/scienceworld_task_objective_audit_2026-09-29.md`.

## Precision boundary

Using the repository's paired-binary normal planning utility, one primary contrast, two-sided α=.05, 80% power and planning discordance 0.30:

| Available independent variation units | Optimistic MDE | Units for 5 pp | Units for 3 pp |
|---:|---:|---:|---:|
| 117 untouched official test variations | 14.04 pp | 940 | 2,614 |

These calculations are planning approximations; they do not account for within-template dependence and do not turn 117 episodes into 117 task families. A Jev-increment claim at 3–5 pp is out of scope for this official test. At most, it could evaluate a large-effect, single-template mechanism after train/dev development, a frozen local/published baseline, and full action/cost qualification. Keep the closed v1 test closed and all official test outcomes sealed until the design is frozen.

## Next gates

1. Preserve and independently audit the already-submitted DiscoveryWorld/ReCoMA baseline `27880628`; do not add a competing GPU generation job while it is pending.
2. Then, on ScienceWorld train/dev only, check observation/variation-ID leakage, candidate action naturalness and frequency, exact classification evaluator, natural baseline checkpoint prevalence, and the full executable prefix-replay protocol. A task action must change available evidence/state; optional score gain by itself is insufficient.
3. Compare continuation with one prespecified executable measurement action and a call/time-matched sham. Add a rate-matched randomized policy and a strong action-specific local comparator before any Jev test. Count all actions, discarded branches, model tokens/prefill, simulator calls, failures, latency and GPU use.
4. Re-audit prior art and confirm the exact task/action distinction. Jev output-conditioned work remains blocked on written TypeSafe terms clarification; no Jev call is authorized by this audit.
5. If the train/dev opportunity or action gate fails, reject this route. If it passes, the held-out test supports only a large-effect within-template mechanism evaluation, not a small-increment confirmatory claim.

No positive result is implied by this source audit.

## Train/dev API metadata firewall result — September 30, 00:05 EDT

The frozen no-model run **27927761** tested only the data-flow boundary over the official train/dev IDs 0–359 (240 train, 120 dev). In every case, evaluator-side `info` included `variationIdx`, `score`, and `reward`; the controller adapter was called with the observation string alone and the serialized controller input did not contain the `variationIdx` field. The result retained only hashes, field names, and booleans. No test IDs, label/outcome values, raw observations, raw info payloads, model inference, Jev, network calls, or gold paths were used.

Slurm records `FAILED 1:0` after 2m22s on 2 CPUs/8 GiB, peak RSS 3,834,128 KiB, zero GPU-hours. The runner had completed successfully; the remote auditor produced a false negative because it parsed and reserialized the protocol JSON before checking a digest defined over the exact frozen bytes. The original remote failure is preserved. A corrected independent local audit hashes the exact frozen protocol bytes and passes 18/18 checks; focused tests pass 9/9. Do not resubmit. This establishes the adapter's direct metadata separation on train/dev, but does not prove semantic observation sufficiency, absence of inferential leakage, natural information-action opportunities, or efficacy. The untouched official test remains sealed.
