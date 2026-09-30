# ScienceWorld conductivity visible-index audit protocol

**Protocol:** `scienceworld_conductivity_visible_index_v1`
**Source:** ScienceWorld `e8216d6044e8e39be9fcb185e3b2dfb602584b52`; archive SHA-256 `b5977fdf71fecaa3cc3985caa9a2f295569db96031ed453041f0e3eef3a6d41b`
**Design:** no-model, no-label, train/dev-only reset-observation audit.

## Scientific question

Does the task text plus initial controller-visible observation reveal enough of the public task-combination dimensions to reconstruct the official variation index? The task source chooses the unknown-substance conductivity label through `Random.nextFloat()` and the Python interface seeds task-maker construction with `variationIdx`. If a simple visible-index decoder recovers the exact index, the public deterministic construction path may make the purportedly hidden label source-reconstructable. This is a task-validity gate, not model accuracy or intervention efficacy.

## Frozen source rule and decoder

The frozen source defines 24 sorted unknown-substance letters, one power-source option, five `partToPower` choices (three bulb colors, motor, buzzer), and five adjacent answer-box color pairs. Its nested enumeration orders unknown substance first, part second, and answer-box pair last. The pre-registered decoder therefore computes:

`variation_idx = letter_index * 25 + part_index * 5 + answer_box_pair_index`.

It extracts the unknown-substance letter and unique answer-box pair from the visible text, then requires exactly one of the five named power components in the initial observation. If the text lacks a unique component or pair, the decoder returns unresolved; it does not guess. Member hashes and source snippets are checked before scanning.

## Split, outputs, and exclusions

The task has 600 source-defined variations. The official general split is frozen as train `0–299`, development `300–449`, and test `450–599`. The remote runner verifies the official API's train and development lists before scanning and never asks the API to enumerate or load the test list.

For each of 450 train/dev resets the output stores only split, variation identifier, initial-observation SHA-256, boolean feature visibility, candidate count, and whether the decoded index equals the evaluator-side variation identifier. It does not serialize raw text, conductivity values/labels, `info` values, scores/rewards, gold paths, model outputs, or Jev responses. The evaluator-side ID is used only for the aggregate decoder check and never passed to the decoder. An independent local auditor verifies exact input-protocol bytes, source-member digests, split coverage, aggregate counts, and zero excluded calls.

The scoped interpretation is deliberately limited: complete index recovery would show that this declared decoder recovers the official combination index from visible state. Partial recovery may still matter for a stochastic label source; no recovery by this decoder cannot prove that every possible semantic or seed-reconstruction attack fails. No efficacy comparison is eligible from this gate alone.

## Executable records

- Frozen protocol: `configs/scienceworld_conductivity_visible_index_v1.json`
- Runner: `scripts/run_scienceworld_conductivity_visible_index.py`
- Independent auditor: `scripts/audit_scienceworld_conductivity_visible_index_result.py`
- Batch request: `cluster/bouchet/scienceworld_conductivity_visible_index_v1.sbatch`
- Submission/accounting receipt: `runs/bouchet-scienceworld-conductivity-visible-index-v1-control/submission.json`
- Output is interpreted only after Slurm terminal accounting and the independent audit both pass.
