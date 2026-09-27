# Public local-model diagnostic records

These artifacts release the recorded local-model tasks, reasoning text, token IDs, outcomes, and resource measurements from completed development runs. They are **not** a held-out policy evaluation, and they do not demonstrate a Jev improvement. Labels and model outputs were copied from the original records without regeneration or rescoring.

| Run | Original independent problems | Recorded continuations | Generator-token allowance | Recorded successes: continue / repair / branch |
|---|---:|---:|---:|---|
| [phase0-20260927](phase0-20260927/) | 6 | 36 | 512 | 2/12, 2/12, 1/12 |
| [phase0-budget1024-20260927](phase0-budget1024-20260927/) | 4 reused development problems | 12 | 1,024 | 0/4, 2/4, 1/4 |

The second run reuses the first four problems and partly the same seeds. Do not treat it as an independent replication or pool these two runs as ten independent problems. Repeated continuations within a problem are correlated for task-population inference; their 48 total continuation records are not 48 independent problems.

The separate [development-v1-20260927](development-v1-20260927/) release contains **12 new original problems and 48 baseline/sham episodes**, with two seeds per condition and a 1,024-token allowance. It uses the recorded `development-v1-online-boundary` protocol and no Jev calls. Its full local outputs, calls, per-episode checkpoint fields, labels, and original summary are included. It must not be silently pooled with either earlier all-arm experiment: checkpoint capture, prompt framing, and backend version differ.

## Included files

- `tasks.jsonl`: original task IDs, prompts, families, and problem data, allowing independent checking of answers.
- `checkpoints.jsonl`: saved prompt and retained token IDs, initial generation, original checkpoint hashes, observable state, and available local token statistics. Hosted judge objects are removed.
- `outcomes.jsonl`: every original episode record, including answer text, original labels/reasons, per-call token IDs and timing, losing-branch work, repair overhead, seeds, and costs. Where the original schema had no separate episode ID, identity is the unchanged tuple `(problem_id, action, repeat)`.
- `manifest.json` and `summary.json`: original experiment settings and recorded aggregates, excluding local paths and hosted service data. The model's repository and revision replace its machine-specific cache location.
- `local_judge.jsonl`, when available: independently prompted **local-model** responses and probabilities, generated without access to Jev outputs. These are separate diagnostic judgments, not truth labels.
- `review_audit.json`, when originally saved alongside the run: the recorded implementation audit.
- `source/` and `source_manifest.json`: historical local experiment/verification source and explicit provenance. The credential/network adapter is omitted; its original hash is retained.
- `release_manifest.json`: original input hashes, hashes of the released files, record counts, omissions, and reproducibility limitations. The manifest does not attempt to hash itself.

**Development-v1 schema:** its `tasks.jsonl` preserves original `{task, prompt_ids}` records, and its checkpoint is nested inside each `outcomes.jsonl` episode (or `null` when not present). It has no artificial shared `checkpoints.jsonl` table. Episode identity is `(problem_id, condition, repeat)`, where condition is `baseline` or `sham`. The exporter requires the completion summary and complete planned episode identity grid before publishing this mode. The original schema differences are intentional.

## Source provenance limitations

For `phase0-20260927`, the runner was reconstructed from the pre-review source by reversing the recorded review patch while the original process was running. This limitation is preserved from its original manifest. Original helper snapshots were not captured. `source/tasks_reference.py` is the exact verifier/task source captured by the later 1,024-token diagnostic, supplied as a clearly labeled verification reference; it is **not** represented as an attested original Phase 0 helper. A complete byte-for-byte reconstruction of the first execution environment is therefore not established.

For `phase0-budget1024-20260927`, the released historical source files match the hashes captured in that run's manifest. The archived runner retains its original optional hosted-client import, but that credential/network module is intentionally absent from this outcome release. Independent outcome inspection needs no hosted client or API key. Historical model generation still requires the recorded open-weight model and compatible runtime; floating-point/kernel and sampling differences can affect reruns.

For `development-v1-20260927`, launch snapshots were originally saved as **flat files under `source/`**, while manifest hashes use original repository-relative names such as `scripts/run_development.py` and `src/jev_control/mlx_backend.py`. `source_manifest.json` preserves both mappings and verifies the saved bytes. These snapshots attest only to this run's captured sources; they do not assert that the current repository helpers are identical. The unused credential/network adapter is excluded with its original declared hash retained. No helper from another run is substituted in development-v1 mode.

## What cannot be reproduced from this release

**All raw Jev responses, probabilities/features, request objects, and per-request usage objects are omitted.** Consequently, a Jev feature ablation, Jev calibration analysis, or Jev-versus-local feature comparison cannot be reproduced from this outcome-only release of hosted-service experiments. Any future release of raw hosted outputs requires separate clarification of the applicable service terms. This restriction does not transform local-model judgments into Jev judgments or remove the independently generated local-model records.

Development-v1 itself has no hosted output to withhold, but its baseline/sham design cannot estimate a Jev feature or policy effect. No credentials, local network configuration, absolute filesystem paths, or unrelated model-cache inventory are included. The project-level accounting report separately describes measured total API spending; these public records do not provide an invoice reconciliation.

## Interpretation and analysis

The historical checkpoint builder generated a block and then rewound to a completed paragraph. Some retained checkpoints are much earlier than the end of that block. All generated prefix tokens, including discarded tails, remained charged. These records identify a **constructed-state, single-intervention** experiment; they are not prospectively captured online-boundary states. In Phase 0, whole-initial-generation entropy/logprob averages can include discarded future text and must not be used as earlier-checkpoint predictors. Retained-token statistics were added in the later run.

The shared allowance matches emitted generator tokens, including discarded candidates. Prompt insertion, repeated prefill, local judging, and hosted acquisition are additional resource dimensions. Branch selection is the recorded mean-generator-logprob heuristic, not an independently validated correctness selector. Existing outputs and labels are preserved even when an implementation limitation is subsequently discovered.

To regenerate the descriptive outcome/cost analysis with the repository's analysis tool, without loading a model or contacting any service:

```sh
python scripts/analyze_screen.py artifacts/phase0-20260927 --output results/public_phase0_analysis.json
python scripts/analyze_screen.py artifacts/phase0-budget1024-20260927 --output results/public_budget1024_analysis.json
```

These local analysis outputs may contain provenance paths created on the reader's machine; they are not automatically approved public artifacts. Do not enable learned-controller evaluation on these tiny development datasets. A grouped cross-fit estimate, even on a later larger dataset, is exploratory unless its inferential conditions are justified.

For an independent verifier audit, load the recorded task data and output text and compare a fresh verifier result **separately** with the stored `outcome` field. Use `source/tasks.py` for the second run or the explicitly qualified `source/tasks_reference.py` for the first. Report disagreements without replacing the original labels. No such rescoring was performed by the exporter.

For development-v1, the strict analyzer supports this public release directly. Set `MODEL_SNAPSHOT` to a local directory containing `tokenizer.json` from the recorded `Qwen/Qwen3-4B-Instruct-2507` revision, `cdbee75f17c01a7cc42f958dc650907174af0554`, then run:

```sh
python scripts/analyze_development.py artifacts/development-v1-20260927 \
  --tokenizer "$MODEL_SNAPSHOT" \
  --output results/development_v1_public_reanalysis.json
```

The explicit tokenizer override is necessary because the public manifest contains a model identity, not a machine-specific cache path. This analysis loads only the tokenizer, validates saved token/text and successive-prefix records, and reruns the strict task verifier without changing stored labels. It does not load model weights, generate text, or call any service. Existing analysis output files are not overwritten; choose a new output filename when rerunning. The default plan is `configs/development_v1.json`.

The analyzer preserves the original `source_sha256` inventory while separately reporting which contents it can verify. It validates the public release and source manifests, requires every released source to match its original launch hash and run identity, and permits only the explicitly declared omission of the unused `src/jev_control/jev.py` adapter. Undeclared missing sources, substitutions, altered input files, and omissions of executed local helpers are rejected. The omitted adapter's bytes are not claimed as verified. These hashes establish consistency of the released inventory, not cryptographic authentication by an external signer. The current verifier must also match the captured verifier hash; a future verifier change requires explicit review.

The completed public reanalysis reproduced all 17 non-provenance top-level fields of `results/development_analysis.json` exactly, including all 48 episode audits, problem-level statistics, costs, and the seeded bootstrap. It found zero outcome disagreements. The original launch-source, verifier, and audit-tokenizer hashes match; task, outcome, and summary objects are identical. Input file hashes can differ because export formatting is normalized, and the manifest is sanitized. Public-release validation metadata therefore differs intentionally from private-run provenance. The original audit and run records were not rewritten.

The source mapping also allows a reader to reconstruct the original repository layout where necessary; the flat snapshot directory alone is not advertised as a standalone runnable generation package.

## Recreating and checking the release

`scripts/export_public_runs.py` reads existing local run files and never imports an inference backend, calls an API, or executes a task verifier. It verifies original record counts, copies recorded labels, checks captured source hashes, and scans the proposed output before writing. A suspected secret, machine address, endpoint, or private path causes failure with **relative filenames only**; matched content is never printed. Existing destinations are not overwritten.

The exporter uses explicit field allowlists. `tests/test_export_public_runs.py` verifies preservation of recorded outcomes/costs/IDs, removal of hosted and machine-specific fields, source-hash checks, original-file immutability, refusal to overwrite, and failure before publishing unsafe content. The public files retain their exact historical distinctions rather than silently merging source versions or recomputing results.

Development-v1 is selected automatically from its exact manifest protocol, or explicitly with `--mode development-v1`. Unknown protocols, cross-protocol records, incomplete identity grids, hash mismatches, and borrowed helper substitutions are rejected. Its targeted tests cover unchanged task/output/call/checkpoint fields, flat-source hash resolution, labels and cost preservation, completion checks, and refusal to mix versions.
