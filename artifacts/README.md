# Public local-model diagnostic records

These artifacts release the recorded local-model tasks, reasoning text, token IDs, outcomes, and resource measurements from two completed development runs. They are **not** a held-out policy evaluation, and they do not demonstrate a Jev improvement. Labels and model outputs were copied from the original records without regeneration or rescoring.

| Run | Original independent problems | Recorded continuations | Generator-token allowance | Recorded successes: continue / repair / branch |
|---|---:|---:|---:|---|
| [phase0-20260927](phase0-20260927/) | 6 | 36 | 512 | 2/12, 2/12, 1/12 |
| [phase0-budget1024-20260927](phase0-budget1024-20260927/) | 4 reused development problems | 12 | 1,024 | 0/4, 2/4, 1/4 |

The second run reuses the first four problems and partly the same seeds. Do not treat it as an independent replication or pool the two runs as ten independent problems. Repeated continuations within a problem are correlated for task-population inference; the 48 total continuation records are not 48 independent problems.

## Included files

- `tasks.jsonl`: original task IDs, prompts, families, and problem data, allowing independent checking of answers.
- `checkpoints.jsonl`: saved prompt and retained token IDs, initial generation, original checkpoint hashes, observable state, and available local token statistics. Hosted judge objects are removed.
- `outcomes.jsonl`: every original episode record, including answer text, original labels/reasons, per-call token IDs and timing, losing-branch work, repair overhead, seeds, and costs. Where the original schema had no separate episode ID, identity is the unchanged tuple `(problem_id, action, repeat)`.
- `manifest.json` and `summary.json`: original experiment settings and recorded aggregates, excluding local paths and hosted service data. The model's repository and revision replace its machine-specific cache location.
- `local_judge.jsonl`, when available: independently prompted **local-model** responses and probabilities, generated without access to Jev outputs. These are separate diagnostic judgments, not truth labels.
- `review_audit.json`, when originally saved alongside the run: the recorded implementation audit.
- `source/` and `source_manifest.json`: historical local experiment/verification source and explicit provenance. The credential/network adapter is omitted; its original hash is retained.
- `release_manifest.json`: original input hashes, hashes of the released files, record counts, omissions, and reproducibility limitations. The manifest does not attempt to hash itself.

## Source provenance limitations

For `phase0-20260927`, the runner was reconstructed from the pre-review source by reversing the recorded review patch while the original process was running. This limitation is preserved from its original manifest. Original helper snapshots were not captured. `source/tasks_reference.py` is the exact verifier/task source captured by the later 1,024-token diagnostic, supplied as a clearly labeled verification reference; it is **not** represented as an attested original Phase 0 helper. A complete byte-for-byte reconstruction of the first execution environment is therefore not established.

For `phase0-budget1024-20260927`, the released historical source files match the hashes captured in that run's manifest. The archived runner retains its original optional hosted-client import, but that credential/network module is intentionally absent from this outcome release. Independent outcome inspection needs no hosted client or API key. Historical model generation still requires the recorded open-weight model and compatible runtime; floating-point/kernel and sampling differences can affect reruns.

## What cannot be reproduced from this release

**All raw Jev responses, probabilities/features, request objects, and per-request usage objects are omitted.** Consequently, a Jev feature ablation, Jev calibration analysis, or Jev-versus-local feature comparison cannot be reproduced from this outcome-only release of hosted-service experiments. Any future release of raw hosted outputs requires separate clarification of the applicable service terms. This restriction does not transform local-model judgments into Jev judgments or remove the independently generated local-model records.

No credentials, local network configuration, absolute filesystem paths, or unrelated model-cache inventory are included. The project-level accounting report separately describes measured total API spending; these public records do not provide an invoice reconciliation.

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

## Recreating and checking the release

`scripts/export_public_runs.py` reads existing local run files and never imports an inference backend, calls an API, or executes a task verifier. It verifies original record counts, copies recorded labels, checks captured source hashes, and scans the proposed output before writing. A suspected secret, machine address, endpoint, or private path causes failure with **relative filenames only**; matched content is never printed. Existing destinations are not overwritten.

The exporter uses explicit field allowlists. `tests/test_export_public_runs.py` verifies preservation of recorded outcomes/costs/IDs, removal of hosted and machine-specific fields, source-hash checks, original-file immutability, refusal to overwrite, and failure before publishing unsafe content. The public files retain their exact historical distinctions rather than silently merging source versions or recomputing results.
