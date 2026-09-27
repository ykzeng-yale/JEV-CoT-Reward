# Repair-action qualification v2 — development only

Frozen before collection, September 27, 2026. This advances curriculum stage 2; it is not a new confirmatory Jev trial or a test of all possible controls.

Eight fresh procedural problems (four graph, four arithmetic; seed 691027, indices 0–7) receive a natural online checkpoint after 256 tokens at the next qualifying newline, capped at 384. There is no lookahead or correctness-based checkpoint selection. Each eligible checkpoint has five actions and two prespecified repeats; all sampled ineligible problems remain recorded without replacement. This yields at most 80 continuations. No Jev calls or model downloads.

The frozen local Qwen3-4B-Instruct-2507 model uses the same 4-bit MLX configuration, temperature .7 and top-p .9 as v1. Every hypothetical episode has 2,048 generated tokens including its shared initial prefix and a 128-token final-answer reserve. Inserted instructions are counted as processed input, not generated output; discarded original segments remain charged. The larger budget is a development setting, **not** a causal comparison to historical 1,024-token runs. Task difficulty, total prefill work, actual duration and truncation must be reported.

| Action | Intervention |
|---|---|
| Continue | Exact-token resume with no added instruction |
| Sham | Retain every token and append a neutral instruction to continue and answer |
| Suffix repair | Original v1 rewind of at most 128 tokens, then its fixed recheck instruction |
| Recheck | Retain every token and append the same recheck instruction as suffix repair |
| Segment repair | Remove the last nonempty paragraph from its exact token boundary, then append the frozen reconstruction instruction |

Segment repair may remove the entire emitted prefix when it has no earlier paragraph boundary; record the count rather than silently substituting a different treatment. The semantic checkpoint newline can occur inside a paragraph. This action is not claimed to be a targeted error-localizing critic. The sham and repair prompts have different token lengths and content; their contrast does not isolate a single linguistic cause. Recheck versus suffix repair shares the instruction, helping isolate retained-history differences at these fixed treatment definitions.

All actions use the same prespecified continuation seed within problem/repeat. This does not prove common-random-number variance reduction. Execution order is randomized before generation. Terminal gold is available only to the post-completion verifier, not to action construction. Save exact prepared token IDs, deletions, inserted instructions, call starts/completions, time, source snapshots and model hashes. An external supervisor bounds wall time and sampled RSS; sampled RSS is not a hard Metal-memory cap.

## Analysis contract and transition

Independently reconstruct treatments, token-call chains, all accounted work, enrollment and terminal labels before interpretation. Report every action and family, exact sample sizes, failures and missingness. Use original problems for population uncertainty; eight problems and two repeats cannot establish stable heterogeneity or policy efficacy. Do not select a best action from the same samples and report its apparent maximum as achieved policy value.

First assess intervention feasibility and whether errors are truncation, invalid arithmetic/formats, ineffective self-correction or lost necessary context. A replicated directional difference would justify a separately frozen replication with more independent problems/repeats; an ambiguous result can justify precision work or a specified redesign. Neither outcome closes the whole research program. Do not train a new controller or scale to Jev until action quality and useful variation are better understood. Branch-selector qualification and a compatible published-controller comparison remain separate required work.

Configuration: `configs/action_qualification_v2.json`. Runner: `scripts/run_action_qualification.py`; intervention code: `src/jev_control/repair_actions.py`. The old v1 experiment code/configuration is unchanged.
