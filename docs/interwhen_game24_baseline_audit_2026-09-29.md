# Interwhen Game24 baseline: source and execution audit

Audit date: September 29, 2026. Source inspected at Microsoft/interwhen commit `2bc515f0c13a7a81b3f72a93958280c60231573c` (MIT), especially [`thinkingPhaseVerifierGame24.py`](https://github.com/microsoft/interwhen/blob/2bc515f0c13a7a81b3f72a93958280c60231573c/interwhen/monitors/thinkingPhaseVerifierGame24.py) and [`game24_example_gt.py`](https://github.com/microsoft/interwhen/blob/2bc515f0c13a7a81b3f72a93958280c60231573c/examples/TTSwithVerification/interwhen/game24_example_gt.py). This is static source inspection plus a local adversarial arithmetic fixture; the authors' benchmark was not run.

## Direct method overlap

The TTS-with-verification Game24 monitor is a direct precedent for real-time judging and steering of a generated reasoning trace:

1. After a configurable warmup, it triggers after every `newline_threshold` reasoning newlines (defaults in the monitor: 15; TTS runner: warmup 4, threshold 20).
2. It appends a side-stream prompt asking the model to extract a partial expression from the existing prefix, with up to 20 generated tokens.
3. A deterministic Game24 checker labels the extracted expression as invalid/dead-end, valid partial, or valid complete. An invalid expression causes feedback and a retry; a valid partial receives no intervention; a valid complete expression causes the monitor to state it has verified the expression and close the thinking phase early. It caps corrective feedback.
4. After natural think termination it injects a final-expression prompt and applies the same checker.

Therefore “a real-time judge checks visible CoT, repairs errors, or decides when to stop” is already implemented. The side stream is itself extra generation from the main model, not a passive read-only judge. It can synthesize a valid expression not yet explicitly present in the trace. A fair comparison must separately charge its input/prefill, generated tokens and latency, and include a same-cost extraction/no-feedback control. The method also changes the final-answer elicitation protocol.

Do not confuse this code path with `examples/EarlyStopping/game24_example.py`: that example's `--monitor` switch uses a k-stable-answer early-stop monitor. The trace-verification experiment is in `examples/TTSwithVerification/interwhen/game24_example_gt.py` (Qwen3-30B-A3B-Thinking-2507 default) and a sibling QwQ-32B example. The `game24_example_gt.py` runner reports 1,362 train examples by default, samples indices across `nlile/24-game` train, and defaults to one attempt. Its monitor threshold, warmup, correction cap, seed/repeats and best-of-k behavior must be frozen explicitly in any comparison.

## Outcome-independence defect in the inspected implementation

The runner calls the same imported `verify_expression` for the monitor and final answer score. Thus the final score is a second application of the same implementation, not an independent verifier. Additionally, that function parses numeric leaves by regex and evaluates arbitrary Python expression syntax with `eval(..., {"__builtins__": None}, {})`; it does not enforce the task's `+,-,*,/` operator alphabet. A local fixture `2**3*3*1` with numbers `[1,2,3,3]` passes the regex multiset and evaluates to 24, despite exponentiation being forbidden. The relevant static source and fixture were checked; the upstream package itself was not imported because its Python dependencies are not installed locally.

This does not establish that the paper's reported results are false. It establishes that a faithful reproduction must add an independent, stricter terminal evaluator and separately report agreement with the upstream monitor's internal checker. A repaired implementation is an adaptation and must preserve a pinned untouched upstream condition if claiming a direct reproduction.

## Independent exact validator added here

`src/jev_control/game24_exact.py` parses an expression AST, accepts exactly four integer leaves using each input number once, permits only binary `+,-,*,/`, and evaluates with `fractions.Fraction`. It never calls Python `eval` and shares no code with interwhen. `tests/test_game24_exact.py` covers a valid rational solution, illegal exponent/floor division, invalid syntax/calls/bools, division by zero, number substitution/multiplicity, and task-size/type errors. The focused validator and existing selector suites pass 21 tests.

The validator only accepts a Python-style expression string; a future adaptation needs a separately audited answer extractor for boxed/LaTeX text. Before any model job, freeze a common task snapshot, exact model/tokenizer/runtime, `continue`, published upstream verifier, sham side-stream, and any Jev/local semantic selector; charge all generation and tool costs, retain every natural trigger/ineligible case, use fresh task-level holdout and independent exact terminal scoring, and decide whether the study is a reproduction or adaptation. The present Bouchet environment has no `vllm` module (PyTorch 2.9.1/CUDA 12.8 modules and Apptainer are available); current repo footprint is 7.7 GB. No vLLM install, data/model download, inference, or Slurm job was started. The lead Mac is occupied by an unrelated model task and has high swap use; mini/aux have existing Ollama/Docker activity and limited disk. Runtime qualification should be an explicit later bounded step after the study question survives the novelty and opportunity gates.

## Research consequence

This direct baseline substantially weakens the residual novelty claim: our prior proposed prefix checking/repair/early stopping is not distinct from interwhen's Game24 monitor. The only Jev-specific possibility left is an empirical ablation on ambiguous semantic states where no executable domain rule already supplies the answer, and even that remains covered generically by learned judge/controller prior art. Do not spend Jev budget or start model inference unless a concrete held-out task demonstrates this setting and written TypeSafe clarification permits the experiment.
