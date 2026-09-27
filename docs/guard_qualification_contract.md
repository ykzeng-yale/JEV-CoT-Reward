# GUARD comparator qualification

Reference reviewed September 27, 2026: [pinned official source](https://github.com/ZHUWEI-hub/GUARD/blob/8d2bc7afcb3d070543604c4cf7830156b30aa486/eval/math_eval_guard.py). Independent executable checks: `src/jev_control/guard_contract.py` and `tests/test_guard_contract.py`.

The source appends the current boundary entropy before computing an index-based 0.90 quantile, then compares strictly. With five through ten observations that quantile is the maximum, so no trigger is possible. Eleven observations permit a trigger. Remaining tokens must exceed 200. Branch temperatures are absolute 0, 0.6 and 1.5, despite the variable name suggesting adjustment. Scoring subtracts mean branch entropy from onset entropy and adds ten for a completion marker. Empty entropy gets negative infinity. Tests preserve these details; markers do not establish correctness.

Our implementation is still **not a complete GUARD comparator**. Required integration work: predictive entropy at the exact boundary position, tokenizer-specific boundary handling, branch instructions/temperature changes, repeated scheduling, completion behavior and all-work budget accounting. The current backend exposes emitted-token statistics; its post-emission callback alone cannot reproduce a pre-sampling forced-EOS transition. The upstream marker detector also acts on token pieces rather than a verified complete answer. Any corrected stopping rule must be labeled an adaptation. The shared-pool entropy component omits all these mechanisms.

## Decision for this curriculum

Keep the v3 run fixed. Do not retrofit a named GUARD row into its identical candidate pools: GUARD changes candidate generation as well as timing. Before prospective full-controller evaluation, implement a separately named GUARD policy adaptation with an all-work ledger, test its event ordering, and run a small fresh runtime qualification. Report both policy differences and hardware/backend differences. A faithful original-runtime replication would additionally require qualified vLLM/CUDA compute; the current Mac MLX backend does not supply that runtime.

This is a baseline integration requirement, not a blocker to the currently running direct-selector development experiment. No new hardware purchase is justified solely for this qualification.

## Executable adaptation, not yet measured

`src/jev_control/guard_adaptation.py` now implements a budgeted segment-boundary variant with CPU fixture tests. It measures entropy at each 64-token boundary, preserves the current-inclusive quantile and branch temperatures, and charges every generated candidate. It omits the unverified completion bonus and uses the task's final-answer convention. These changes are explicit method differences, so result tables must say “segment-boundary GUARD adaptation.” The backend adapter and independent real-run audit still require qualification after the current inference worker finishes. No performance result exists for this implementation. Injected branch/final instructions count as processed prompt tokens, not generated output; final reserve and discarded work remain in the generation cap.
