# Common-pool branch selection: development contract under construction

This is the next baseline work package, not an executed experiment. No data from the completed prospective v1 test may be used to choose its settings.

## Primary-source boundary

Rechecked September 27, 2026: [GUARD source at 8d2bc7a](https://github.com/ZHUWEI-hub/GUARD/blob/8d2bc7afcb3d070543604c4cf7830156b30aa486/eval/math_eval_guard.py), and [paper v1](https://arxiv.org/html/2604.14528v1). The implementation ranks branches by onset minus mean branch entropy and adds a completion-marker bonus. It also uses an entropy-history trigger, branch-specific prompts/temperatures, and its own budget loop. Our common-pool entropy selector preserves only the ranking component. It is not a faithful end-to-end reproduction and must never be labeled simply “GUARD” in result tables. A completion marker does not establish correctness.

## What the comparison must isolate

The v1 branch action combined producing two short candidates with choosing by likelihood. Its weak outcome could reflect insufficient candidate diversity, poor ranking, or spending budget that continuation needed. These are distinct hypotheses.

Use a fixed candidate pool at a natural development checkpoint, then record choices from uniform random, likelihood, entropy reduction, and a local semantic judge before generating any future continuation outcomes. All selectors see the same candidate texts; only cheap selectors use logged token statistics. The semantic selector's access difference and cost must be explicit. Jev may later be added through the same observable-text contract after the local comparator is qualified.

To diagnose selection, independently continue every candidate under the same remaining-budget contract with repeated seeds. This expensive all-candidate collection supplies measurements, not deployable free information. Never pick a candidate using those terminal outcomes and call it an operational selector. If a selected candidate already ended, preserve its actual terminal state rather than resume past EOS.

Separate three baselines: ordinary continue using the full allowance; candidate generation plus uniform selection; and candidate generation plus each informed selector. Charge the full pool to every hypothetical branch policy, including rejected candidates. Charge judging separately and reserve or subtract it consistently for any matched-total-resource comparison. Report actual shared collection costs separately. Missing scores or malformed local outputs need a frozen fallback, recorded as failure, not a retry-until-valid procedure.

The implemented `branch_selectors.py` currently provides ranking and strict local-choice parsing only. A runner, complete budget contract, candidate count/length grid, source pinning, integration tests and frozen schedule remain required before launch. Running repair qualification is not modified to add branches midstream.

## Requirement for the later published-method comparator

A ranking-component adaptation cannot satisfy the whole “strong compatible published controller” requirement. A separate integration must preserve a method's actual trigger and intervention semantics, or justify every adaptation, and establish that its input/runtime assumptions are available. RSD needs distinct draft/target roles; math-specific PRMs need a compatible task domain. Use this branch diagnostic to qualify machinery, not as a substitute for the published-method comparison.
