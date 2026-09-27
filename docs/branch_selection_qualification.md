# Common-pool branch selection: development contract under construction

This is the next baseline work package, not an executed experiment. No data from the completed prospective v1 test may be used to choose its settings.

## Primary-source boundary

Rechecked September 27, 2026: [GUARD source at 8d2bc7a](https://github.com/ZHUWEI-hub/GUARD/blob/8d2bc7afcb3d070543604c4cf7830156b30aa486/eval/math_eval_guard.py), and [paper v1](https://arxiv.org/html/2604.14528v1). The implementation ranks branches by onset minus mean branch entropy and adds a completion-marker bonus. It also uses an entropy-history trigger, branch-specific prompts/temperatures, and its own budget loop. Our common-pool entropy selector preserves only the ranking component. It is not a faithful end-to-end reproduction and must never be labeled simply “GUARD” in result tables. A completion marker does not establish correctness.

## What the comparison must isolate

The v1 branch action combined producing two short candidates with choosing by likelihood. Its weak outcome could reflect insufficient candidate diversity, poor ranking, or spending budget that continuation needed. These are distinct hypotheses.

Use a fixed candidate pool at a natural development checkpoint, then record choices from uniform random, likelihood, entropy reduction, and a local semantic judge before generating any future continuation outcomes. All selectors see the same candidate texts; only cheap selectors use logged token statistics. The semantic selector's access difference and cost must be explicit. Jev may later be added through the same observable-text contract after the local comparator is qualified.

To diagnose selection, independently continue every candidate under the same remaining-budget contract with repeated seeds. This expensive all-candidate collection supplies measurements, not deployable free information. Never pick a candidate using those terminal outcomes and call it an operational selector. If a selected candidate already ended, preserve its actual terminal state rather than resume past EOS.

Separate three baselines: ordinary continue using the full allowance; candidate generation plus uniform selection; and candidate generation plus each informed selector. Charge the full pool to every hypothetical branch policy, including rejected candidates. Charge judging separately and reserve or subtract it consistently for any matched-total-resource comparison. Report actual shared collection costs separately. Missing scores or malformed local outputs need a frozen fallback, recorded as failure, not a retry-until-valid procedure.

The implemented `branch_selectors.py` supplies ranking and strict local-choice parsing. `branch_experiment.py` and `scripts/run_branch_qualification.py` now implement the common-pool collection, pre-outcome durable decisions, separate local-judge accounting, early-ended candidate handling and source snapshots. Twelve targeted tests pass. `configs/branch_qualification_v2.json` specifies eight fresh development problems, three 128-token candidates, two continuations per candidate, a 2,048 generator-token allowance and a separate 128-token local selector cap. An independent complete-run audit and integration review remain required before interpreting results. This runner has not been launched; the repair experiment is unchanged.

## Requirement for the later published-method comparator

A ranking-component adaptation cannot satisfy the whole “strong compatible published controller” requirement. A separate integration must preserve a method's actual trigger and intervention semantics, or justify every adaptation, and establish that its input/runtime assumptions are available. RSD needs distinct draft/target roles; math-specific PRMs need a compatible task domain. Use this branch diagnostic to qualify machinery, not as a substitute for the published-method comparison.

## Launch and audit status

Launched September 27, 2026 after repair v2 completed and its audit passed. Runtime records are in `runs/branch-qualification-v2-20260927` and its separate control directory. The independent auditor is `scripts/audit_branch_qualification.py`; analysis is `scripts/analyze_branch_qualification.py`. They check/report full-pool accounting, exact pre-outcome choices, terminal candidates, local failure fallback and original-problem weighting. These tools must pass on the completed actual run before any result is claimed. Local acquisition is charged only to the local-semantic policy, despite sharing collection records. No Jev calls are made.
