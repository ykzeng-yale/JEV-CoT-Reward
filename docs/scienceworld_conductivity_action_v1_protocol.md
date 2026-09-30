# ScienceWorld conductivity: frozen development action study v1

This protocol specifies a prospective, non-model development census. It records no result or claim that the circuit has already worked in the simulator. The machine-readable authority is `configs/scienceworld_conductivity_action_v1.json`; execution is `scripts/run_scienceworld_conductivity_action_study.py`, pure controller primitives are in `src/jev_control/scienceworld_conductivity_policy.py`, and the independently authored endpoint and replay checks are in `src/jev_control/scienceworld_conductivity_action_audit.py` and `scripts/audit_scienceworld_conductivity_action_study.py`. Freeze their bytes and checksums together before submission. Changes after an attempted run require a separately identified input bundle and a documented reason.

## Scientific question and scope

Does a fixed controller that constructs a simple circuit and reads its visible indicator achieve higher terminal task success than the same circuit policy whose final classification decoder receives no measurement signal? This tests use of an observed measurement by a specified deterministic decoder. It also supplies a strong cheap measurement baseline for any later controller study.

The primary contrast is **measurement minus masked measurement**, paired within each official development variation. It is not Jev efficacy, novel theory, a new task-family result, or proof that measurement is necessary for every possible controller. Public simulator source and seed construction may permit other ways of reconstructing a target. A successful deterministic meter could make this task too easy to support a useful Jev contribution; a subsequent selective policy would have to improve a meaningful cost–success tradeoff against this baseline.

## Corrected visible-input contract

The policy receives exactly three visible fields: `task_description = env.taskdescription()`, the text returned by `reset`/`step`, and a sorted, deduplicated sequence of strings from `get_valid_action_object_combinations()`. The input hash covers all three fields. The pure policy never receives an environment handle, the mixed `info` dictionary, variation index, private object tree, source conductivity label, score, reward, goal progress, or completion flag. The stock Python step adapter computes some of these internally; the claim is that they are not forwarded to or consumed by the controller.

The description supplies the target substance, workshop location, and conductive/nonconductive answer-container mapping. A source label is never an input to the measurement decoder. Train-only majority calibration is a separate evaluator operation; its single Boolean output is frozen before development policies run. For the input census, evaluator-private train/dev labels are used only to compare equal-input label consistency, without changing the policy or reporting individual label values.

This corrects the scope of the prior conductivity audit. That audit hashed **initial look text alone**, omitting the separately supplied task description and legal actions. Its exact-look repetition and conflicting-label counts are valid for that restricted input. They do not demonstrate identical full controller inputs or invalidate generalization from the intended full-input train/dev split. The public combination order uses 25 variations per substance: official train contains B,C,D,E,F,G,H,J,K,L,M,N and dev contains O,P,Q,R,S,T. Their full task descriptions are therefore disjoint by construction. The earlier look-only filter retaining 30 dev rows is not applied here. The new full-input census checks the contract directly without treating hash uniqueness as statistical independence or absence of every semantic shortcut.

## Panel, calibration and execution order

Use the pinned ScienceWorld 1.3.0 source revision `e8216d6044e8e39be9fcb185e3b2dfb602584b52` and archive SHA-256 `b5977fdf71fecaa3cc3985caa9a2f295569db96031ed453041f0e3eef3a6d41b`. The configuration pins the JAR, Python adapter, task, task maker, unknown-substance constructor, generic connection action and electrical-component source members. No gold path is generated, requested or copied.

First enumerate all 300 official training variations (IDs 0–299) and all 150 official development variations (300–449) for the full-input census. Compute the training-majority conductivity prior from training labels only; an exact tie predicts nonconductive (`False`). Retain exact hashes, target groups and label-equality flags, not raw label values.

Then run one fixed integration gate on **training IDs 0, 125 and 250**, using the measurement policy. This checks the adapter, visible-action execution, circuit construction and readable indicator contract before any development action outcomes are executed. A scientifically incorrect prediction is not a failed integration gate and cannot justify tuning or retry. A policy/adapter execution failure stops before the development panel, preserves diagnostics, and requires a specific reviewed correction. There are no serial outcome-selected pilot panels.

After that gate passes, run all **150 dev variations × four policies = 600 development episodes**. Each policy reloads the same pinned variation in a fresh episode and repeats the frozen visible-navigation/focus prefix. Policy order is measurement, masked measurement, random measurement, continuation prior. Independent full-trace replay must verify initial and post-action hashes; policy order is fixed and wall-time comparisons consequently remain descriptive. IDs 450–599 are never loaded by this study.

This is the complete available official dev panel, not an arbitrary small pilot or a power claim. It contains only **six target-substance groups**, with 25 variations per group. Report the fixed-panel results and each target group; do not present 150 independent population units or claim task-family transfer. Any six-group bootstrap is a descriptive sensitivity analysis with a very small number of groups, not a reliable confirmatory confidence interval.

## Measurement controller and controls

The controller parses the task's visible instructions, navigates using only visible legal doors/locations, and focuses on the specified substance. Navigation is capped at 12 actions. It chooses the lexicographically first compatible battery, indicator and three unused wires available through legal terminal referents. The electrical plan is independently authored from ordinary series-circuit semantics:

1. Battery anode connects through wire 1 to the indicator cathode.
2. Indicator anode connects through wire 2 to the unknown substance.
3. The unknown substance connects through wire 3 to the battery cathode.

There are six connection actions. Generic connection actions choose the unknown object's free terminals; the controller does not inspect hidden terminals or material properties. The source's polarity convention places the indicator's anode on the ground-return branch. The policy waits for five settling actions, then pads as specified below and inspects the indicator using an **actual legal `look at <referent>` action**. The parser also accepts `examine`, but the pinned legal-action generator emits `look at`; a synonym is not assumed to appear in the legal-action list. The description must contain one unambiguous named-device state, “which is on” or “which is off.” Absent or ambiguous readings are recorded failures, never silently mapped to nonconductive.

The four policies share the parsed task contract and frozen navigation/focus procedure:

| Policy | Physical actions after the shared prefix | Final classification |
|---|---|---|
| `measurement` | Six connections, five settling waits, padding and indicator inspection | Parsed visible on/off reading |
| `masked_measurement` | Identical circuit work and inspection | Frozen training majority; measurement text does not enter this decoder |
| `random_measurement` | Identical circuit work and inspection | Frozen SHA-256 bit of initial full-view hash and seed `20260930`; no variation ID |
| `continue_prior` | Wait padding; no circuit or inspection | Frozen training majority |

For each nonfailed episode, reserve the final action for answer placement, giving exactly **32 post-reset action calls**. The measurement arms inspect at action 31 and place at action 32; continuation waits until final placement. The reset itself invokes the simulator's initial look and is separately part of execution cost. All three measurement arms should have identical actions and hashes until final placement; their final answer may differ. The randomized arm randomizes classification, not measurement frequency: all three measurement arms attempt measurement on every eligible episode.

Equal action-call count does **not** imply equal intrinsic workload, simulator runtime, or efficient deployed cost. Record `intrinsic_action_count` before artificial padding: navigation/focus plus final placement for continuation; navigation/focus plus connections, five settling waits, inspection and placement for measurement arms. Record actual wall time and all attempted/completed actions, including failures. In particular, do not present padded continuation as having the same natural cost as a meter.

## Outcomes and independent auditing

Primary success requires independently verified placement of the specific target into the correct answer-container ancestry, completion of the ordered focus/placement goals, and absence of task failure. The evaluator runs only after the policy has irrevocably stopped. It checks the source-defined target, class-to-box mapping and hidden conductivity label against the runtime object tree and ordered goals. Optional circuit-building progress, rounded score, requested action text, or the API `done` flag cannot substitute for this endpoint.

Keep classification correctness, placement correctness, chosen-box/actual-placement consistency, ordered-goal completion, task failure, target uniqueness, and terminal success distinct. Missing or ambiguous targets and policy failures remain in the 150-row denominator. Infrastructure failures abort the run with preserved traceback and action context; they are not valid negative policy results. Predictions combined with correctness reveal the underlying label even if the label is not a separate output field; this is a controller-information separation, not a label-confidentiality guarantee. Retain episode-level traces/outcomes in the ignored immutable run directory and publish only aggregate summaries.

After terminal completion, verify scheduler exit/resources and every frozen input/output checksum. Independently replay all recorded actions and compare the initial/full and stepwise visible-input hashes, legal-action membership, measurement-reading hash, and final endpoint flags. Check panel completeness, train/dev-only scope, calibration, matching measurement prefixes, action bounds, schema, costs and failure counts before inspecting aggregate success rates.

Report paired wins, losses and ties, absolute success counts, the primary paired difference, both prespecified secondary contrasts (measurement versus random measurement and continuation), per-target results, eligible/completed circuits, absent/ambiguous readings, action failures, all costs, and the correction from look-only to full-input grouping. Do not suppress a failed arm or select a favorable subgroup. No design changes may be based on partial dev outcomes. Any later outcome-informed adaptation uses separately frozen fresh data.

## Resource envelope and decision

Plan one normal-tier `pi_fl426` CPU job with **2 CPUs, 8 GiB RAM, 120 minutes, no GPU**, 12 MB primary-result cap and 120 MB total-bundle cap. It makes zero model, Jev or external network calls. The prior 450-row reset/source-validity census took 196 seconds on a 4-CPU allocation; this establishes only a measured lower-bound calibration for initialization work, not a measured action-study throughput or a linear runtime guarantee. This study can execute 19,200 development actions plus up to 96 training-gate actions, resets and a separate independent replay. The 120-minute ceiling is a bounded conservative allowance; actual completion time, peak memory and CPU accounting must be reported. It does not justify an unbounded retry or another allocation if results disappoint.

A passing run establishes a reproducible fixed-panel measurement baseline and estimates the specified decoder's information-use contrast. The frozen development-signal rule requires all 150 × four policy episodes and independent replay, at least 95% measurement execution, a positive overall measurement-minus-masked difference, and positive differences in at least five of the six substance groups. This is an explicit development decision rule, not a confirmatory p-value or Jev scale-up criterion. Retain all scientific policy failures in each 150-row denominator; any infrastructure failure invalidates the panel. A null or negative run diagnoses this policy/task pairing and does not authorize an outcome-selected resubmission. Neither outcome alone establishes Jev value or completes the broader research objective. Advance a later controller comparison only if there is a scientifically meaningful cost–success question beyond always applying this cheap meter, with fresh evaluation units and a frozen fair comparison; do not scale the previously negative branch/repair actions on the strength of a meter result.
