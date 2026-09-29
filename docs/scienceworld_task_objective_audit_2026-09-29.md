# ScienceWorld task-objective audit: exploration claims and primary endpoints

**Date:** September 29, 2026
**Source:** `allenai/ScienceWorld` commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52`
**Purpose:** determine whether three candidate unknown-property task templates support a valid study of exploration/control, and specify what the official evaluator does and does not establish. This is static source inspection; no hidden target values, gold paths, model outputs, or episode scores were read.

## Main finding

These tasks contain useful measurement and experimentation affordances, but the official main goal can be answered by choosing an answer object without first performing the experiments. The experimental interactions appear as optional score subgoals. Therefore **benchmark score gain alone cannot identify better scientific inference or useful exploration**: it mixes answer correctness with optional task-progress bonuses, movement, setup, and interaction. A controller could improve total score by harvesting easy optional goals, even if it learns no more about the unknown property. Conversely, an informative probe may not change the terminal answer unless followed by a correct answer action.

For this family, separate (a) independently scored final classification accuracy, (b) terminal success, (c) intermediate official task score, (d) action/instrument cost and latency, and (e) evidence that the chosen intervention changes the final classification. Do not call optional-goal completion “discovery,” “innovation,” or CoT quality.

## Task-specific mechanism

| Official task | Required main sequence in source | Optional experimental/progress goals | Measurement implication |
|---|---|---|---|
| `measure-melting-point-unknown-substance` | Focus the thermometer, focus the unknown substance, then focus the above/below answer box (`TaskUseInstrumentThermometer3.scala`, lines 171–183). The task text defines a threshold classification, but does not require taking a temperature reading. | Thermometer use before and after heating, heating device activation, heating the sample by at least 20 C, moving/picking up/containing the sample (`TaskUseInstrumentThermometer3.scala`, 185–225). | Instrument use is a plausible information-gathering action, but terminal correct classification is distinct from that action; measure both directly. |
| `test-conductivity-of-unknown-substances` | Focus the unknown sample, then put it in the correct conductive/nonconductive answer container (`TaskElectricalConductivity2.scala`, lines 154–170). | Move to sample, connect the sample's two terminals, connect it through wires to a power source and bulb (`TaskElectricalConductivity2.scala`, 172–183). | Circuit construction/testing is a plausible probe; answer placement remains the main classification endpoint. |
| `mendelian-genetics-unknown-plant` | Focus the correct dominant/recessive answer box (`TaskMendelianGenetics2.scala`, lines 205–213). | Seed collection and planting, soil/water/pot setup, growing up to six plants across multiple life stages, hive/pollinator and mature-seed goals (`TaskMendelianGenetics2.scala`, 215–258). | The long action chain greatly expands optional progress but does not make action count or total score a direct measure of discovering dominance. Use classification accuracy plus explicit cost and evidence measures. |

These task descriptions and scores must be treated as public interface fields only when the model/controller receives them. The Python API documents `score`/`reward`, `taskDesc`, `valid`, `look`, and `inv` in its returned `info` fields (`scienceworld.py`, lines 410–452). If these are visible in every arm, freeze the same visibility and prompts for all arms and disclose it. The aggregate score is not a hidden independent evaluator; its exposure can change policy behavior and optional-goal gaming.

## Identification consequences

1. **Primary endpoint:** final correct target class under an independently implemented answer-key audit, with official terminal success as a separate metric. Report action costs, invalid actions, token use and latency. Do not substitute aggregate score for accuracy.
2. **Intervention estimand:** at a predeclared common checkpoint, compare a proposed measurement action with continue-as-proposed and a work/time-matched noninformative action. Apply the same downstream policy and budgets. If replayed branch state equality is not established for the full prefix, use episode-level randomized assignment rather than claim paired counterfactuals.
3. **Optional score:** report score change as secondary and decompose it by subgoal; it is contaminated by movement, setup and optional progress goals. Show whether an arm gains task score without improving final classification.
4. **Fairness:** every arm sees the same task text, legal-action list, observation, score/reward exposure, tool schema, downstream model, turn/token budget and stop rule. Include a strong action-specific cheap/local baseline and rate-matched random action. A Jev arm requires the existing written TypeSafe terms clarification and honest API latency/cost accounting.
5. **Generalization:** the three chosen task templates have 345 nominal test variations in total, but only three templates. Keep splits intact and analyze clustering by template. This can support only a within-template held-out-variation claim; it cannot support unseen-task-family OOD, broad scientific discovery or innovation.

## Go/no-go consequence

The no-model preflight (Slurm `27913546`) proves that pinned simulator version 1.3.0 can reproduce nine selected single-action test-variation checkpoints. It does not qualify task outcome effects or long prefixes. ScienceWorld is therefore still a **candidate mechanism sandbox**, not a ready confirmatory benchmark. Before any agent evaluation, freeze the visible information contract and classification endpoint; verify exact-prefix replay for the chosen trajectories or switch to randomized episode assignment; use train/dev only for action and checkpoint selection; keep test variations untouched. Before any Jev call, satisfy the separate contractual gate. For claims of scientific innovation or task-family OOD, use an independent benchmark with many suitably independent task families and an outcome rubric that measures discovery rather than optional simulator score.

## Source links

- [Unknown melting-point task](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/simulator/src/main/scala/scienceworld/tasks/specifictasks/TaskUseInstrumentThermometer3.scala)
- [Unknown conductivity task](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/simulator/src/main/scala/scienceworld/tasks/specifictasks/TaskElectricalConductivity2.scala)
- [Unknown-plant Mendelian task](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/simulator/src/main/scala/scienceworld/tasks/specifictasks/TaskMendelianGenetics2.scala)
- [Python simulator interface](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/scienceworld/scienceworld.py)

All links are pinned to the audited source revision. This document does not authorize reading hidden answer values or running an agent model.
