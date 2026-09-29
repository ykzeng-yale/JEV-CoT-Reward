# Fresh scientific-exploration benchmark audit: ScienceWorld

**Date:** September 29, 2026
**Purpose:** identify a fresh interactive evaluation candidate after the full 120-episode DiscoveryWorld panel is used for development-baseline characterization. This is a primary-source/code audit only; no agent inference or benchmark outcome was run.

**Qualification update (September 29, 2026):** a separately frozen no-generation CPU preflight has now completed and passed independent audit as Slurm `27913546` (ScienceWorld 1.3.0; 30 tasks / 7,207 variations; 9/9 sampled held-out-variation, one-action replay digests match; 199 output files hash-verified; 0 GPUh). This verifies runtime/split/replay feasibility only. It does not change the task-family, efficacy, or discovery claim boundaries below. Exact artifacts: `results/scienceworld_transfer_preflight_27913546.json` and `results/scienceworld_transfer_preflight_27913546_audit.json`.

## Decision

ScienceWorld is the strongest currently audited *next interactive environment candidate* for a held-out-variation study of information-seeking actions. It fits long-horizon scientific experiment selection better than static code-generation benchmarks and provides an official variation-level split. It is **not** a confirmed OOD-by-task-family or scientific-innovation benchmark for this project. The eventual residual question remains whether a typed Jev judgment changes action selection and improves independently measured task outcomes over a strong action-specific cheap/local controller at full cost; no controller novelty or Jev benefit is established.

Do not start a ScienceWorld model run yet. Its no-generation environment/replay audit is complete, but first finish and audit ReCoMA job 27880628, then freeze a controller-visible checkpoint/action contract and compatible baselines from train/dev only; leave test variations untouched and resolve TypeSafe terms before any Jev call.

## Source and code evidence

- Official paper: [ScienceWorld: Is your Agent Smarter than a 5th Grader?](https://aclanthology.org/2022.emnlp-main.775/).
- Official repository: [allenai/ScienceWorld](https://github.com/allenai/ScienceWorld).
- Source audited locally at commit `e8216d6044e8e39be9fcb185e3b2dfb602584b52` (HEAD on September 29, 2026). Repository declares Apache-2.0 for its code; redistribute bundled media only after checking its separate attribution/license.
- README lists **30 task types and 7,207 variations**. Independently parsing the checked-in table reproduced the 30 and 7,207 totals. The environment is an interactive text world with state-changing actions and simulator feedback, unlike answer-only QA.
- The public API exposes train/dev/test variation accessors. The Scala `PythonInterface.getSets()` implementation partitions each task's variation indices into disjoint groups; repository tests assert pairwise disjointness. The paper describes the benchmark's randomized object/substance variations and train/development/test protocol. This is a held-out-*variation* split: the task templates remain shared across splits.
- Three high-fit tasks explicitly involve unknown properties: `measure-melting-point-unknown-substance` (300 total; 75 test variations by the current split implementation), `test-conductivity-of-unknown-substances` (600; 150 test), and `mendelian-genetics-unknown-plant` (480; 120 test). Together these yield **345 nominal test variations across only three task templates**. Variations are not interchangeable with independent task families; report task-template clustering and do not use 345 as 345 independent science domains.
- Current repository warns that version 1.3.0 changes deterministic object UUID enumeration relative to 1.2.x and can change physics trajectories, ambiguous action resolution, gold paths, and observation order. Any study must pin one version and re-run reset/replay and evaluator qualification under that exact version.

## Comparison with other audited benchmark families

| Benchmark | Public scale/interface | Fit for this project's next test | Boundary |
|---|---|---|---|
| DiscoveryWorld | 120 official episodes, 24 theme/difficulty strata × 5 supported seeds; interactive simulation and terminal scorecards | Best current baseline panel; a full-panel Qwen/ReCoMA run is already pending | Once used, the same 8 themes/24 strata are development-exposed; not a fresh confirmatory set or a powered 3-pp test |
| ScienceWorld | 30 task templates, 7,207 variations, interactive experiments/actions, simulator reward, disjoint variation splits | Best current next benchmark for held-out variation and long-horizon measurement/action selection | Only 30 templates; held-out variations do not mean unseen task families. Simulator task reward is not a measure of scientific novelty or discovery quality |
| DiscoveryBench | 264 literature-derived tasks plus 903 synthetic tasks; goal + dataset, analysis and hypothesis/workflow evaluation ([paper](https://arxiv.org/abs/2407.01725), [official code](https://github.com/allenai/discoverybench)) | Useful later for data-driven hypothesis/workflow selection across many tasks | Not a native interactive simulator; its tasks/evaluator do not directly isolate an intervention at a shared live state |
| ScienceAgentBench | 102 tasks from 44 publications in four disciplines, self-contained code outputs and execution/cost evaluation ([paper](https://arxiv.org/abs/2410.05080), [official code](https://github.com/OSU-NLP-Group/ScienceAgentBench)) | Stronger domain-authenticity and executable-output comparator for code-assisted discovery | Mostly code-generation/self-debug workflows, not a common live experimental-action interface; rubric and execution endpoints require a precise target-aligned subset |
| AstaBench | 2,400+ scientific-research problems across multiple capabilities; standardized tools/cost accounting ([paper](https://arxiv.org/abs/2510.21652), [official code](https://github.com/allenai/asta-bench)) | Potential broad follow-up if an appropriate test task suite has a common sequential/action interface | Overall count mixes different task families; the count alone does not yield a powered matched-prefix intervention test. The benchmark remains prior art, not a novel contribution |

The separate [ScienceWorld code and paper](https://github.com/allenai/ScienceWorld) describe a large action space and held-out variation split; the exact paper reports 7,207 variations and distinct train/dev/test allocation. [ScienceAgentBench's primary paper](https://arxiv.org/abs/2410.05080) documents its 102 tasks, expert validation, executable program targets, and the distinction between code/execution scoring and rubric review. These are source-reported benchmark properties, not results from our experiments.

## Candidate future protocol (not frozen or authorized to launch yet)

1. Select one task/action family using only benchmark source and development split. A reasonable screen is an explicit unknown-property task in which the environment offers multiple legal measurement/experiment actions; define exact eligibility from the observation and available actions before evaluating model performance.
2. Define the action set at a shared, prospectively sampled checkpoint: continue with the agent's proposed action; a predeclared executable information-seeking alternative; a work-matched sham; and a rate-matched random selector. Do not treat a fluent or “novel-looking” thought as scientific progress.
3. Replay state from reset using exact logged actions under a pinned simulator build; establish deterministic equality of observation, score state, and remaining budget. If the environment is not exactly cloneable, define the randomized rollout target instead of claiming individual checkpoint effects.
4. Use independent simulator-generated terminal success/progress measures; keep the controller away from hidden evaluator/gold-path state. Separately report hypothesis quality only if an executable or blinded expert rubric is preregistered.
5. Compare a cheap action-specific local safety/value model and a compatible published controller. A direct LW2S reproduction is only valid if omission and fall-through semantics match; otherwise label it as an adaptation. A Jev arm is gated on written TypeSafe permission and must include API acquisition latency, cost, failures, and all downstream computation.
6. Train/develop on official train (and tune once on dev), then freeze all choices before touching test variations. The primary estimand may generalize across held-out variations within the chosen templates only. A family-OOD claim requires held-out task templates or a separate environment and enough independent families.
7. Size using task/template-clustered variance from development, not variation count alone. Include actual turns/tokens/prefill, simulator calls, discarded branches, wall time, failed episodes, GPU-hours, and capped Jev spend. Keep the already closed v1 set closed.

## What was not verified

The source audit did not execute the Java simulator: this Mac has no Java runtime and `py4j` is absent. Thus variation counts and split membership above are derived from official checked-in source/README and repository tests, not a local runtime replay. No model/API calls, scheduler jobs, benchmark labels, raw traces, or new outcome records were created. Required next infrastructure evidence is a bounded no-generation replay on a compatible runtime before any study protocol can be frozen.
