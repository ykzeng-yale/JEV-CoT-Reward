# Terminal-Bench v4.0.0 task-manifest audit

**Audit date:** September 29, 2026
**Source:** official `harbor-framework/terminal-bench` release `v4.0.0`, exact Git commit `452bf305c6daa62fc59061d22133a7cbc7c1572e`; matching Harbor dataset tag `terminal-bench/terminal-bench@4.0.0`. The repository was fetched with a filtered, no-checkout Git clone; only the task tree and selected task metadata were read. No task was run, no container/model was started, and no benchmark prompt or task artifact was copied into this repository or a model-training corpus.

## What this audit establishes

The release tree contains **66 `task.toml` manifests**. It provides task-specific verifiers, but tasks are individually authored and heterogeneous; the count is not evidence of 66 exchangeable instances of one treatment population. The official Harbor dataset page also warns that the suite is evolving, includes GPU and multi-container tasks, and recommends hosted environments such as Modal or Daytona. Therefore “Terminal-Bench has tests” does not mean every task is directly runnable on Bouchet or that its verifier is automatically tamper-proof.

The strongest structural match among the metadata reviewed is `terminal-bench/freight-dispatch-shift`:

- Its metadata describes a stateful, partial-information dispatch problem with multiple chronological event-feed/plan cutoffs and irreversible commitment points.
- Its verifier runs separately and is described as checking state persistence, temporal visibility, feasibility and terminal plan fields. Its environment metadata requests no GPU.
- This is a useful **single-task mechanism case study candidate** for asking whether a monitor notices stale/incomplete state before committing. It does not itself define a Jev intervention, expose a population of independent tasks, establish a contribution distinct from PCCC/CVT-RL, or solve the need for an executable local runner. Replaying checkpoints within this one task would be clustered within task and cannot be counted as independent samples.

Another 60-hour task, `takens-embedding-lean`, illustrates the same limitation in a different direction: a deep formal problem may provide long reasoning traces, but it is one problem rather than thousands of independent problem units. The release includes other multi-step science/engineering and operations tasks, but pooling them as iid without a defensible common estimand would be invalid.

## Power implication

For an intentionally optimistic calculation, if all 66 distinct tasks were treated as independent, exchangeable paired binary outcomes with 30% discordance, a single two-sided .05 comparison would have an approximate 80%-power minimum detectable risk difference of **18.6 percentage points**. The tasks are in fact heterogeneous and the candidate above is only one task, so this calculation overstates what the release can establish for a task-specific controller. Repeated checkpoints, seeds, or replays on a task do not turn it into new independent tasks. The release may support descriptive cross-task evaluation or a case study; it is not an adequate 3-pp confirmatory dataset.

## E1 disposition

**Do not use Terminal-Bench v4.0.0 as the main powered Jev-vs-local efficacy study.** Retain `freight-dispatch-shift` only as a candidate for qualitative mechanism/evaluator feasibility, if its full task contract, license, sandbox, test-integrity and prior-art audit later pass. A confirmatory direction needs a public family with many independent generated instances or a benchmark designed for a clear task-level population, a natural checkpoint/action rule, and published-method distinction. A synthetic task generator is not automatically sufficient: validate its distribution and executable outcome oracle independently before sampling study instances. No inference or API call was made here.

## Provenance and boundaries

- Release page: https://github.com/harbor-framework/terminal-bench/releases/tag/v4.0.0
- Harbor dataset documentation: https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4
- Task metadata was read from the pinned Git tree only for this feasibility review; benchmark content is not included in this repository.
- This is a source/data-structure audit, not an efficacy experiment, a complete security review of task verifiers, or a claim that the benchmark is invalid.
