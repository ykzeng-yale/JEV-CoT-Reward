# InterWhen K-stable Game24 baseline adaptation v1

## Question and scope

Can the published InterWhen K-stable Game24 stopping rule be executed on our
qualified Bouchet CUDA stack and reduce generated tokens while preserving exact
Game24 outcome on a fixed public development sample? This is a baseline/runtime
qualification, not a Jev experiment, confirmatory test, InterWhen reproduction,
or novelty/efficacy claim.

## Frozen task and source

The task list has 96 evenly spaced rows from `nlile/24-game` train at revision
`0106b176e4648061d60c32ddb85ac0816699652d`, sampled with the index rule in the
pinned upstream example. Task IDs and numbers are frozen in
`data/interwhen_kstable_game24_v1/tasks.json` (SHA-256
`f20fea0807ad1b56acfaee276bd752df95fff9a760156c6f86bd0ebb0632d3a0`). The
task list contains no solutions or difficulty/outcome metadata. An independent
exact-arithmetic search over the four numbers confirmed that all 96 sampled
tasks are solvable; no witness expressions were retained or used in prompts.
Upstream source
is Microsoft/interwhen commit
`2bc515f0c13a7a81b3f72a93958280c60231573c`, MIT; exact comparator is
`KstableAnswerGame24Monitor`, with `k=2`, as set in its Game24 example.

## Arms and procedure

Run every frozen problem in both arms, with one fixed attempt and the same
initial seed `42 + dataset_row_idx`; a triggered post-stop finalization call uses
`initial_seed + 1000003` so it cannot silently reuse the start of the first
sampling stream:

* `continue`: sample a single uninterrupted answer, capped at 1,536 generated
  tokens.
* `interwhen_kstable_k2`: apply the published K-stable repeated-equation
  stopping condition to the visible generated trace. On a trigger, close the
  reasoning phase and let the same model produce its final answer using only
  the remaining token allowance. The adapter uses safe arithmetic parsing for
  detector candidates; this and the smaller receiver checkpoint are declared
  departures from the upstream runtime.

Use the same Qwen3-4B-Instruct-2507 BF16 checkpoint (revision
`cdbee75f17c01a7cc42f958dc650907174af0554`), temperature 0.6 and top-p 0.95.
Per-call timeout is 900 seconds; the full experiment's cooperative deadline is
5.5 hours inside a six-hour Slurm allocation. Run offline from the existing
Bouchet project cache. No Jev or other hosted API
is called. Continue is the primary comparison; early stopping is intended to
change compute, so generated-token and wall-time savings are measured rather
than padded away.

## Outcomes, costs, and analysis

The terminal answer extractor reads only the final boxed expression after a
reasoning-close marker (or the final boxed expression when no reasoning marker
is present). The independent AST/Fraction verifier rejects Python evaluation,
unsupported operators, altered/missing/multiply-used inputs, and inexact
answers. Preserve every task, raw trace, trigger/equation, generated and
injected-control token count, call record, timing, failure, and exact outcome.
Report paired task-level accuracy difference, Wilson intervals per arm,
paired discordance and a task bootstrap interval, trigger frequency, generated
tokens, prompt/prefill tokens, latency, failures and actual GPU time. These
training-split data are used only for baseline qualification; do not claim
independent test performance.

## Resource envelope and decision

One standard-tier GPU (`gpu_h100,gpu_b200,gpu_h200`), four CPUs, 32 GiB host
memory, six-hour Slurm limit, and a 5.5-hour cooperative wall bound. The model
must be read from the pre-existing pinned cache; the job is offline and exits
if any required snapshot file is absent. No model/data download or Jev spend is
permitted. Scale-up is not automatic: an audit-passing run can establish
runtime/task compatibility and baseline behavior only. A later scientific
study still needs a surviving task/action gap, written TypeSafe terms
clarification before Jev calls, a prospective holdout, and outcome-powered
design.
