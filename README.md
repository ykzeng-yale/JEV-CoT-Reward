# JEV-CoT-Reward

Local-first research on whether a fast typed judge supplies useful information for **choosing reasoning interventions**. The generator is frozen; terminal answers are checked independently. Jev is a feature source, never the gold verifier.

**Completed diagnostic:** on 24 fresh prospective problems, always-continue, cheap-feature and Jev-feature controllers each solved 11/24; the local-judge controller solved 12/24. No Jev improvement was established. [Results, costs and no-scale decision](docs/prospective_v1_gate.md).

**Prior-art finding:** intervention advantage and same-prefix action experiments already have direct prior art. This project is a replication and proposed extension on semantic-feature value, transfer, and selective acquisition. A Jev-specific improvement has not been established.

**Active research:** the [v2 curriculum](docs/research_curriculum_v2.md) reopens the broader program, beginning with baseline adequacy and action qualification. The v1 no-scale decision applies to that configuration only.

Latest curriculum evidence: [repair qualification v2](docs/action_qualification_v2_results.md) completed 80 audited continuations. Full-segment repair solved 12/16 versus 11/16 for continue; uncertainty is wide and no improvement is established.

## Research package

- [Working manuscript](paper/main.tex): compiling development draft; incomplete experiments and theory integration are explicitly marked.
- [Branch-selection qualification contract](docs/branch_selection_qualification.md): separates candidate quality, ranking and total costs.

- [Prior-art and source-code audit](docs/literature_audit.md): 25 papers, pinned code references, important novelty corrections.
- [Staged study protocol](docs/study_protocol.md): estimands, treatments, grouping, comparators, costs, and decision gates.
- [Theory with proofs](docs/theory.md): identification, information value, regret, sequential assumptions, audits, transfer, and selective acquisition. Standard results specialized to this design, not new general theorems.
- [Independent theory review](docs/theory_review.md): eligibility targets, adaptive replication, cross-fit dependence, and acquisition accounting.
- [Baseline contracts and binary-outcome theory](docs/baseline_contracts.md): paper-inspired comparisons, tie-aware gate target and its surrogate-loss bound.
- [Observed hardware and resource plan](docs/resources.md).
- [Verified Mac compute pool](docs/mac_compute.md) and [active project scope](docs/project_goal.md).
- [First executed results](docs/results.md): separates model experiments, mathematical validation, and remaining work.
- [Source provenance](docs/source_manifest.json), [spending stages](configs/stages.json).
- [Public local-model diagnostics](artifacts/README.md): original outcomes, checkpoints, costs and source snapshots; raw hosted-judge responses excluded.

The earlier conversation's download links and claimed synthetic numbers were not present in this initially empty repository. All results here must point to actual saved runs.

## Local setup

Python 3.12; Apple Silicon for the supplied MLX backend. This uses cached open weights and does not download a model automatically.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test,mlx]'
.venv/bin/python -m pytest -q
```

`requirements.lock.txt` records the exact Apple Silicon environment used. On another OS, install the core dependencies and provide a compatible backend; `mlx-metal` is platform-specific.

```sh
.venv/bin/python scripts/run_screen.py \
  --model /absolute/path/to/local/Qwen3-4B-Instruct-2507/snapshot \
  --problems 6 --repeats 2 --budget 512 \
  --prefix-tokens 128 --final-reserve 96 \
  --output runs/my-phase0
```

Add `--jev` to collect the seven frozen rubric features at each unique eligible checkpoint. Jev responses do not choose actions during this all-arm data collection. Omitting the flag runs entirely locally. The first local judge comparator independently asks the same questions:

```sh
.venv/bin/python scripts/local_judge.py \
  --model /absolute/path/to/local/model --run runs/my-phase0
.venv/bin/python scripts/analyze_screen.py \
  runs/my-phase0 --output results/my-phase0-analysis.json
```

Read each script's `--help` for implemented limits. These small arithmetic/graph tasks test the instrumentation; they are not a substitute for audited research benchmarks.

The completed development diagnostic captures an online newline boundary after 256 tokens, capped at 384, without looking ahead and rolling back. It compares baseline generation (uninterrupted until the shared final-answer reserve) with exact-prefix pause/resume on 12 fresh problems and two seeds:

```sh
.venv/bin/python scripts/run_development.py \
  --model /absolute/path/to/local/Qwen3-4B-Instruct-2507/snapshot \
  --output runs/development-v1
```

[Frozen settings](configs/development_v1.json). This diagnostic makes no Jev calls. Stochastic resume uses a fresh recorded RNG seed; it does not claim bitwise replay equivalence. Earlier generate-then-rewind diagnostics remain separately labeled and are not pooled with online checkpoints.

Both development conditions solved 12/24 episodes (graph 10/12, arithmetic 2/12). All 24 paused prefixes matched their uninterrupted counterparts before pausing, and the independent audit found no outcome or accounting disagreement. This small sample does not establish equivalence. [Development gate](docs/development_gate.md): proceed only to a limited exploratory mechanism screen.

The next frozen schedule is 24 fresh problems × three actions × four continuations, with one online checkpoint and 1,024 generated tokens per hypothetical episode. The [mechanism configuration](configs/mechanism_v1.json) records the full contract; `scripts/run_mechanism.py --help` describes the runner. It completed 288 continuations; continue solved 47/96, repair 40/96 and branch 37/96. The [mechanism gate](docs/mechanism_v1_gate.md) did not support scaling. Jev queries use the central ledger with a $1 cumulative stage cap. The independent local judge and grouped analysis follow collection.

After completion, `scripts/audit_mechanism.py` reconstructs all recorded token/call chains and recomputes original outcomes without changing labels. Learned mechanism analysis requires that passed report via `--integrity-report`; changed input files invalidate it. The audit loads tokenizer files only. It does not replay model computation or establish that a controller helps.

For longer authorized runs, `scripts/run_bounded.py` records process identity and stops its own process group at a wall-clock, sampled-RSS, or log-size limit. RSS is not a hard limit on Metal/unified memory. Use a new control directory for every run; retain incomplete outputs and receipts.

## Budget and credential handling

The total user authorization is **$25 for Jev**. The adapter additionally starts with a **$0.25 cumulative Phase 0 cap**. It pins `jev-1.13.0`, batches independent questions, caches by state/model/rubric, reserves liability before dispatch in a process-safe SQLite ledger, reconciles provider usage, and never automatically retries. Failed/ambiguous attempts keep their conservative reserve.

Use `TYPESAFE_API_KEY`, or a user-private `~/.config/jev-cot-reward/api_key` file with permissions `0600`. Never place a credential in a command committed to Git, a notebook, or a result. This session's supplied credential was configured privately outside the repository.

All workers must share `~/.local/state/jev-cot-reward/jev.sqlite3`. Do not reset/delete that ledger between runs, and do not copy it to independent workers that spend concurrently. The cap covers requests through this client, not unrelated account usage. Price assumptions use the [official model page](https://docs.typesafe.ai/models), checked September 27, 2026; reverify before future campaigns. The reserve is $0.01 per attempt, above the documented maximum request charge at that price. This is client enforcement, not a provider-side billing cap.

## What is implemented

- Exact-token prefix reconstruction, seeded MLX generation, in-memory 4-bit quantization, log-probability/entropy collection.
- Continue, bounded suffix repair, and two-candidate branching with a fixed likelihood selector; all generated candidates count against the allowance.
- Independent shortest-path and safe exact-rational arithmetic verification.
- Persistent raw public checkpoint/outcome records, grouped analysis, rubric cache and spending ledger.
- Unit tests for cost guards, concurrency, validators, and intervention mechanics; real greedy pause/resume smoke checks.

The first generator is **Qwen3-4B-Instruct-2507**, a non-thinking instruction model. We intervene on explicitly generated explanations, not private reasoning. Matched generator-token budgets are not matched total FLOPs or latency. The selector is a deliberately simple baseline, not a PRM reproduction.

Completed: the 24-problem mechanism screen and separately frozen 24-problem prospective single-checkpoint deployment diagnostic. Gated extensions not pursued: strong published control baselines, sequential control, task/generator transfer, generator RL, and scientific-discovery evaluation. Advancement depends on instrument quality and measured decision signal, not on spending the full API allocation.

## Reuse

This harness was written independently; the inspected upstream repositories are research references, not vendored dependencies. Preserve upstream licenses if their code is later incorporated. The TypeSafe skill used for this work is [typesafe-ai](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md), installed in the user's local skills directory.
