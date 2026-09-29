# Paper evidence and next decisions

## ScienceWorld qualification and live model job — September 29, 19:31 EDT

- [x] Source-pinned ScienceWorld 1.3.0 CPU preflight, Slurm `27913546`: full 30-task / 7,207-variation inventory and split partition integrity; 9/9 selected held-out-variation reset/one-action digests match; 199 output hashes verified; 4 CPU/16 GiB/87s/0 GPUh; zero model/Jev/network/gold-path calls. Independent auditor and focused tests pass. This is reproducibility feasibility, **not** action usefulness or OOD evidence.
- [ ] Full DiscoveryWorld ReCoMA baseline `27880628` is still pending under `pi_fl426`/normal with no allocated GPU-hours. Current start estimate: September 30, 20:44 EDT; 2,386/2,386 frozen input hashes pass. Do not duplicate it.
- [ ] ScienceWorld's train/dev-only checkpoint contract and an agent-level controller comparison are not yet frozen or run. Its 345 candidate test variations span only three task templates; no unseen-family, innovation, or small-effect claim follows from those counts.
- [ ] Fresh held-out action/controller experiment remains gated on baseline characterization, common-prefix protocol, cost-matched cheap/local controller, train/dev-only selection, preserved test split, and TypeSafe written clarification before any Jev use.
- [x] Source-level endpoint audit found that unknown melting, conductivity, and Mendel task measurement/experiment actions are optional score goals while main success requires the target answer action; official aggregate `score/reward` is exposed by the Python API. A future study must make independently audited final classification the primary endpoint, aggregate score secondary/decomposed, and equalize score visibility across arms. See `docs/scienceworld_task_objective_audit_2026-09-29.md`.

## Live execution checkpoint — September 29, 19:04 EDT

No scientific outcomes since the prior audited development result. ReCoMA full-panel job **27880628** is still pending under `pi_fl426`/normal; 0 GPU-hours, exact input hashes PASS 2,386/2,386, estimated start Oct 1 20:44 EDT. A single same-account array test-only now estimates Oct 2 03:22 EDT, so no replacement was made. Full visible-ReAct trajectory and operational-cost audit still depends on allocation and terminal completion. The local Macs remain occupied or unsuitable for competing generation. See the exact queue receipt and dated state record.

## Live execution checkpoint — September 29, 17:04 EDT

No new outcome data since the audited repair replication. ReCoMA full-panel baseline **27880628** remains pending (`Priority`), zero GPU-hours, with an estimated start September 30, 14:01 EDT. Its 2,386-entry checksum gate passes. A test-only four-by-two-GPU alternative forecast an array element later than the existing request, so the current submission remains intact. This is only a queue decision; no baseline or efficacy result has completed.

## Execution and audit update — September 29, 15:10 EDT

The latest full-panel ReCoMA baseline, Slurm **27880628**, is still pending with no allocated GPUs or GPU-hours; no outcomes exist yet. Its submitted input tree's exact checksum gate is now verified at the correct remote path: 2,386/2,386 entries passed after byte-exact `.pyc` restoration. Do not count infrastructure success as reasoning/controller evidence.

The local repair-action replication previously had a stale `running` process receipt. Its full saved data now pass a fresh audit and reanalysis (24 problems, 480 outcomes, 631 calls, zero label disagreements). Continue 63/96, sham 68/96, suffix repair 57/96, recheck 56/96, and segment repair 54/96. Segment repair vs sham is −14.58 pp (unadjusted descriptive clustered interval −27.08 to −2.08); due to multiple exploratory contrasts, two families, and repeated outcomes per problem, treat this as development evidence only. The exact tested repairs do not merit scale-up; Jev's incremental value, transfer, and discovery remain untested. See the dated run reconciliation in `docs/current_research_state.md`.

## Current evidence status — September 29, 2026, 14:48 EDT

- [x] DiscoveryWorld patched structural feasibility: all 120 configurations, 24 strata, initial/post-action exact replay, endpoint separation and source hashes independently passed (`results/discoveryworld_e1a_runtime_patch2_lead_audit.json`). Scope is reproducibility under a disclosed patch only; no LLM or Jev outcomes.
- [ ] Full official-panel published ReCoMA-ReAct baseline: Slurm `27880628` is PENDING, `AllocTRES=(null)`, 0 GPU-hours. Receipt `runs/bouchet-recoma-discoveryworld-full-v2-control/submission.json`. It is Qwen3-4B-Instruct-2507 BF16 and requests externalized per-action `thought`; it does not expose hidden/internal thinking-mode CoT. No duplicate while pending; full terminal trace/resource audit required before analysis.
- [ ] Natural action usefulness and a common-prefix intervention/sham/randomized comparison remain untested. The 120-episode official panel is finite and underpowered for small effects. Preserve a fresh evaluation set.
- [ ] Jev incremental value versus strong cheap/local and compatible published controls remains untested and Jev API use remains gated by written terms clarification; current Jev spend for this run is zero.
- [x] InterWhen/K-stable published-baseline adaptation independently audited: 54/96 versus continuation 60/96; −6.25 pp (descriptive interval −14.58 to +2.08). Negative/imprecise; no Jev inference.

## September 29 consolidated experiment program

The full remaining study sequence and gate dependencies are in
`docs/remaining_experiment_program_v1.md`. Use its task-level power calculator
(`scripts/paired_binary_power.py`) before freezing any fresh efficacy sample.
Current status: no candidate task/action passes all gates; Jev-dependent runs
await written terms clarification; InterWhen adaptation job 27857096 is
pending and must be audited before any outcome interpretation. Do not describe
that baseline adaptation as Jev evidence.

Terminal-Bench v4.0.0 source audit found 66 heterogeneous tasks, not a powered
population for a 3-pp confirmatory comparison; `freight-dispatch-shift` is at
most a one-task mechanism case study. See
`docs/terminal_bench_v4_task_manifest_audit_2026-09-29.md`.

## September 29 prior-art boundary additions

- [ ] Do not claim first Jev use for scientific semantic decisions; compare with [Jev for Scientific Decisions](https://arxiv.org/abs/2609.24965).
- [ ] Do not claim first Jev long-horizon agent control; distinguish [JEV-Star](https://arxiv.org/abs/2609.27331), including the preprint's limited/confounded evidence, from reasoning-prefix interventions.
- [ ] Do not claim first Jev use in intervention studies or downstream population outcomes; distinguish [KITE](https://arxiv.org/abs/2609.27535), whose intervention target is simulated human behavior.
- [ ] Candidate study remains narrow: visible LLM prefix + explicit intervention + independent terminal validator + full cost + cheap/local/published controls + fresh transfer. This empirical boundary is not a novelty claim already established.
- [ ] No Jev-dependent sampling/training until provider terms are clarified in writing.
- [ ] Treat interwhen TTS Game24 as direct monitor/repair/early-stop prior art, not a merely adjacent baseline.
- [ ] Charge interwhen's same-model expression-extraction side stream; compare to a matched extraction/no-feedback arm because it can itself generate a solution.
- [ ] Independently re-score terminal answers; do not reuse interwhen's monitor `verify_expression` as the sole outcome evaluator. Reject Python operators beyond the task grammar.
- [ ] Before any Game24 run, decide direct-reproduction versus adaptation, pin model/runtime/task split, audit answer extraction, and resolve the lack of a Bouchet vLLM module with a bounded runtime plan.

## September 29, 06:16 EDT prior-art update

Add *REFLEX with Jev* (arXiv:2609.26532v1) to the required direct-JeV comparison set. It already demonstrates Jev as an in-trajectory bounded-action layer with confidence-based fallback, cross-fallback tests, and a cheap generative cascade comparator. Our distinct empirical question cannot be “does Jev supervise an agent?” It would have to isolate exposed reasoning-prefix repair/branch decisions and independently measured terminal outcomes under complete matched costs, with a compatible published baseline. *Jev vs. LLMs as Rubric Judges* (arXiv:2609.29769v1) adds evidence about label agreement and correlated judge errors but does not answer action utility. See the dated prior-art audit for exact overlap and limits. Neither paper supplies a new positive result for this project.

Current prerequisites remain unresolved: the arithmetic feasibility gate failed; the audited repair/branch pilots do not show a reliable benefit; the official interwhen runner is not runtime-qualified; and written TypeSafe clarification is needed for the proposed Jev-output-conditioned academic controller and any derived release. Do not convert the completed prior-art review into an efficacy claim or restart the closed studies.

## September 29 update: horizon feasibility and novelty boundary

The frozen 120-task eight-operand arithmetic screen ended normally and its independent audit passed. It yielded 15 incorrect first-addition claims among 41 explicit claims, below the preregistered 20-error gate; it collected no terminal outcomes and made no Jev calls. A secondary equation parser is exploratory and does not change the gate. Do not claim an arithmetic correction benefit or continue this intervention lane under the same checkpoint rule.

The prior-art audit finds direct duplication of the previously proposed action-value estimand and same-prefix branching protocol in *Calibration Is Not Control*; *CausalFlow* covers step-level counterfactual repair; *interwhen* covers trace monitoring and verifier-guided steering; and the newly audited *Learning What to Skip* (LW2S, arXiv:2609.30734v1) directly covers action-specific counterfactual skip safety, calibration, guards, and sequential fall-through across math, QA, and code. No methods-level novelty gap is established. The only plausible residual is Jev's incremental value as a judgment source against an LW2S-style local safety model on the same task/action/prefix and fully matched cost, contingent on fresh task/action qualification and written TypeSafe terms. LW2S's arXiv source archive lacks runnable scripts/data although its paper mentions separate supplements; inspect and hash these artifacts before claiming a replication. Exact notes: `docs/lw2s_prior_art_audit_2026-09-29.md` and `docs/prior_art_gap_audit_2026-09-29.md`.

Jev-output-conditioned controller training is paused pending TypeSafe clarification of current MCA §2.3(b). No Jev request was made in this update. Local-only baseline qualification and independent outcome studies remain available.

## September 29, 02:20 local launch note — superseded by the 04:28 handoff above

The six-number opportunity screen audited 120/120 fresh checkpoints and found ten incorrect first-addition claims (8.3% of eligible checkpoints; Wilson 95% interval 4.6–14.7%). Its frozen gate for action rollouts was 20, so that study correctly stopped before outcomes. A fresh eight-operand horizon screen (PID 56403) tests whether more complex, longer arithmetic construction yields enough natural intermediate errors. If it meets the gate, the next study must still compare correction, matched neutral feedback and continuation with all arm costs; the feasibility screen itself cannot support efficacy or Jev value. If it misses, abandon this simple equality-correction route and move to another independently checkable obligation/task family.

## September 28, 22:15 local evidence update

The 25-checkpoint, 50-continuation verified arithmetic-correction development test is terminal and independently audited (53 calls, zero outcome disagreements, zero timeouts). Correction 42/50 versus prior matched sham 40/50; problem-weighted difference +4 points, descriptive interval [-6,+16]. Five paired episodes favored correction, three favored sham. It changes the next decision: a detectable explicit arithmetic error does not yet justify spending computation on this correction action. The arm processed 3,347 additional prompt tokens, while measured service time was 16 seconds lower in this nonconcurrent comparison; neither is a complete compute-cost improvement claim. Do not scale the same action or fit a controller to its 25 selected checkpoints. The next scientific bottleneck is to qualify a treatment with repeatable terminal usefulness on fresh states, then test whether Jev semantic features add value beyond executable checks, cheap text and local judges.

Earlier static-sham replication and Jev narrow arithmetic-sensor findings are in `docs/current_research_state.md` and `paper/main.tex`. The sensor's poor 0.5-threshold specificity (1/25 natural false claims) narrows its viable role; it does not establish failure on contradiction, uncertainty or exploration-role judgments. Closed v1 data remain excluded from adaptation.

Updated September 27, 23:56. Development findings are not confirmatory evidence. Closed v1 test remains closed.

| Claim or prerequisite | Evidence | Remaining requirement |
|---|---|---|
| Exact-prefix and budget instrumentation | Audited MLX and CUDA runs | Keep immutable source and call audits for every changed controller |
| Semantic selector increment | V3 Jev42/96 vs likelihood43/96, continue48/96 | No demonstrated increment; do not scale unchanged selector |
| Adaptive branching beyond interruption | CUDA GUARD6/12 vs sham6/12 | Randomized baseline audited:GUARD/random/sham8/12, two discordant GUARD/random tasks; fixed96-problem precision study27727778 submitted |
| Fair interruption and runtime envelope | Continue had one180s call timeout; segmented policies none | Label historical comparison runtime-censored; any prospective comparison must freeze an equal episode deadline and compatible call caps, or explicitly target those operational policies |
| Static and randomized policies | Static continuation and segmented sham; random timing now running | Random opportunity rate does not guarantee matched intervention counts or cost |
| Learned intervention value | V1 exploratory models showed no gain | Establish useful actions and fresh outcome supervision before expensive controller training |
| Strong cheap/local/Jev attribution | Common-pool selectors audited | Separate local judge and learned cheap-text controls still needed for general claims |
| Published comparator | Explicit segment-boundary GUARD adaptation qualified | Do not call faithful reproduction; document source differences |
| Sequential, horizon and generator transfer | Not established | Fresh studies only after a useful frozen controller; no pooling BF16/MLX |
| Theory | Conditional identification/regret/acquisition statements in draft | Verify assumptions match deployed estimand; no arbitrary OOD guarantee |
| Reproducibility | Frozen records, safe releases, audit-bound analyses | Complete final release and dependency/model/accounting manifests |

Next decision depends on the complete randomized timing experiment. If GUARD and random/sham remain indistinguishable with sparse intervention, do not repeatedly enlarge tiny timing pilots. Diagnose action opportunity and continuation budget, then specify a substantive fresh action-value study with a uniform runtime contract and a sample-size calculation. If an effect appears, use it only for design and power planning; reserve fresh data for confirmation. Neither outcome justifies claiming Jev efficacy.

Use paired problem-level variance for planning, but do not set sample size to zero from zero observed discordance. Report sensitivity over plausible discordance (e.g.0.1–0.3), minimum meaningful effect and actual throughput. Evaluate precision within families only where enough independent problems exist. Degenerate family-stratified bootstrap intervals in these tiny deterministic-outcome samples are not certainty/equivalence certificates.
