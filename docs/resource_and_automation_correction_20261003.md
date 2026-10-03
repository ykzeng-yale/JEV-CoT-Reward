# Resource and automation correction — October 3, 2026

The user asked for a systematic correction of the recurring workflow and explicitly allowed deletion if it was not making progress. The six-hour heartbeat `jev-six-hour-scientific-research` was deleted through the app tool. Its successful delete receipt is preserved privately at `runs/resource-routing-20261003-user-correction/automation-delete-receipt.json`; the former automation TOML is absent. The previous prompt is retained in `automation-before.toml`. No replacement automation or goal was created. Historical heartbeat/active-job entries below the latest state pointers are historical observations, not current instructions or reservations.

## Actual compute state

Fresh authenticated direct SSH to Bouchet works. `squeue` contains no JEV/ReCoMA running or pending job. The current unrelated HAFT CPU jobs were left untouched. Receipt-derived `sacct` identities verify all ten final-phase parents as terminal; production 27983982 and CPU audit 27987263 remain COMPLETED 0:0. Project `pi_fl426` still permits normal-tier jobs. Its latest quota snapshot shows 553/4096 GiB project usage and 1/10240 GiB scratch usage. These establish access and storage eligibility, not a GPU reservation or a promised start time. Mixed partition state alone does not establish available schedulable GPUs.

There is **no current JEV GPU wait**. The next factorial experiment was not submitted. Its conservative prompt exists in the prepared design, but no complete executor currently installs that prompt and runs all four arms. The five prepared source patches are not yet a complete frozen runtime/input/submission bundle. The analyzer's claimed arm PASS status must be connected to actual independent provenance and endpoint auditing. These are implementation gaps. The expired September 30 generation window and 16 GPU-hour phase cap remain historical phase limits; a later bounded compute envelope was not documented. None of these facts can be reported as an inaccessible cluster or a pending experiment.

Local resource checks establish the following:

| Host | Observed capacity | Task suitability now |
| --- | --- | --- |
| Lead M5, 32 GiB | Approximately 100% CPU occupied, 31 GiB used, about 4 GiB disk free | Preserve unrelated Lean/R work; do not load another model. |
| Mac mini M4, 16 GiB | Approximately 72% CPU idle, 14.6 GiB disk free; existing Ollama workload | Bounded CPU analysis, simulator work and audits are possible subject to admission checks. |
| Auxiliary M2 Max, 32 GiB | Approximately 57% CPU idle, 24.1 GiB disk free; 3.8 GiB swap used and substantial compressed memory | CPU work is possible; a local model requires memory/latency qualification before admission. |

All three Macs are accessible. The lead's existing Qwen snapshot matches all ten pinned model files, 8,060,896,487 bytes, by actual SHA-256 checks. This proves weight identity, not local inference qualification. Mini and auxiliary Mac have no staged pinned Qwen/runtime for this experiment. The existing ReCoMA worker/backend explicitly require CUDA, which Apple GPUs do not provide. A separately implemented and frozen MLX experiment is feasible in principle, but cannot run this CUDA program unchanged or be pooled with its BF16 CUDA outcomes. A local port is an engineering task, not grounds to declare all local research impossible.

Raw scheduler, quota, process/start-identity and cache receipts remain private under `runs/resource-routing-20261003-user-correction/`. No model inference, download, process eviction, service change, Slurm submission or email occurred in this correction session.

## Close the unnecessary fresh-seed bookkeeping gate

Seven existing manifest/submission artifact hashes were independently checked and match. Recorded lineage supports seed 1 as fresh **model-trajectory development evaluation**: the structural CPU study included seeds 0–4 without model generation; the proposed full model panel was cancelled before allocation; actual production used seed 0; runtime qualification used seed 5. The detailed digest receipt is `seed1-recorded-lineage.json` in the private correction directory.

Seed 1 has prior simulator reset/replay exposure and shares the same eight task families. It is neither completely untouched final data nor a transfer evaluation. This closes the documented model-trajectory provenance question without another pilot; it does not release the unfinished executor or silently amend an immutable prepared design.

## Remaining work in scientific dependency order

1. Finish a single complete operational-baseline executor: original/conservative prompts by 30/90-action horizons, four immutable arm manifests, durable stop/cost records, and the existing independent v3 auditor adapted to each arm. Validate the complete execution paths together with CPU fixtures. Observational stop tags alone do not justify another tiny scientific GPU pilot. The 24-unit, 96-cell development experiment has a hard upper bound of 11,712 calls and 4,684,800 generated tokens; its 10-pp decision margin is not a powered confirmatory guarantee.
2. Establish useful observable action choices where a strong cheap baseline still has headroom, using complete continuation/intervention/sham paired outcomes and independent endpoints. Horizon improvement is not action-selector value.
3. Only after that action prerequisite passes, compare static, cheap/local, sham, random/rate-matched, compatible published controllers and Jev under fair total costs. Preserve the service-use gate and existing cumulative $25 Jev ledger.
4. Only after controller qualification, freeze independent family/transfer and sequential evaluations. Shared seed variation is not transfer. Depth/trim/prune, training and discovery stages remain conditional; launching them all now would not be scientifically justified.

The unchanged evidence supports a pivot rather than scaling the original Jev claim. The audited cheap conductivity measurement signal is real in its finite development panel, but a saturated cheap decoder provides no demonstrated Jev headroom. The ReCoMA baseline remains 0/24 official success under its frozen envelope. No new positive Jev increment, intervention effect, theory novelty or completed full curriculum follows from this resource correction. Removing the stalled automation does not claim research completion.
