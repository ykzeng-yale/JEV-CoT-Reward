# Bouchet execution track

## September 30, 09:08 EDT — authenticated live-state refresh

Using the authenticated `mac-aux → Bouchet` route, `squeue -u yz2324` identifies task-owned ReCoMA baseline **27880628** as `PENDING` (reason `Priority`) on the frozen `pi_fl426`/normal H200/B200 request. `sacct -j 27880628` reports `PENDING`, `ExitCode=0:0` placeholder, elapsed `00:00:00`, 0 allocated CPUs, no `AllocTRES`, and no start/end time; therefore **0 GPU-hours have accrued**. Its receipt still pins the same workdir, Qwen3-4B-Instruct-2507 BF16 runtime and 2,386/2,386 input checksum pass. No duplicate or alternative submission was made. The large running/pending `pi_gt353` array belongs to the separate WDSM run and was not touched. Current association/fair-share values were inspected for both authorized accounts; this is not a reason to switch the specific frozen ReCoMA run's sponsor.

CPU control `27941320` is terminal `COMPLETED 0:0` (4 CPUs/16 GiB, 00:03:16, MaxRSS 2,116,344 KiB, 0 GPU-hours) and has already been independently audited. No task-owned model inference is running on the Macs. Lead PID **83281** still matches the unrelated process's recorded command and September 27 09:09:22 start identity; PID **57951** is absent. Mini has an existing Ollama service and aux has a Docker workload with heavy swap; no service or process was stopped.

The next compute-dependent step is unchanged: wait for actual allocation/start of `27880628`, then monitor by Slurm ID, retrieve complete traces after terminal state, verify hashes/resources, and run the independent result auditor before any outcome-rate analysis. Pending estimates are not reservations and the current queue state is not scientific evidence.

## September 30, 03:16 EDT — ARMAP data-provenance gate; primary job still queued

The ARMAP code/checkpoint audit shows the released ScienceWorld reward model scores complete transcripts and triggers retry/reflection; it cannot be called a same-prefix action selector. Its pinned 4,064-row public training corpus contains two records with both “dominant” and “recessive” terms but exposes no task/variation IDs. Exact held-out sample disjointness is therefore unverified, not proven false. The saved scanner writes only aggregate term/schema counts and pinned hashes; three focused tests pass. It emitted no raw rows, task labels, preference values, or trajectories; no model inference, Jev calls, or project-held-out outcomes were used. Keep this checkpoint out of the primary independent comparison pending a provenance gate; consider only a separately frozen train-only adaptation if justified. See `docs/scienceworld_published_baseline_screen_2026-09-30.md`.

Authenticated `squeue`/`sacct` at 03:06 EDT show full-panel DiscoveryWorld ReCoMA job **27880628** `PENDING/Priority`, zero allocated GPUs/GPU-hours. `scontrol` at 03:13 EDT confirms account `pi_fl426`, normal QOS, 8 GPUs/32 CPUs/256 GiB, and the same immutable workdir `/nfs/roberts/project/pi_fl426/yz2324/discoveryworld-recoma-react-full-v2`. Preserve the job; `StartTime=2026-10-01T10:36:58` is only an estimate. No competing model job was submitted. Local lead PID 83281 still belongs to the unrelated WDSM worker at its receipt-recorded start time; mini/aux are not qualified for this project model. The outcome-dependent next decision waits on actual allocation and independent full-panel audit.

## September 30, 00:10 EDT — pending GPU job reconfirmed; published baseline screen completed

Authenticated `squeue`, `scontrol`, and `sacct` reconfirm job **27880628** is `PENDING/Priority`, `AllocTRES=(null)`, zero elapsed time and zero GPU-hours under `pi_fl426`/normal. Its 8-GPU H200/B200, 32-CPU, 256-GiB request and immutable workdir are unchanged; `StartTime=2026-10-01T10:36:58` remains only an estimate. Do not duplicate or switch accounts.

Independent ScienceWorld source/literature work is now complete for this round. The no-model firewall remains locally audited 18/18. Published baseline screen finds ARMAP's official ScienceWorld runner uses full-trajectory scoring plus reflection/restarts (not a same-prefix action selector), Llama 3.1 8B, a 10-step config, and a custom JAR. It is a possible separately labeled end-to-end comparison or a substantial port, not a directly compatible comparator. Exact source review: `docs/scienceworld_published_baseline_screen_2026-09-30.md`. No additional cluster job was submitted while the full-panel GPU job remains pending.

## September 30, 00:05 EDT — CPU firewall run audited; GPU baseline remains queued

The authenticated `mac-aux -> bouchet` route verified full-panel job **27880628** remains `PENDING/Priority`, with no allocated TRES or GPU-hours. It remains under `pi_fl426`/normal, requests eight H200/B200 GPUs plus 32 CPUs and 256 GiB, and has immutable workdir `/nfs/roberts/project/pi_fl426/yz2324/discoveryworld-recoma-react-full-v2`. Latest mutable `StartTime=2026-10-01T10:36:58` is not a reservation. Keep this existing request; do not duplicate or switch accounts.

Separate no-model ScienceWorld firewall job **27927761** is terminal `FAILED 1:0`, elapsed 2m22s, 2 CPUs/8 GiB, peak RSS 3,834,128 KiB, zero GPU-hours. The runner completed all 360 train/dev cases; failure was only the remote independent audit hashing reserialized rather than exact protocol bytes. Retrieved files/checksums are preserved under `runs/bouchet-scienceworld-mendelian-firewall-v1-control/retrieved/`; corrected local independent audit passes 18/18 checks. No retry is warranted. This verifies direct `info`-metadata exclusion by the adapter, not semantic leakage or efficacy. See `docs/current_research_state.md` and the result/resource JSON for the exact audit trail.

## September 29, 23:48 EDT — CPU job audited; separate GPU job is genuinely queued

The `mac-aux -> bouchet` authenticated route is working. CPU job **27868803** is receipt-backed and terminal (`COMPLETED 0:0`, `pi_fl426`/normal `day`, 1 CPU/8 GiB, 00:05:03, 690,208 KiB peak RSS, 0 GPU-hours). Full remote output checksums and the seven retrieved result/environment artifacts pass; the independent cluster and local structural audits both pass 120/120 instances and 24 strata, with no model, outcome-rate, or Jev data. Complete retrieval receipt and scheduler evidence: `runs/bouchet-discoveryworld-e1a-v2-retrieved-27868803/retrieval-receipt.json`.

Separately, job **27880628** remains `PENDING/Priority`, with no allocated TRES or GPU-hours. It asks for one eight-GPU H200/B200 node plus 32 CPUs/256 GiB for 48 hours under `pi_fl426`/normal; current mutable estimated start is Oct 1 06:06:56 EDT. This is a scheduling wait, not a credential or resource-discovery failure. Keep the existing request; do not duplicate or switch account. The CPU job completion clears only the Linux replay gate; it cannot replace the queued model-based natural-checkpoint/cost census. Continue the independent action/evaluator/baseline audit while it waits.

## September 29, 23:06 EDT — eight-GPU study remains queued

Fresh scheduler records over the authenticated `mac-aux` route: job **27880628** is `PENDING/Priority`; `AllocTRES=(null)`, elapsed `00:00:00`, and 0 allocated GPU-hours. Account/QOS are `pi_fl426`/normal. Request is still 8 GPUs, 32 CPUs, 256 GiB, 48 h; immutable `WorkDir` is `/nfs/roberts/project/pi_fl426/yz2324/discoveryworld-recoma-react-full-v2`. Mutable `StartTime=2026-09-30T22:36:50` is not a reservation. The frozen task receipt remains `runs/bouchet-recoma-discoveryworld-full-v2-control/submission.json`; last exact input-tree verification was 2,386/2,386. Preserve this job and wait for actual allocation. No new model workload was submitted because local lead/mini/aux workloads remain in use.

## September 29, 21:18 EDT — authoritative status refresh

Authenticated `squeue`, `scontrol`, and `sacct` over the trusted `mac-aux` route agree that task-owned job **27880628** remains `PENDING/Priority` under `pi_fl426`/normal: `AllocTRES=(null)`, elapsed `00:00:00`, and **0 GPU-hours**. The request remains 8 GPUs, 32 CPUs, 256 GiB, 48 hours; `WorkDir=/nfs/roberts/project/pi_fl426/yz2324/discoveryworld-recoma-react-full-v2`. Slurm's current `StartTime=2026-09-30T23:12:53` is mutable and nonbinding. Preserve this one job; no duplicate or account/QOS changes. The manifest was previously verified 2,386/2,386, and its frozen receipt is `runs/bouchet-recoma-discoveryworld-full-v2-control/submission.json`.

The independent ScienceWorld Mendelian source audit now has corrected, archive-verified member hashes and explicit endpoint/precision limits (`docs/scienceworld_mendelian_candidate_audit_2026-09-30.md`). The audit does not consume the queued GPU baseline or establish outcomes. The next cluster-side action remains waiting for actual allocation; on transition, inspect scheduler allocation, run logs and the independent panel auditor before interpreting any outcome.

## September 29, 20:10 EDT — prefix replay result audited

Slurm **27916067** is terminal `COMPLETED 0:0`: standard `day`, `pi_fl426`, 4 CPU/16 GiB, 126 seconds, no GPUs. The independent result auditor confirms 9 frozen test-variation episodes / **195 of 195** one-step transitions matched across the paired runtime copies; all 199 run-manifest files passed. This is technical prefix replay only under the pinned version and fixed random-legal-action sequences; it does not show that action branches improve outcomes or that all generated prefixes replay. Conservatively exclude those nine variation IDs from any future untouched confirmatory endpoint.

The separate baseline **27880628** remains `PENDING/Priority`, no allocation or GPU-hours. Current `StartTime` is a mutable September 30, 20:44 EDT estimate. The main 2,386-file checksum gate still passes; do not duplicate or alter its immutable inputs.

## September 29, 20:02 EDT — task-owned jobs reconciled

- **27916067 `scienceworld-prefix`**: `PENDING/Priority`, `pi_fl426`/normal `day`, 4 CPU/16 GiB/45 min/0 GPUs, `AllocTRES=(null)`, `StartTime=Unknown`; no allocation yet. Test-only pseudo-ID `27916064` estimated Sep 30 11:04:41 but is not a reservation. The six-file input integrity gate passed remotely. Receipt: `runs/scienceworld-prefix-replay-v1-control/submission.json`.
- **27880628 `recoma-dw-full-v2`**: still pending, 8 GPU/32 CPU/256 GiB, `AllocTRES=(null)`, 0 GPU-hours, `StartTime=2026-09-30T20:44:04` estimate. Exact 2,386-file gate passes. No task-owned GPU inference has started.
- **27913546 `scienceworld-preflight`**: terminal `COMPLETED 0:0`, 87 seconds, CPU-only; the result and 199-file output manifest independently pass. It is infrastructure evidence only.

The two new CPU study artifacts live in separate immutable roots. The prefix job was submitted with `WorkDir=/home/yz2324`; its script has been verified to change to the absolute input root before checksum validation and uses absolute run/log output paths. This submit-directory detail is recorded, does not alter input provenance, and is not a reason to cancel/restart it.

## September 29, 19:31 EDT — no-model ScienceWorld qualification completed; GPU baseline pending

The separately frozen CPU-only ScienceWorld preflight ran once as Slurm **27913546** under `pi_fl426`/normal on `day`; `sacct` reports `COMPLETED`, `0:0`, 87 seconds, 4 CPU, 16 GiB, no GPU (0 GPU-hours). The pinned Java/Python/py4j runtime executed the 30-task / 7,207-variation split check and nine held-out-variation, one-legal-action deterministic replay pairs. The result reports 9/9 matching state digests, with no model, Jev, network, raw-content, score, or gold-path outputs. All 199 run-manifest files verify; the local independent audit is `results/scienceworld_transfer_preflight_27913546_audit.json`. This is only environment/replay feasibility; it does not indicate controller performance or OOD transfer.

Main job **27880628** remains `PENDING/Priority`, `AllocTRES=(null)`, 0 GPU-hours. Latest `scontrol StartTime=2026-09-30T20:44:04` is an estimate only. Its exact submitted input gate remains 2,386/2,386; the 2-GPU-array `--test-only` estimate (element Oct 2 03:22 EDT) is later, so keep the current full-node request. Next decision depends on that baseline's actual startup, terminal trace audit, and costs. No Jev call has been made.

## September 29, 19:04 EDT — refreshed queue comparison

Job **27880628** remains `PENDING/Priority` under `pi_fl426`/normal, with no allocated TRES and 0 GPU-hours. The mutable `scontrol StartTime` is **October 1, 20:44 EDT**. The remote frozen input manifest again passed **2,386/2,386** checks. Since the estimate moved later, one current `--test-only` comparison of the previously considered four-element 2-GPU array estimated a single element at **October 2, 03:22 EDT**. This is later than the existing estimate and is not an array completion forecast. No replacement or duplicate was submitted. Two unrelated user jobs were observed by their `scontrol` working directories and left untouched. Keep the existing job and recheck its authoritative receipt/scheduler identity next time.

## September 29, 17:04 EDT — latest queue reconciliation and bounded alternative check

Authenticated scheduler state for task-owned job **27880628**: `PENDING`, reason `Priority`, `AllocTRES=(null)`, `sacct` elapsed zero, and **0 GPU-hours**. The mutable start estimate remains **September 30, 14:01:23 EDT**; it is not a reservation. `inputs/SHA256SUMS` was checked again at the path used by the submitted script: 2,386/2,386 files pass. No outcomes or inference process exist yet.

One scheduler-only comparison tested a potential four-element 2-GPU array using the same account (`pi_fl426`), normal QOS and compatible H200/B200 partitions. `sbatch --test-only` estimated an element at October 1, 15:11 EDT, later than the existing 8-GPU request's estimate. It is not a real job and does not reserve resources; it also does not estimate when all array elements would finish. The result does not justify canceling/replacing the current job. No second submission, account switch, or allocation occurred. See the current local receipt for the exact test-only response and decision.

## September 29, 15:05 EDT — checksum gate reverified after exact-byte repair

The final verification used the directory the submitted script actually checks, `.../discoveryworld-recoma-react-full-v2/inputs`, rather than the run root. Remote `sha256sum -c SHA256SUMS` returned **2,386/2,386 OK, zero failures** after all 61 regenerated CPython bytecode files were restored from the bundle whose SHA-256 is recorded in the receipt. The scheduler still reports job **27880628 PENDING**, `ReqNodeNotAvail,_May_be_reserved_for_other_job`, no allocated TRES, elapsed `00:00:00`, and zero GPU-hours; its current estimated start is September 30 14:01:23 EDT, not a reservation. No duplicate was submitted. The script hash, panel hash and v2 protocol hash remain unchanged.

The independently reviewed outcome auditor and its tests were copied to the remote run root (outside `inputs/`); remote hashes match the local copies. The exact input archive and submitted script remain the source of truth for job execution. For future CPU preflights that import bundled Python source, disable bytecode writes or operate on a disposable copy so the act of checking a frozen bundle cannot mutate it. Do not patch this already-submitted job in place.

## September 29, 14:48 EDT — DiscoveryWorld full-panel ReCoMA baseline queued

The latest task-owned GPU job is **27880628** (receipt `runs/bouchet-recoma-discoveryworld-full-v2-control/submission.json`), submitted once under `pi_fl426`/`normal` at 14:41:25 EDT. At 14:45:59 EDT, `squeue`, `scontrol`, and `sacct` agreed it was `PENDING` with `ReqNodeNotAvail,_May_be_reserved_for_other_job`, `AllocTRES=(null)`, and zero elapsed/GPU-hours. `scontrol` currently estimates September 30, 14:01:23 EDT as the start; this can move and is not a reservation. No duplicate submission is appropriate.

The immutable remote root is `/nfs/roberts/project/pi_fl426/yz2324/discoveryworld-recoma-react-full-v2`; command is `discoveryworld_recoma_react_full_v2.sbatch`. Request: one node, 8 GPUs on `gpu_h200,gpu_b200`, 32 CPUs, 256 GiB RAM, 48 hours, standard QOS. The eight worker processes cover the full official 120-task / 24-stratum × five-seed panel. Frozen panel hash: `84175595f412277b614d2c06773412994fdb9e26c3730ded851ff486f04ade8e`; Slurm script hash: `6ef3127a2909e89a2ffe0839d76ba42265ce235c4638d2f19e312368e20015e1`; uploaded input tarball hash: `aca00780b20f7df9664497a0665adfab90ca6992b7a56ffefed69cd4fdbf306f`. Remote `SHA256SUMS` passed all bundled files. CPU-only no-generation preflight passed one complete 15-task shard (summary copied to the local receipt directory); the allocated job independently preflights all eight shards before generation.

**Study boundary:** this is an open-weight Qwen3-4B-Instruct-2507 BF16 adaptation of the published ReCoMA ReAct baseline—not a GPT-4o reproduction or a Jev/intervention effect. Its explicit visible per-action `thought` field measures externalized ReAct traces, not hidden/internal thinking-mode CoT. It is development-only and cannot support OOD or innovation claims. Worst-case completion ceiling is 2.976M tokens; actual usage is determined from per-call traces after completion. No Jev credential was transferred and no Jev request is configured.

One unrelated pending WDSM job, **27878675**, was confirmed by its work directory on one B200 under `pi_gt353`; leave it untouched. Together, the requests remain within the user's verified 16-GPU limit. On completion of 27880628, collect exact scheduler accounting, logs, all worker outputs and checksums, then run the versioned local independent outcome auditor before inspecting rates. The current auditor adds Slurm-success/account/allocation and generation-model/token-cap acceptance checks; these do not modify the running job or its immutable inputs.

Access verified September 27, 2026 through `mac-aux` and its authenticated Bouchet connection. Direct lead-Mac SSH reaches interactive authentication but is not yet usable noninteractively. The auxiliary route works without moving private keys or weakening MFA. Both `pi_fl426` and `pi_gt353` list normal/interactive associations. This project's first job uses the current default `pi_fl426`; do not switch PI accounts merely for better priority.

## Submitted job

Job **27713897**, `jev-cuda-qual-v1`, submitted at 21:21:54 Eastern. Actual Slurm submission succeeded; initial state PENDING, not yet a completed experiment. One GPU eligible in `gpu_h100,gpu_b200,gpu_h200`, four CPUs, 32 GiB RAM, 30 minutes, normal QOS. A single multi-partition submission avoids duplicate allocations. H100 test-only estimates were earlier than H200/B200 at inspection, but those are neither reservations nor actual start guarantees.

Files: `cluster/bouchet/qualification.sbatch`, `cluster/bouchet/qualify_cuda.py`. Local authoritative receipt: `runs/bouchet-cuda-qualification-v1-control/submission.json`; remote run is below the user's pi_fl426 project directory in `JEV-CoT-Reward/runs/cuda-qualification-v1-20260927`. Receipt and exact script hashes were copied there too.

The public ungated model is Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Precision is BF16, not local MLX 4-bit. Pinned Yale modules: PyTorch 2.9.1/CUDA 12.8 and Transformers 4.55.2. Both imports were verified on the login host; inference occurs only inside the allocated GPU job. PyTorch's official [Blackwell support notes](https://pytorch.org/blog/pytorch-2-7/) support choosing a CUDA 12.8 stack, but actual allocated hardware still needs the runtime check.

Quota was checked before authorizing the approximately 8 GB model download: FL project reported 532/4096 GiB used and scratch 1/10240 GiB used (daily quota snapshot). Cache resides in project storage; script rejects a snapshot footprint exceeding 16 GiB. No API credential, private dataset or held-out research test was transferred.

## Qualification and scale-up gate

The job records hardware/runtime/model identity, exact-token greedy continuation versus pause/resume, an 8k context probe, four-way batched generation, throughput and peak GPU allocation. These are infrastructure checks, not evidence of Jev efficacy. A prefix mismatch stops promotion; diagnose numerical/generation differences before changing protocol. A nonzero exit requires logs and Slurm accounting review, not a blind retry.

After termination, retrieve every output, stdout/stderr, the submission receipt and `sacct` accounting. Check model revision, actual GPU, outputs and all assertions. Do not combine BF16 outcomes with the existing 4-bit study. The next justified step is a CUDA backend implementation and fresh runtime/action qualification with separately frozen configuration. Only then distribute counterfactual rollouts across independent GPU workers. Larger B200 allocations must be justified by observed memory/throughput and task size; a 4B pilot does not require an eight-GPU node.

The original Mac worker continues unchanged. Monitor both identities on recurring research turns, without duplicate submissions or another local inference process. The two jobs are independent and use distinct hardware.

## September 29, 11:26 EDT — InterWhen K-stable baseline adaptation

The first runtime/outcome qualification of the published InterWhen K-stable Game24 stopping baseline is queued as **job 27857096**. Submission receipt: `runs/bouchet-interwhen-kstable-game24-v1-control/submission.json`; source commit `06dc08aeab5ad538b11bca5a26e7065232f479db`; remote code checkout `/home/yz2324/project_pi_fl426/yz2324/JEV-CoT-Reward-kstable-06dc08a`. It uses the existing cached Qwen3-4B-Instruct-2507 snapshot, one standard-tier GPU, 4 CPUs, 32 GiB and six hours. Frozen 96-task public sample, task hash, independent solvability audit, paired arms and outcome/cost auditor are in `docs/interwhen_kstable_game24_v1_protocol.md`.

Actual scheduler check: `squeue -j 27857096` reports **PD** (pending); `scontrol` reports `JobState=PENDING`, `StartTime=Unknown`, and no allocated TRES. The nonbinding `sbatch --test-only` estimate was September 30 at 02:54:58 on H100. There is no GPU usage until allocation. Do not resubmit. Once running, continue monitoring this exact job; after terminal state retrieve its outputs and logs, then run the independent audit before analysis. This is a Qwen3-4B adaptation of the public InterWhen comparator, not a reproduction of its larger thinking model, a Jev comparison, or evidence that our main task/action gate passes.

## Failure diagnosis and bounded retry

Job 27713897 actually received an NVIDIA B200 (driver 580.178.04, CUDA 12.8 visible) and failed after 23 seconds with exit 132 during `hf_xet` model download. The retrieved trace identifies SIGILL in the native Xet download client; inference had not begun. Logs and manifest are preserved locally under `runs/bouchet-cuda-qualification-v1-retrieved`. This is an infrastructure failure, not a failed model experiment.

Changed retry **27714091** uses a new immutable `cuda-qualification-v1b-20260927` directory, the same resources/model/precision, and `HF_HUB_DISABLE_XET=1` with a startup assertion. This documented [Hugging Face setting](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet) disables the native transfer library; ordinary HTTP download is used. Core dumps are disabled for the retry. Its receipt is `runs/bouchet-cuda-qualification-v1b-control/submission.json`; initial scheduler state is PENDING. Do not restart the original job.

The saved-output checker `scripts/audit_cuda_qualification.py` requires successful Slurm accounting and verifies IDs, pinned model/runtime, exact resume tokens, context/batch counts and throughput arithmetic. Eight CPU tests cover valid and tampered evidence. This check is infrastructure consistency, not independent model inference or research efficacy.

## Tokenizer failure and verified alternate path

Job 27714091 completed the 8,060,896,487-byte snapshot download, then failed with SIGILL inside the native tokenizer's BPE deserializer on B200. It ran 1 minute 42 seconds and produced no inference result. Disabling Xet fixed the first failure; this trace identifies a different native component. Preserve the evidence under `runs/bouchet-cuda-qualification-v1b-retrieved`.

Before another submission, the pure-Python `Qwen2Tokenizer` was exercised on the login host (tokenization only, no inference) and matched six exact text and chat-template fixtures exported from the pinned local tokenizer, including Unicode and reasoning delimiters. Attempt **27714861** uses this checked Python tokenizer, retains disabled Xet and the same cached model/resources, and verifies those fixtures again inside the allocation before loading weights. It has its own immutable v1c directory and receipt at `runs/bouchet-cuda-qualification-v1c-control/submission.json`. No successful CUDA inference or research result is claimed yet. If this route encounters the same failure again, stop resubmitting that route and diagnose environment portability before further GPU allocations.


## Latest state: CUDA research-loop submission

Bouchet qualification **27714861 COMPLETED 0:0** on H200 in 3:35. Retrieved record audit passed (`results/bouchet_cuda_qualification_v1c_audit.json`): exact greedy resume, 8k context, batch generation; peak allocation 9.31 GiB, measured batch throughput 5.78 tokens/s. This establishes compatibility, not acceleration or scientific benefit.

Submitted next job **27717762** (actual submission), receipt `runs/bouchet-cuda-guard-runtime-v1-control/submission.json`, remote `runs/cuda-guard-runtime-v1-20260927`. One GPU, pi_fl426 normal tier, 4 CPUs, 32 GiB, 2 hours; fresh two-problem CUDA continue/segmented-sham/GUARD-adaptation qualification. No Jev credential. Frozen protocol `docs/cuda_guard_runtime_v1_protocol.md`. All 371 CPU tests pass. On completion retrieve complete records and run the independent guard qualification auditor before analysis/scaling. Local worker 57951 remains the separate frozen MLX study; do not pool BF16 and MLX results. Monitor this newest job, not completed or failed qualification attempts.


## September 27, 22:26 continuation

Job 27717762 COMPLETED 0:0, H200, 14:30 (0.2417 GPU-hours). Retrieved under `runs/bouchet-cuda-guard-runtime-v1-retrieved`; independent audit `results/cuda_guard_runtime_v1_audit.json` passed: two eligible tasks, six outcomes, 75 calls, two natural branch triggers, zero label disagreements. All three policies solved the same first task and failed the second; no benefit established. Initial retrieval requested nonexistent top-level requirements.lock, but the actual frozen requirements.lock.txt is preserved inside outputs/source; all audit-required records were retrieved and verified.

Next actual submitted job **27719471**, receipt `runs/bouchet-cuda-guard-development-v2-control/submission.json`, remote `runs/cuda-guard-development-v2-20260927`: twelve fresh development tasks/36 potential episodes, one GPU, four CPUs,32 GiB,four hours, pi_fl426 normal. Protocol `docs/cuda_guard_development_v2_protocol.md` frozen before generation. Monitor this job and local worker 57951 (original start identity verified). No pooling across backends. Audit terminal records before analysis; investigate timeouts and enrollment along with outcomes. 371 CPU tests passed.


## September 27, 23:26 completed audits and randomized baseline

Local14363 completed0 in455s; audit reconciled12outcomes/153calls/3triggers: all policies3/4. CUDA27719471 completed0 in23:56 (0.3989GPUh); audit reconciled36outcomes/430calls/3triggers: continue5/12,sham6/12,GUARD6/12. No advantage over sham. Reports `results/guard_runtime_v1_audit.json`, `results/cuda_guard_development_v2_audit.json` and corresponding summaries. Audit initially rejected harmless mean-entropy reduction roundoff; fixed by independently validating the stored mean then using it for exact runtime decision reconstruction. Tamper/roundoff tests pass. Original records unchanged. Missing AppleDouble frozen-source files were recovered byte-for-byte over SSH; all frozen hashes passed.

New actual job **27724738**, receipt `runs/bouchet-cuda-guard-random-v3-control/submission.json`, outputs remote `runs/cuda-guard-random-v3-20260927`. Twelve fresh tasks/four policies add random branching with probability3/55 per eligible boundary, frozen from completed v2 opportunities. Same budget, ranking, generation and finalization as GUARD. This matches opportunity rate in expectation, not realized costs/counts. Protocol `docs/cuda_guard_random_v3_protocol.md`. One standard GPU,4CPU,32GiB,4h. 375CPUtests passed. No live local model now; previous jobs terminal, never restart. Next: full audit and paired timing comparison after terminal, then choose controller study based on complete results.


## September28,00:26 randomized result and precision stage

Job27724738 completed0 in26:17 (0.4381GPUh). Retrieved `runs/bouchet-cuda-guard-random-v3-retrieved`; full audit passed48outcomes/567calls,zero disagreements. Audit field natural_triggers totals both policies:2GUARD and2random. GUARD/random/sham each8/12;continue10/12 with one call timeout. GUARD-minus-random0pp,descriptive paired interval−25 to+25pp,two discordant problems. Reports `results/cuda_guard_random_v3_{audit,analysis}.json`. No efficacy/equivalence.

Actual submitted job **27727778**, receipt `runs/bouchet-cuda-guard-precision-v4-control/submission.json`, remote `runs/cuda-guard-precision-v4-20260928`:96fresh tasks/384 potential policy episodes, unchanged actions,random3/55,primaryGUARD-vs-random. Protocol `docs/cuda_guard_precision_v4_protocol.md` frozen before generation. Planning halfwidth8.2pp under observed discordance,not3pp power. Measured linear cost3.50GPUh; bounded1GPU/4CPU/32GiB/8h,pi_fl426 normal. 377tests passed. No local model live. Audit and analyze on terminal; do not adapt from partial rates. This is substantive precision development,not another backend qualification or final test.


## September28,12:34 completed precision study and repair replication

27727778 COMPLETED0:0 in1:33:37 (1.5603GPUh). Retrieved `runs/bouchet-cuda-guard-precision-v4-retrieved`; audit passed96eligible/384outcomes/4583calls/zero label disagreements. Continue54/96,sham66/96,GUARD61/96,random59/96. GUARD-minus-random+2.08pp (descriptive−6.25 to+10.42);GUARD-minus-sham−5.21pp (−12.5 to+2.08). GUARD37branches,random14; not realized-rate matched. One continuation timeout. Reports `results/cuda_guard_precision_v4_{audit,analysis}.json`. No semantic efficacy, no equivalence, no positive branching claim. Stop scaling this exact timing policy unchanged.

Launched fresh repair replication **PID79898,start Mon Sep28 12:34:41 2026**, receipt `runs/action-replication-v3-control/process.json`, output `runs/action-replication-v3-20260928`. Config `action_replication_v3.json`,protocol `docs/action_replication_v3_protocol.md`:24freshproblems,4repeats,5qualifiedactions,up to480outcomes,4h local bound,no Jev. Primary segment-repair-vs-sham and disjoint-repeat action preference diagnostic. No cluster job remains live. 380tests passed before launch. Auditor extended to the new frozen design; never modify generation source in the run directory. Next terminal audit then action replicability, not noisy winner training.

## September 29, 11:46 EDT — InterWhen runner failure and versioned retry

The first InterWhen adaptation job **27857096** is terminal `FAILED`, exit `1:0`, after 00:02:46 on an H200 (one GPU, four CPUs, 32 GiB; batch MaxRSS 9,445,280 KiB). The Qwen checkpoint loaded; two of 192 episodes were written. A progress-print call passed `flush=True` to `json.dumps`, so the first progress report raised a `TypeError` and stopped the run. This is a code defect, not a scheduler, model-load, or VRAM failure. Allocation was 0.0461 GPU-hours. Its partial outcomes were excluded; the independent complete-run auditor rejected the incomplete matrix before any rates were interpreted.

Raw logs and partial artifacts with hashes are preserved at `runs/bouchet-interwhen-kstable-game24-v1-failed-27857096/`. Corrected runner, regression test and v1b protocol use a new source commit, submission receipt, and `runs/interwhen-kstable-game24-v1b-${SLURM_JOB_ID}` output path. Only logging changed; model, task sample, monitor, continuation, generation, and exact evaluator remain frozen. Targeted tests 7/7 and full suite 407/407 passed; `bash -n`, `py_compile`, and diff whitespace checks passed. Before submitting v1b, pin the new checkout in a distinct immutable remote path, inspect queue and project quota, run `sbatch --test-only` for this single resource shape, and submit once. Charge both failed and successful allocations to the total. If the same failure repeats, stop; no third retry.


## September 29, 12:20 EDT — InterWhen v1b active on B200

Cause-specific retry job **27858691** was actually submitted and has started. Its exact receipt is `runs/bouchet-interwhen-kstable-game24-v1b-control/submission.json`; source commit `269b5e8ec1384cc870430f76e0b781a59dce9e20` is staged under `/nfs/roberts/project/pi_fl426/yz2324/JEV-CoT-Reward-kstable-269b5e8`, separate from the failed v1 checkout. Remote SHA-256 hashes matched all twelve local source/config/data files. The model snapshot was already cached, so no transfer or new download occurred. The job runs on B200 node `a1116u05n01`, `pi_fl426` standard tier, one GPU, four CPUs, 32 GiB, six hours; no Jev calls.

At 12:30 EDT, fresh `sacct` showed RUNNING with 35:30 elapsed, one B200 GPU allocated (0.5917 GPU-hours), four CPUs, and 32 GiB; `scontrol` agreed on node `a1116u05n01`, account `pi_fl426`, and immutable `WorkDir`. The site Slurm build rejected a formatted `squeue -o` query, so state verification relied on successful `sacct` and `scontrol` records instead. No outcomes have been analyzed before completion. Next action is to allow this immutable job to finish; retrieve and audit complete records and actual resource usage before interpreting outcomes. The same logger defect has now had its one allowed retry; no third attempt for that cause.
