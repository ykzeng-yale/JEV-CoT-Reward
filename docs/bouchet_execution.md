# Bouchet execution track

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
