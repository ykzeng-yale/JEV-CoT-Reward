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
