# InterWhen K-stable Game24 adaptation v1b retry protocol

## Purpose and scope

This is a cause-specific retry of the previously frozen 96-task, two-arm public-train development adaptation in `docs/interwhen_kstable_game24_v1_protocol.md`. It is not a new hypothesis, additional arm, Jev study, confirmatory evaluation, or faithful reproduction of InterWhen's original larger-model result. The sole implementation repair is to serialize the progress JSON separately and pass `flush=True` to `print`; the prior runner incorrectly passed it to `json.dumps` and exited after the first paired task.

## Preserved failed attempt

Job `27857096` terminated FAILED, exit `1:0`, after 00:02:46 on one H200. `sacct` reports one H200 GPU, four CPUs, 32 GiB RAM, and batch MaxRSS 9,445,280 KiB. The pinned Qwen3-4B BF16 checkpoint loaded and the runner wrote two episodes (one task pair) before the progress-log `TypeError`. The manifest records 2 completed episodes of 192 and status `failed`. The complete outcome auditor refused the partial run because it was incomplete; no policy success rates were calculated or interpreted.

Preserved locally under `runs/bouchet-interwhen-kstable-game24-v1-failed-27857096/`: raw Slurm `.out/.err`, partial run manifest/summary/episodes/calls, and `failure_receipt.json`. These files are immutable failure evidence. The partial episodes are excluded from the retry's outcomes; their compute is included in cumulative experiment cost.

## Retry delta and unchanged scientific contract

- Same 96 frozen public `nlile/24-game` train tasks and exact task SHA from v1.
- Same two policies (`continue`, `interwhen_kstable_k2`, k=2), model/tokenizer revision, BF16, generation parameters, seeds, independent AST/Fraction terminal evaluator, and cost fields.
- Same one-GPU normal-tier allocation class and cached model; no weight/data download or Jev call.
- New source commit and immutable output path `runs/interwhen-kstable-game24-v1b-${SLURM_JOB_ID}`; no overwrite/resume of job 27857096 records.
- The partial failure cost (0.0461 allocated GPU-hours) remains in total resource accounting; v1b records its own full allocation and outcomes.

The progress-output regression test and full test suite must pass before a single retry submission. The new job must use a source checkout at the committed v1b hash and its own submission receipt. If it encounters this same logging failure again, do not submit a third job; preserve evidence and stop this adaptation until redesigned.

## Analysis gate

Do not inspect v1b outcome rates until it is terminal, all 192 episodes and call ledgers are retrieved, checksums and Slurm accounting reconcile, and `scripts/audit_interwhen_kstable_adaptation.py` passes. If it fails or is incomplete, retain all artifacts and do not pool partial v1 or v1b outcomes. On full audit success, report paired outcomes, independent exact checker agreement, stop frequency, token/prefill/latency costs, timeouts, and the sum of GPU allocation for the failed attempt plus completed retry. Conclusions remain limited to this baseline adaptation on this public development sample.
