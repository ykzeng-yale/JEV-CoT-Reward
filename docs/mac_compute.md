# Mac compute pool qualification

The first remote JEV workload, **bounded standard-library theory validation**, has now completed on both the Mac mini and auxiliary MacBook. Keep instrumented Qwen3 generation on the lead MacBook: its pinned model, project environment and exact-token backend have already been exercised. Neither peer currently has the same generator environment verified.

This read-only qualification was collected on **2026-09-27, 19:42–19:45 UTC**, using the owner's `mac-ssh-compute` skill. Machine-readable observations are in [mac_pool_qualification.json](../results/mac_pool_qualification.json). The private registry, account names, addresses and SSH keys remain outside the repository.

## Verified transport

| Direction | Noninteractive SSH | Read-only SFTP |
|---|---|---|
| Lead → mini | Passed | Passed |
| Lead → auxiliary | Passed | Passed |
| Mini → auxiliary | Passed | Passed |
| Auxiliary → mini | Passed | Passed |
| Mini / auxiliary → lead | Pending enrollment | Pending enrollment |

Each verified SSH check returned `Darwin` with exit status zero. SFTP read the public operating-system version plist into `/dev/null`. Existing strict host-key checking, batch mode, origin-specific identities and pinned trust were used. No agent forwarding was used. The peers' lead alias is not enrolled and the registry has no lead host-key pin; its listener was not assessed. The lead can submit work and pull results without accepting inbound SSH.

## Current resources

| Host | CPU / GPU | RAM | Disk free | Swap used | `memory_pressure -Q` free | Power |
|---|---|---:|---:|---:|---:|---|
| Lead MacBook | M5, 10 / 10 cores | 32 GiB | 51.96 GiB | 14.92 GiB | 59% | AC, charged |
| Mac mini | M4, 10 / 10 cores | 16 GiB | 14.75 GiB | 0.22 GiB | 75% | AC |
| Auxiliary MacBook | M2 Max, 12 / 30 cores | 32 GiB | 17.57 GiB | 8.70 GiB | 46% | AC attached, 80% |

These are snapshots, not resource reservations. The free percentage is not an allocation budget. Existing swap does not establish current paging activity. Process RSS can double-count shared pages and does not capture all GPU or virtual-machine allocations.

The mini had Codex, Chrome and Ollama processes. The auxiliary had substantial Codex and Chrome activity plus Docker processes; its load averages were approximately 5.8, 7.1 and 7.3. No Ollama process was observed there during this probe. The lead also had active desktop and virtual-machine tooling. Only process basenames and aggregate RSS were retained; private command arguments and experiment outputs were not inspected.

## Existing runtimes and model metadata

| Host | Inspected runtime | Relevant cached models | Implication |
|---|---|---|---|
| Lead | Project Python 3.12.13, MLX 0.32.2, mlx-lm 0.31.3; uv 0.11.17 | Primary Qwen3-4B-Instruct-2507 snapshot plus existing GGUF/MLX alternatives | Continue current instrumented local work |
| Mini | Managed Python 3.12.12; uv 0.9.30; Ollama CLI 0.34.0 | Ollama Kimina-Prover-Distill-1.7B, about 3.79 GiB | Standard-library validation can run without installing dependencies; current model is a different task-specific generator |
| Auxiliary | Python 3.12.10; Homebrew Python 3.14.4 with MLX 0.31.2; uv 0.11.7; Ollama CLI 0.21.2 | Ollama Qwen2.5 0.5B / 1.5B, about 0.37 / 0.92 GiB | Candidate for a separately qualified small-model comparison |

No `mlx-lm` package was found in the inspected peer interpreters. This is a bounded search of known runtime locations, not a claim about every project environment. Other cached models were unrelated embedding or task-specific models and were not selected.

The primary lead snapshot is `Qwen/Qwen3-4B-Instruct-2507` revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Previous load, generation and checkpoint checks are described in [resources.md](resources.md). This audit did not repeat inference.

Ollama manifests and all their referenced blob sizes matched on disk, but content digests and execution were not verified. A cache entry or CLI version does not establish model usability, server compatibility, speed, accuracy or exact-token checkpoint support. Replacing the project backend with an Ollama model would change the generator and may change the experiment mechanics; treat that as a separate qualified condition.

## First bounded remote workload

The frozen `scripts/validate_theory.py` was copied to unique task directories and run sequentially using existing Python 3.12 interpreters. [remote_theory_check.py](../scripts/remote_theory_check.py) pins both validator and supervisor SHA-256 values, verifies the uploaded bytes, makes both inputs and their directory read-only, records actual worker PID/process-group/start identity, and retrieves the result over strictly verified SSH/SCP. It refuses to overwrite an existing local result.

| Host | Executed at UTC | Python | Worker PID | Checks | Exit / timeout |
|---|---|---|---:|---:|---|
| Mini | 2026-09-27 19:50:08 | 3.12.12 | 52451 | 14 / 14 | 0 / no |
| Auxiliary | 2026-09-27 19:50:24 | 3.12.10 | 51971 | 14 / 14 | 0 / no |

Results are saved in [mac_mini_theory_validation.json](../results/mac_mini_theory_validation.json) and [mac_aux_theory_validation.json](../results/mac_aux_theory_validation.json). Both full validation objects exactly matched a fresh local run of the same frozen source. This establishes transport, file-write/job/result-retrieval access and cross-host numerical replication of finite examples; it is not an independent mathematical proof or evidence of Jev effectiveness.

Executed validator SHA-256: `7bf71ceaaf622811066d1d0c18ffd7a91a95fba1487f59a11aa427378303c095`.

Executed supervisor SHA-256: `ecb918e6e07f851bba1de7515a60697dd4055196dad316c05090f55853f2a51e`.

Each worker had a 60-second CPU limit, a 120-second wall timeout and a 16 MiB per-file limit. The observed supervisor-to-completion intervals were about 0.09 seconds on the mini and 0.13 seconds on the auxiliary. **No hard RSS limit was enforced.** Each was one trusted, finite, standard-library worker; no model, API or service was involved. At least 10 GiB free disk was required before task creation. The tiny frozen inputs and private execution logs remain on each host for audit, owned by this project's lead workflow for any later cleanup.

The auxiliary's refreshed pre-run memory query still reported 46% system-wide free; swap used had increased to approximately 9.27 GiB. This admitted only the tiny CPU check, not a model workload.

For a future larger CPU task, the initial proposed envelope is one process, at most two CPU threads, 2 GiB process memory, a ten-minute wall limit, at most 512 MiB task storage and 50 MiB output. Keep at least 10 GiB disk free. The current runner does not enforce all these proposed limits and must be extended before being used for a workload that needs them. A supervisor must enforce the chosen bounds, record the actual PID and process start identity, and stop only its own job if a limit is crossed. A platform resource limit that does not cover physical or GPU memory is not a substitute for observation.

Keep frozen inputs separate from outputs; record script/input digests, runtime version, start/end times, return status and logs. Pull results back from the lead, compare against the same frozen local validation, and retain the immutable evidence. State who owns cleanup. Do not alter peer workloads to create capacity. No live scientific experiment should move to this route merely because the connectivity check passed.

Later auxiliary inference should begin only after a separate pinned-model/runtime smoke qualification and a fresh resource check. A conservative first small-model trial would use one model process, context at most 2,048 tokens, at most 128 generated tokens, a two-minute wall limit, 4 GiB process-memory ceiling and small text outputs. No Ollama service was started.

Paid Jev requests should remain on the lead with the project's single spending ledger. Remote workers can return local-only records for centralized judging; copying a ledger to independent spending workers would defeat the cumulative cap. Keep any future model endpoint on loopback.

## Audit boundaries

The initial resource audit installed, downloaded and copied no packages or models and launched no inference or experiment job. The separately authorized follow-up copied only the tiny validator and supervisor and ran the finite checks described above. No service or SSH trust was changed, and no private keys, credentials, environment values, DTR outputs or private command arguments were read. The resource snapshot JSON remains an account of the initial read-only audit; the two execution receipts record the later work. Transport success establishes a usable connection, not scientific evidence or reserved compute capacity.
