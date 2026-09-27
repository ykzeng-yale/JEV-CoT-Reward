# Observed compute and a local-first execution plan

Inventory collected 2026-09-27. See `results/resources.json` for the machine-readable observation and `scripts/collect_resources.py` for a repeatable, read-only probe. This is a host-specific snapshot, not a guaranteed resource reservation.

## Actual available host

| Resource | Observation |
|---|---|
| Computer | MacBook Pro, Apple M5 |
| CPU / GPU | 10 logical CPU cores; 10-core Apple GPU; Metal available |
| Unified memory | 32 GiB |
| Disk free at final inventory | 51.6 GiB; APFS usage changes with snapshots/caches |
| OS | macOS 26.5.2, arm64 |
| Memory pressure context | About 15 GiB swap was already in use before this study |
| NVIDIA GPU | None found on this host |
| Existing tooling | `uv`; Python 3.12/3.13 managed installations; reusable MLX runtime |

Do not reserve the machine's full 32 GiB for inference. For the first experiments, use one quantized 4B model process and sequential arms. Existing applications and work were left running. No packages or models were downloaded for the audit.

The initial Mac mini SSH probe returned **Connection refused**. Later on 2026-09-27, another authorized Codex workflow completed Remote Login and public-key enrollment. This project then independently verified noninteractive SSH commands and an SFTP download of the server's public host key through the existing `mac-mini` alias, with strict host-key checking. The downloaded public key matched the trusted pin. Private keys remained on their originating machines; no service or experiment was changed by this verification.

The mini reports **Apple M4, 10 CPU cores, 16 GiB unified memory, macOS 15.3.1**, and approximately **14.8 GiB disk free**. `memory_pressure -Q` reported 73% system-wide memory free and swap usage was 233.31 MiB; these are transient observations, not a reservation. Ollama processes are present, but installed model weights and inference throughput were not checked in this connectivity test. A bounded quantized-model workload remains subject to fresh resource checks and coordination with existing work. The reusable connection is documented in the owner's `mac-ssh-compute` skill; its private network inventory is kept outside this repository.

## Reusable local models and runtimes

Model weight sizes below are bytes on disk, not peak inference memory. Cache snapshots contain public open-weight models. `results/resources.json` records public model identities, revisions and weight-file sizes; machine-specific cache paths are omitted.

| Model snapshot | Weight size | Best immediate role |
|---|---:|---|
| Qwen/Qwen3-4B-Instruct-2507, `cdbee75f17c01a7cc42f958dc650907174af0554` | 7.49 GiB BF16 | Primary instrumented generator; quantize to 4 bits in memory |
| unsloth/Qwen3-4B-Instruct-2507-GGUF, `a06e946bb6b655725eafa393f4a9745d460374c9` | 2.33 GiB Q4_K_M | Alternative llama.cpp route if needed |
| Qwen/Qwen2.5-3B-Instruct-GGUF | 1.96 GiB Q4_K_M | Small second generator, if a GGUF runtime is introduced |
| Qwen/Qwen2.5-7B-Instruct-GGUF | 4.36 GiB across two shards | Later model-size comparison |
| mlx-community/Qwen2.5-Coder-7B-Instruct-4bit | 3.99 GiB | Already quantized MLX coding extension |

Use the local Hugging Face snapshot for the pinned primary revision above. Set `MODEL_SNAPSHOT` to its absolute directory when running experiments; the harness refuses a remote model identifier and does not download weights.

Verified project runtime:

```text
.venv/bin/python (Python 3.12.13)
mlx==0.32.2
mlx-lm==0.31.3
```

The initial feasibility probe reused an existing local MLX environment. The project now has its own `.venv` with MLX 0.32.2 and mlx-lm 0.31.3, pinned in `requirements.lock.txt`; the real paired experiments ran through that project environment. Its installed footprint is approximately 464 MiB, mostly reusable cached packages. Other projects' filesystem locations are not needed to reproduce this setup.

The available Qwen3 checkpoint is the **Instruct-2507** variant. A study using it measures interventions on its observable scratchpad/answer prefix. It must not be described as an experiment on hidden reasoning or silently equated with the original Qwen3 thinking model. Quantization changes the generator and is held fixed across study arms.

## Actual local generation checks

The prompt requested a short calculation of `17 * 23` and a `FINAL:` integer. Both runs returned `391`. These are instrumentation checks, **not research evidence that Jev improves reasoning**.

| Configuration | Emitted tokens, including EOS | Generation wall time | Tokens / wall second | MLX process peak |
|---|---:|---:|---:|---:|
| Cached BF16 | 45 | 4.160 s | 10.82 | 8.105 GB |
| In-memory 4-bit affine, group size 64 | 117 | 2.716 s | 43.08 | 3.814 GB |

Load took about 3.25 s for BF16 and 2.05 s for in-memory quantization. The different output lengths reflect different quantized-model generations; these measurements are not a controlled speed benchmark. Short prompts overstate throughput attainable with repeated long-context prefills. Long-context throughput must be measured before committing to a large run.

Both configurations passed a greedy sham check: the token sequence obtained by generating 16 tokens, pausing, and resuming from the **same original prompt IDs plus those exact 16 IDs** matched uninterrupted generation. This checks one short deterministic case; it does not establish universal bitwise equivalence, stochastic RNG continuation equivalence, or equivalence after changing instructions.

Evidence:

- `results/resource_smoke.json`: BF16 run and sham check.
- `results/resource_smoke_quantized.json`: quantized run and sham check.
- `src/jev_control/mlx_backend.py`: exact-token generation, fixed sampling, token caps, cooperative timeout, and a gold-free likelihood selector.

The backend casts logits to float32 before normalization. The installed mlx-lm otherwise normalizes BF16 logits at their original precision, which rounded the selected-token log probabilities to zero in the initial probe. Entropy and mean log probability are computed from the untempered model distribution; these are not a calibrated correctness score. Branch selection by mean token log probability is intentionally a simple instrumentation baseline and can favor fluent but incorrect text.

Timeouts are checked between emitted tokens. A prefill or Metal kernel already running is not interrupted. MLX can compute one look-ahead token; emitted-token accounting includes EOS but does not count an unconsumed internal look-ahead as emitted output. All wall time should still be recorded. For a hard wall-clock cap, wrap the experiment process in an outer supervisor.

## What to allocate now

1. **Use existing compute:** a single 4-bit Qwen3-4B process, one active sequence, explicit visible reasoning, and exact-token checkpoint records. No new GPU is required for feasibility work.
2. **Use the frozen development-tested schedule:** the current mechanism screen allows 1,024 generated tokens per hypothetical episode and one online checkpoint at the first newline after 256 tokens, capped at 384. Earlier 512-token instrumentation is retained separately. Difficulty changes require fresh development data and a new protocol.
3. **Retain paired arms:** continue/repair/branch use the same cached checkpoint and remaining generation budget. Count discarded candidates and repair generations. Any added instruction tokens increase prefill cost and must be recorded.
4. **Reserve disk:** the current 51.6 GiB can accommodate code, dependency setup, and a small text/ID-based pilot. Do not duplicate model weights or save full logits/KV caches. Additional storage or cleanup becomes useful only after a justified expansion; this audit deletes nothing.
5. **Scale on evidence:** at the observed short-prompt rate, 1 million generated tokens would take roughly 6.4 device-hours before additional long-prefill/verification costs. This is a rough extrapolation, not a throughput promise. A 26-million-token pilot would be about a week on this device even at that rate, so first run a materially smaller decision experiment.

The user's **$25 total Jev limit** is separate from local compute. Local generation and controller fitting need no paid model API. Price, batching, latency, retry reserve, and actual billed input-token usage should be handled by the Jev budget ledger; do not extrapolate that a hosted judge is free because local inference dominates wall time.

The [later pool qualification](mac_compute.md) verifies SSH/SFTP and bounded CPU execution on both peers. It records their runtime/model availability and transient resources. Neither peer has yet passed this generator's inference qualification; before moving a justified longer model experiment there, refresh available storage and concurrent workloads, then measure the actual inference envelope. Larger confirmatory experiments may benefit from a 48–80 GB NVIDIA worker, but the present host is sufficient to test instrumentation and an initial signal without purchasing hardware.

## Repeat the audit

```sh
.venv/bin/python scripts/collect_resources.py \
  --python "$(pwd)/.venv/bin/python" \
  --output runs/resources-new.json
```

Use `--ssh-host mac-mini` for the verified remote route. The alias selects the enrolled client identity and pinned host-key file. The script uses batch mode and strict host-key checking; it will not add trust entries or request passwords. A direct hostname does not necessarily select those alias-specific settings.
