#!/usr/bin/env python3
"""Hardware-only long-context stress using an already loaded pinned ReCoMA model.

The ordinary controller's stop strings/EOS are deliberately disabled to charge
the full completion envelope. This is not a task trajectory or scientific
outcome and must be kept out of episode-rate denominators.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SYNTHETIC_TEXT = (
    "Hardware context qualification only. The repeated text below represents "
    "synthetic visible observations and action history, without any task, "
    "private label, terminal endpoint or scientific outcome. "
    "Describe a hypothetical next action and its visible evidence."
)
FILLER = " Observation: visible room, inventory, instrument reading and action history."


def exact_context_ids(base_ids: list[int], filler_ids: list[int], target_tokens: int) -> list[int]:
    """Preserve a chat-template prefix/suffix and pad its middle to exact length.

    This synthetic sequence is a hardware envelope, not claimed to be a natural
    controller prompt or a tokenizer round-trip of a scientific observation.
    """
    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool) or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    if not base_ids or not filler_ids or any(type(x) is not int or x < 0 for x in base_ids + filler_ids):
        raise ValueError("nonempty nonnegative integer token sequences required")
    if len(base_ids) > target_tokens:
        raise ValueError("base prompt exceeds target; refuse to silently truncate")
    missing = target_tokens - len(base_ids)
    # Retain the complete final chat-template/open-assistant boundary. The
    # arbitrary insertion point is disclosed and never used as a task prompt.
    split = max(0, len(base_ids) - min(32, len(base_ids)))
    padding = (filler_ids * math.ceil(missing / len(filler_ids)))[:missing]
    return base_ids[:split] + padding + base_ids[split:]


def _ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _save(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_context_stress(generator, output_path, target_tokens=32768, completion_tokens=400):
    """Run one separately logged stress call; reuse model/tokenizer, load nothing.

    The returned report contains all prefill/output token IDs, synchronized
    latency and peak GPU memory. It never calls generator.generate(), modifies
    its task counters/usage file, constructs an environment or reads outcomes.
    A failure leaves a durable report and propagates to block scaling.
    """
    path = Path(output_path)
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise FileExistsError("context stress refuses an existing record")
    if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool) or completion_tokens <= 0:
        raise ValueError("completion_tokens must be a positive integer")
    if target_tokens > 65536 or completion_tokens > 400:
        raise ValueError("stress exceeds the separately bounded context/completion envelope")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "starting", "scope": "synthetic_hardware_only_not_scientific_outcome",
        "scientific_episode": False, "hosted_calls": 0, "jev_calls": 0,
        "model_calls": 0, "model": getattr(generator, "model_id", None),
        "model_revision": getattr(generator, "model_revision", None),
        "target_input_tokens": target_tokens, "requested_output_tokens": completion_tokens,
        "max_new_tokens": completion_tokens, "min_new_tokens": completion_tokens, "do_sample": False,
        "generated_tokens": 0, "processed_prompt_tokens": 0,
        "input_tokens": 0, "output_tokens": 0,
        "ordinary_eos_and_stop_strings_disabled": True,
        "model_reload": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "global_source_prompt_bound_proven": False,
        "source_bound_note": "10000-character history limit does not bound injected dynamic observation/dialog/action fields.",
    }
    started = time.monotonic()
    cuda_ready = False
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("stress requires exactly one already allocated visible CUDA GPU")
        if record["model"] != MODEL or record["model_revision"] != REVISION:
            raise RuntimeError("stress model identity is not the pinned Qwen snapshot")
        model = generator.model
        if model.dtype != torch.bfloat16 or model.config._attn_implementation != "sdpa":
            raise RuntimeError("stress requires the same BF16/SDPA backend")
        limit = getattr(model.config, "max_position_embeddings", None)
        if not isinstance(limit, int) or target_tokens + completion_tokens > limit:
            raise RuntimeError("stress context plus completion exceeds the model context contract")
        tokenizer = generator.tokenizer
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": SYNTHETIC_TEXT}],
            tokenize=True, add_generation_prompt=True,
        )
        if not isinstance(base, list):
            raise TypeError("synthetic chat template must return a token-ID list")
        filler = tokenizer.encode(FILLER, add_special_tokens=False)
        ids = exact_context_ids(base, filler, target_tokens)
        record.update(input_tokens=len(ids), input_token_ids=ids,
                      input_token_ids_sha256=_ids_digest(ids),
                      base_chat_tokens=len(base), synthetic_padding_tokens=len(ids)-len(base),
                      cuda_device=torch.cuda.get_device_name(0), attention_implementation="sdpa",
                      precision="bfloat16", model_context_limit=limit)
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        mask = torch.ones_like(input_ids)
        # Clone rather than alter the controller's generation configuration.
        config = copy.deepcopy(model.generation_config)
        config.eos_token_id = None
        config.forced_eos_token_id = None
        config.stop_strings = None
        config.do_sample = False
        config.pad_token_id = tokenizer.pad_token_id
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cuda_ready = True
        generation_started = time.monotonic()
        record.update(model_calls=1, processed_prompt_tokens=len(ids))
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids, attention_mask=mask, generation_config=config,
                max_new_tokens=completion_tokens, min_new_tokens=completion_tokens,
                do_sample=False, use_cache=True,
            )
        torch.cuda.synchronize()
        duration = time.monotonic() - generation_started
        output_ids = generated[0, len(ids):].tolist()
        record.update(generated_tokens=len(output_ids), output_tokens=len(output_ids),
                      output_token_ids=output_ids,
                      output_token_ids_sha256=_ids_digest(output_ids),
                      generation_elapsed_seconds=duration,
                      elapsed_seconds=duration,
                      output_text=tokenizer.decode(output_ids, skip_special_tokens=True),
                      full_completion_envelope_reached=len(output_ids) == completion_tokens,
                      peak_allocated_gpu_bytes=int(torch.cuda.max_memory_allocated()),
                      peak_reserved_gpu_bytes=int(torch.cuda.max_memory_reserved()))
        if len(output_ids) != completion_tokens or duration <= 0 or not math.isfinite(duration):
            raise RuntimeError("stress did not complete the exact charged token/time envelope")
        record["status"] = "completed"
        return record
    except BaseException as exc:
        record.update(status="infrastructure_failure", exception_type=type(exc).__name__,
                      exception_message=str(exc),
                      failed_generation_tokens_unknown=record["model_calls"] == 1 and record["generated_tokens"] == 0)
        if cuda_ready:
            record.update(peak_allocated_gpu_bytes=int(torch.cuda.max_memory_allocated()),
                          peak_reserved_gpu_bytes=int(torch.cuda.max_memory_reserved()))
        raise
    finally:
        record["total_elapsed_seconds"] = time.monotonic() - started
        _save(path, record)
