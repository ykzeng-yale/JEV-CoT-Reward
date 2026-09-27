#!/usr/bin/env python3
"""Fresh development-only framing and online pause/resume diagnostic.

The sham resumes exact emitted IDs with a fresh, recorded random seed. This is
a distributional implementation comparison, not common-random-number replay.
Both conditions share the same final-answer reserve protocol.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_control.mlx_backend import MLXBackend
from jev_control.tasks import make_task, verify
from run_screen import FINAL

PROMPT_SUFFIX = (
    "\nWork concisely: start with an actual calculation or candidate, not a "
    "restatement of the input. Use short lines for calculations and checks. "
    "Keep intermediate work under 600 words. Follow the specified FINAL: format."
)


def boundary_now(backend, token_ids, minimum):
    if len(token_ids) < minimum:
        return False
    text = backend.decode(token_ids)
    return text.endswith("\n") and "FINAL:" not in text


def calculation_marker(text):
    """Surface diagnostic only: neither truth nor semantic-quality labeling."""
    return bool(re.search(r"\d\s*[+*/−-]\s*\d|(?:distance|cost|dist|total)\s*[:=(]|\d\s*=\s*\d", text, re.I))


def episode(backend, task, prompt, condition, seed, budget=1024,
            final_reserve=96, checkpoint_target=256, checkpoint_cap=384):
    if condition not in {"baseline", "sham"}:
        raise ValueError("Unknown condition")
    if not 0 < checkpoint_target <= checkpoint_cap < budget-final_reserve or final_reserve < 0:
        raise ValueError("Invalid development token envelope")
    calls = []; tokens = []; overhead = []; checkpoint = None
    first_cap = budget-final_reserve if condition == "baseline" else checkpoint_cap
    kwargs = {}
    if condition == "sham":
        kwargs["stop_when"] = lambda ids: boundary_now(backend, ids, checkpoint_target)
    initial = backend.generate(prompt, first_cap, seed, timeout_s=180, **kwargs)
    calls.append(initial); tokens.extend(initial.token_ids)
    if condition == "sham":
        if initial.finish_reason == "checkpoint":
            checkpoint = {
                "token_ids": list(tokens), "text": backend.decode(tokens),
                "prefix_sha256": hashlib.sha256(json.dumps(prompt+tokens, separators=(",", ":")).encode()).hexdigest(),
                "position": len(tokens), "calculation_marker": calculation_marker(backend.decode(tokens)),
            }
        remaining_reasoning = budget-final_reserve-sum(g.generated_tokens for g in calls)
        if initial.finish_reason not in {"stop", "timeout"} and remaining_reasoning:
            continued = backend.generate(prompt+tokens, remaining_reasoning, seed+1, timeout_s=180)
            calls.append(continued); tokens.extend(continued.token_ids)
    if calls[-1].finish_reason not in {"stop", "timeout"}:
        remaining = budget-sum(g.generated_tokens for g in calls)
        if remaining:
            if "FINAL:" not in backend.decode(tokens):
                injection = backend.encode_text(FINAL)
                tokens.extend(injection)
                overhead.append({"kind": "finalization_instruction", "tokens": len(injection)})
            final = backend.generate(prompt+tokens, remaining, seed+2, timeout_s=180)
            calls.append(final); tokens.extend(final.token_ids)
    spent = sum(g.generated_tokens for g in calls)
    assert spent <= budget
    text = backend.decode(tokens)
    return {
        "problem_id": task["id"], "family": task["family"], "condition": condition, "seed": seed,
        "text": text, "outcome": verify(task, text), "calls": [g.to_dict() for g in calls],
        "generated_tokens": spent, "elapsed_seconds": sum(g.elapsed_seconds for g in calls),
        "prompt_tokens_processed": sum(g.prompt_tokens for g in calls), "overhead": overhead,
        "checkpoint": checkpoint,
        "checkpoint_status": ("not_requested" if condition == "baseline" else
                              "eligible" if checkpoint else
                              "completed_early" if initial.finish_reason == "stop" else
                              "timeout" if initial.finish_reason == "timeout" else
                              "answer_phase_before_checkpoint" if "FINAL:" in initial.text else "no_boundary_before_cap"),
    }


def summarize(rows):
    conditions = {}
    for condition in ("baseline", "sham"):
        selected = [r for r in rows if r["condition"] == condition]
        conditions[condition] = {
            "n": len(selected), "successes": sum(r["outcome"]["success"] for r in selected),
            "generated_tokens": sum(r["generated_tokens"] for r in selected),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in selected),
            "checkpoint_status": dict(Counter(r["checkpoint_status"] for r in selected)),
            "calculation_marker_checkpoints": sum(bool(r["checkpoint"] and r["checkpoint"]["calculation_marker"]) for r in selected),
            "families": {f: {"n": sum(r["family"]==f for r in selected),
                              "successes": sum(r["outcome"]["success"] for r in selected if r["family"]==f)}
                         for f in sorted({r["family"] for r in selected})},
        }
    return {"kind": "development_framing_and_online_sham_not_jev_effect", "episodes": len(rows),
            "independent_problems": len({r["problem_id"] for r in rows}), "conditions": conditions,
            "warning": "Small development sample; distinct resume RNG seeds. Not an equivalence test or confirmatory policy evaluation."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--problems", type=int, default=12)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--task-seed", type=int, default=391027)
    p.add_argument("--sampling-seed", type=int, default=271027)
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--checkpoint-target", type=int, default=256)
    p.add_argument("--checkpoint-cap", type=int, default=384)
    p.add_argument("--final-reserve", type=int, default=96)
    args = p.parse_args()
    if not 1 <= args.problems <= 24 or not 1 <= args.repeats <= 4:
        p.error("Development bounds: 1–24 problems, 1–4 repeats")
    if not 0 < args.checkpoint_target <= args.checkpoint_cap < args.budget-args.final_reserve or args.final_reserve < 0:
        p.error("Invalid token envelope")
    if args.output.exists():
        p.error("Use a new output directory; partial runs are preserved, never silently rerun")
    args.output.mkdir(parents=True)
    source = args.output / "source"; source.mkdir()
    hashes = {}
    for path in [Path(__file__), ROOT/"scripts/run_screen.py", *sorted((ROOT/"src/jev_control").glob("*.py"))]:
        data = path.read_bytes(); (source/path.name).write_bytes(data)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    manifest = {**vars(args), "output": str(args.output), "started_unix": time.time(),
                "protocol": "development-v1-online-boundary", "prompt_suffix": PROMPT_SUFFIX,
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                "temperature": .7, "top_p": .9, "source_sha256": hashes,
                "packages": {x: importlib.metadata.version(x) for x in ["mlx", "mlx-lm", "numpy", "transformers"]},
                "finalization": "Same reserve intervention in both conditions; baseline uninterrupted until final reserve.",
                "resume_rng": "Fresh recorded seed on resume; no common-random-number or bitwise stochastic claim."}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    backend = MLXBackend(args.model, quantization_bits=4)
    rows = []
    for index in range(args.problems):
        task = make_task(index, args.task_seed)
        task["prompt"] += PROMPT_SUFFIX
        prompt = backend.encode_chat([{"role": "user", "content": task["prompt"]}])
        with (args.output/"tasks.jsonl").open("a") as f:
            f.write(json.dumps({"task": task, "prompt_ids": prompt})+"\n")
        schedule = [(c,r) for r in range(args.repeats) for c in ("baseline", "sham")]
        random.Random(args.sampling_seed+index).shuffle(schedule)
        for condition, repeat in schedule:
            seed = args.sampling_seed+index*1000+repeat*10
            row = episode(backend, task, prompt, condition, seed, args.budget, args.final_reserve,
                          args.checkpoint_target, args.checkpoint_cap)
            row["repeat"] = repeat; rows.append(row)
            with (args.output/"outcomes.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps({"problem": index, "condition": condition, "repeat": repeat,
                              "success": row["outcome"]["success"], "tokens": row["generated_tokens"],
                              "checkpoint": row["checkpoint_status"]}), flush=True)
    summary = summarize(rows)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
