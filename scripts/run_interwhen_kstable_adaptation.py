#!/usr/bin/env python3
"""Evaluate a pinned InterWhen K-stable monitor adaptation on frozen Game24 tasks."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_control.cuda_backend import CUDABackend
from jev_control.game24_exact import InvalidExpression, _integer_leaves, check_game24
from jev_control.prospective import durable_record
from audit_interwhen_kstable_taskset import audit as audit_taskset


CONFIG = ROOT / "configs/interwhen_kstable_game24_v1.json"
TASKS = ROOT / "data/interwhen_kstable_game24_v1/tasks.json"
PROMPT_TEMPLATE = """You are solving the Game of 24.

You are given four numbers: {numbers}

Produce a valid arithmetic expression using all four numbers exactly once,
only +, -, *, /, and evaluating exactly to 24. Reason step by step, and put
your final answer containing only the expression within \\boxed{{}}."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pinned_model_identity(model_path: Path) -> dict:
    model_path = model_path.resolve()
    revision = json.loads(CONFIG.read_text())["model"]["revision"]
    if model_path.name != revision:
        raise ValueError(f"Model snapshot directory must be the pinned revision {revision}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Pinned model config missing: {config_path}")
    config = json.loads(config_path.read_text())
    weights = sorted(p for p in model_path.iterdir() if p.name.endswith((".safetensors", ".bin")))
    if not weights:
        raise FileNotFoundError("Pinned model snapshot has no local weight files")
    hashes = {p.name: sha256_file(p) for p in weights}
    return {"path": str(model_path), "config_commit_hash": config.get("_commit_hash"),
            "weight_files_sha256": hashes,
            "snapshot_revision": model_path.name,
            "weight_revision_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def normalize_equation(text: str) -> str:
    for old, new in ((r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"),
                     ("×", "*"), ("÷", "/"), ("−", "-"), ("–", "-"), ("—", "-")):
        text = text.replace(old, new)
    text = text.replace(r"\,", "").replace(r"\ ", "")
    return re.sub(r"\s+", "", text).lower()


def extract_equation(line: str) -> str | None:
    """Subset of the pinned upstream line extractor; arithmetic is parsed safely."""
    boxed = re.search(r"\\boxed\{([^{}]+)\}", line)
    if boxed:
        return boxed.group(1).strip()
    patterns = (
        r"(?:So\s+)?(?:the\s+)?(?:expression|answer|solution|result)(?:\s+would\s+be|\s+is|\s*:\s*)\s*([\d() +\-*/×÷\\]+?)(?:\.|,|$|\s*=)",
        r"\b(?:So|Then|Thus|Therefore|Hence)\b[,]?\s+([\d() +\-*/×÷\\]+?)\s*=\s*24",
        r"([\d() +\-*/×÷\\]+)\s*=\s*24",
        r"([\d() +\-*/×÷\\]+?)\s+(?:evaluates?\s+to|equals?|gives?)\s+24",
    )
    for pattern in patterns:
        m = re.search(pattern, line, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if re.search(r"[+\-*/×÷]|\\(?:times|cdot|div)", candidate):
                return candidate
    # The upstream monitor also recognizes standalone parenthesized equations.
    m = re.search(r"(\([^)]+\)\s*[+\-*/×÷]\s*(?:\([^)]+\)|\d+))", line)
    return m.group(1).strip() if m else None


class KStableDetector:
    """K=2 repeated full-number equation detector inspired by InterWhen source."""
    def __init__(self, expected_numbers: list[int], k: int = 2):
        self.expected = Counter(expected_numbers)
        self.k = k
        self.trigger_equation: str | None = None
        self.trigger_line_count: int | None = None

    def stable(self, text: str) -> bool:
        previous = None
        count = 0
        for line_no, line in enumerate(text.splitlines(), start=1):
            expr = extract_equation(line)
            if not expr:
                continue
            normalized = normalize_equation(expr)
            try:
                tree = ast.parse(normalized, mode="eval")
                if Counter(_integer_leaves(tree)) != self.expected:
                    continue
            except (SyntaxError, InvalidExpression):
                continue
            if normalized == previous:
                count += 1
            else:
                previous, count = normalized, 1
            if count >= self.k:
                self.trigger_equation = expr
                self.trigger_line_count = line_no
                return True
        return False


def extract_final_boxed(text: str) -> str | None:
    marker = "</think>"
    if marker in text:
        search = text.rsplit(marker, 1)[1]
    elif "<think>" in text:
        return None
    else:
        search = text
    starts = list(re.finditer(r"\\boxed\{", search))
    if not starts:
        return None
    start = starts[-1].end()
    depth, end = 1, start
    while end < len(search) and depth:
        if search[end] == "{":
            depth += 1
        elif search[end] == "}":
            depth -= 1
        end += 1
    if depth:
        return None
    expr = search[start:end - 1].strip()
    expr = normalize_equation(expr)
    frac = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac.search(expr):
        expr = frac.sub(r"(\1/\2)", expr)
    expr = re.sub(r"\)\s*\(", ")*(", expr)
    expr = re.sub(r"\)\s*(\d)", r")*\1", expr)
    expr = re.sub(r"(\d)\s*\(", r"\1*(", expr)
    expr = re.sub(r"\s*=\s*[\d.]+\s*$", "", expr)
    return expr


def prompt(numbers: list[int]) -> str:
    return PROMPT_TEMPLATE.format(numbers=", ".join(map(str, numbers)))


def progress_record(completed_episodes: int) -> str:
    """Serialize progress separately from print's stream-control arguments."""
    return json.dumps({"completed_episodes": completed_episodes,
                       "completed_tasks": completed_episodes // 2})


def run(args):
    config = json.loads(CONFIG.read_text())
    tasks_meta = json.loads(TASKS.read_text())
    if sha256_file(TASKS) != config["dataset"]["tasks_sha256"]:
        raise ValueError("Frozen task checksum differs from protocol")
    if len(tasks_meta["tasks"]) != config["dataset"]["sample_size"]:
        raise ValueError("Frozen task count differs from protocol")
    taskset_audit = audit_taskset(TASKS)
    if taskset_audit["status"] != "passed" or taskset_audit["tasks_sha256"] != config["dataset"]["tasks_sha256"]:
        raise ValueError("Frozen task set failed independent exact-solvability audit")
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse run directory: {args.output}")
    args.output.mkdir(parents=True)
    for name in ("episodes.jsonl", "calls.jsonl"):
        (args.output / name).touch()
    start = time.monotonic()
    deadline = start + args.wall_seconds
    source_files = [CONFIG, TASKS, Path(__file__), ROOT / "data/interwhen_kstable_game24_v1/README.md",
                    ROOT / "scripts/audit_interwhen_kstable_taskset.py",
                    ROOT / "docs/interwhen_kstable_game24_v1_protocol.md",
                    ROOT / "cluster/bouchet/interwhen_kstable_game24_v1b.sbatch",
                    ROOT / "src/jev_control/cuda_backend.py", ROOT / "src/jev_control/game24_exact.py",
                    ROOT / "src/jev_control/prospective.py"]
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files}
    manifest = {
        "status": "initializing", "config": config, "source_sha256": source_hashes,
        "task_file_sha256": sha256_file(TASKS), "taskset_audit": taskset_audit,
        "started_unix": time.time(),
        "claim_limit": "runtime/baseline adaptation only; no Jev efficacy or InterWhen reproduction",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    import torch
    import transformers
    if torch.__version__.split("+")[0] != "2.9.1" or torch.version.cuda != "12.8" or transformers.__version__ != "4.55.2":
        raise RuntimeError(f"Unexpected runtime: torch={torch.__version__}, cuda={torch.version.cuda}, transformers={transformers.__version__}")
    backend = CUDABackend(args.model, temperature=config["generation"]["temperature"],
                          top_p=config["generation"]["top_p"], max_context_tokens=16384)
    manifest.update({
        "status": "running", "packages": {"torch": torch.__version__,
        "transformers": transformers.__version__}, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model_revision": config["model"]["revision"], "dtype": "bfloat16",
        "temperature": config["generation"]["temperature"], "top_p": config["generation"]["top_p"],
        "model_identity": pinned_model_identity(Path(args.model)),
    })
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    max_tokens = config["generation"]["max_new_tokens"]
    row_count = 0
    try:
        for item in tasks_meta["tasks"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Cooperative wall deadline reached")
            nums, row_idx = item["numbers"], item["row_idx"]
            user_prompt = prompt(nums)
            prompt_ids = backend.encode_chat([{"role": "user", "content": user_prompt}])
            seed = 42 + row_idx
            for policy in config["policies"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Cooperative wall deadline reached")
                episode_start = time.monotonic()
                detector = KStableDetector(nums, k=2) if policy == "interwhen_kstable_k2" else None
                emitted_lines = 0

                def stop_when(ids):
                    nonlocal emitted_lines
                    if detector is None or not ids:
                        return False
                    tail = backend.tokenizer.decode(ids[-8:], skip_special_tokens=False)
                    if "\n" not in tail and "</think>" not in tail:
                        return False
                    text = backend.tokenizer.decode(ids, skip_special_tokens=False)
                    if "</think>" in text:
                        return False
                    line_count = text.count("\n")
                    if line_count <= emitted_lines:
                        return False
                    emitted_lines = line_count
                    return detector.stable(text)

                initial = backend.generate(prompt_ids, max_tokens, seed,
                                           timeout_s=config["generation"]["call_timeout_seconds"],
                                           stop_when=stop_when if detector else None)
                initial_raw_text = backend.tokenizer.decode(initial.token_ids, skip_special_tokens=False)
                all_text = initial_raw_text
                generated = initial.generated_tokens
                prompt_processed = initial.prompt_tokens
                seconds = initial.elapsed_seconds
                records = [initial.to_dict()]
                injected_close_tokens = 0
                injected_close_ids = []
                triggered = bool(detector and detector.trigger_equation and initial.finish_reason == "checkpoint")
                if triggered:
                    close_ids = backend.encode_text("</think>")
                    if (not close_ids or
                        backend.tokenizer.decode(close_ids, skip_special_tokens=False) != "</think>"):
                        raise RuntimeError("Tokenizer did not round-trip the </think> control token")
                    injected_close_tokens = len(close_ids)
                    injected_close_ids = close_ids
                    remaining = max(0, max_tokens - generated)
                    if remaining:
                        resumed = backend.generate(prompt_ids + initial.token_ids + close_ids, remaining,
                                                   seed + 1000003,
                                                   timeout_s=config["generation"]["call_timeout_seconds"])
                        resumed_raw_text = backend.tokenizer.decode(resumed.token_ids, skip_special_tokens=False)
                        all_text += "</think>" + resumed_raw_text
                        generated += resumed.generated_tokens
                        prompt_processed += resumed.prompt_tokens
                        seconds += resumed.elapsed_seconds
                        records.append(resumed.to_dict())
                expression = extract_final_boxed(all_text)
                checked = check_game24(expression, nums) if expression else None
                episode = {
                    "dataset_row_idx": row_idx, "numbers": nums, "policy": policy,
                    "seed": seed, "prompt": user_prompt, "trace_and_answer": all_text,
                    "final_expression": expression,
                    "exact_outcome": bool(checked and checked.valid),
                    "exact_failure_reason": None if checked is None or checked.valid else checked.reason,
                    "early_stop_triggered": triggered,
                    "trigger_equation": detector.trigger_equation if detector else None,
                    "trigger_line_count": detector.trigger_line_count if detector else None,
                    "monitor_trace_prefix": initial_raw_text if triggered else None,
                    "initial_finish_reason": initial.finish_reason,
                    "generated_tokens": generated, "injected_control_tokens": injected_close_tokens,
                    "injected_close_token_ids": injected_close_ids,
                    "prompt_tokens_processed": prompt_processed, "model_service_seconds": seconds,
                    "episode_wall_seconds": time.monotonic() - episode_start,
                    "calls": len(records),
                    "call_records": records,
                }
                durable_record(args.output / "episodes.jsonl", episode)
                for call in records:
                    durable_record(args.output / "calls.jsonl", {
                        "dataset_row_idx": row_idx, "policy": policy,
                        "generated_tokens": call["generated_tokens"],
                        "prompt_tokens": call["prompt_tokens"],
                        "elapsed_seconds": call["elapsed_seconds"],
                        "finish_reason": call["finish_reason"],
                    })
                row_count += 1
            print(progress_record(row_count), flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - start,
                        completed_episodes=row_count)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.output / "summary.json").write_text(json.dumps({"status": manifest["status"],
            "completed_episodes": row_count, "planned_episodes": 2 * len(tasks_meta["tasks"]),
            "analysis_status": "Do not analyze policy outcome rates before the complete independent record audit."}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=19800)
    args = parser.parse_args()
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error("wall seconds must be finite and positive")
    run(args)
