#!/usr/bin/env python3
"""Gated all-arm mechanism collection from an online emitted-only checkpoint.

Prepare/review the development gate before invoking this runner. This script
does not train a policy or promote itself to another experimental stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_control.features import QUESTIONS, SCHEMA_VERSION, judge_state
from jev_control.mlx_backend import MLXBackend
from jev_control.tasks import make_task, verify
from run_development import PROMPT_SUFFIX, boundary_now, calculation_marker
from run_screen import ACTIONS, retained_statistics, rollout

PROTOCOL = "mechanism-v1-online-emitted-checkpoint"
CONFIG = ROOT / "configs/mechanism_v1.json"
STATE_SEGMENTATION = "Prior paragraphs as history; last retained paragraph, possibly partial, as latest_segment. Split only retained text at its last double newline after stripping trailing whitespace."


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed while recording its revision")
    return digest.hexdigest()


def model_identity(path):
    """Pin local assets by content; never resolve a model through the network."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("model must be an existing local model snapshot")
    config = json.loads((path / "config.json").read_text())
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {
        ".json", ".safetensors", ".bin", ".npz", ".model", ".txt", ".jinja", ".tiktoken",
    })
    weights = [p for p in files if p.suffix in {".safetensors", ".bin", ".npz"}]
    if not weights:
        raise ValueError("Local snapshot has no weight files")
    hashes = {str(p.relative_to(path)): file_sha256(p) for p in files}
    weight_hashes = {str(p.relative_to(path)): hashes[str(p.relative_to(path))] for p in weights}
    return {"path": str(path), "config": config, "file_sha256": hashes,
            "weight_files_sha256": weight_hashes,
            "weight_revision_sha256": hashlib.sha256(json.dumps(weight_hashes, sort_keys=True).encode()).hexdigest(),
            "snapshot_revision": path.name if path.parent.name == "snapshots" else None,
            "config_commit_hash": config.get("_commit_hash")}


def freeze_sources(output):
    paths = [ROOT / "scripts" / name for name in (
        "run_mechanism.py", "run_screen.py", "run_development.py", "analyze_screen.py", "local_judge.py", "run_bounded.py")]
    paths += sorted((ROOT / "src/jev_control").glob("*.py"))
    paths += [CONFIG, ROOT / "configs/stages.json", ROOT / "pyproject.toml", ROOT / "requirements.lock.txt"]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        data = path.read_bytes()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    return hashes


def schedule_for(index, seed, repeats=4):
    schedule = [{"action": action, "repeat": repeat,
                 "seed": seed + index * 10000 + ACTIONS.index(action) * 1000 + repeat * 10}
                for action in ACTIONS for repeat in range(repeats)]
    random.Random(seed + index).shuffle(schedule)
    return schedule


def planned_tasks(args):
    planned = []
    for index in range(args.start_index, args.start_index + args.problems):
        task = make_task(index, args.seed)
        task["prompt"] += PROMPT_SUFFIX
        planned.append({"index": index, "task": task, "initial_seed": args.seed + index,
                        "schedule": schedule_for(index, args.seed, args.repeats)})
    return planned


def checkpoint_from_initial(backend, task, prompt, initial, target, cap):
    """Eligibility follows the actual stop event; no future scan or truncation."""
    if initial.finish_reason != "checkpoint":
        reason = ("completed_before_checkpoint" if initial.finish_reason == "stop" else
                  "timeout_before_checkpoint" if initial.finish_reason == "timeout" else
                  "answer_phase_reached_before_checkpoint" if "FINAL:" in initial.text else
                  "no_boundary_before_cap")
        return None, reason
    retained = list(initial.token_ids)
    if (not target <= len(retained) <= cap or initial.generated_tokens != len(retained)
            or not boundary_now(backend, retained, target)):
        raise ValueError("Backend reported an invalid checkpoint stop")
    text = backend.decode(retained)
    # A newline stops generation, but the rubric needs the coherent paragraph
    # containing that line. No continuation or gold data enters this split.
    parts = text.rstrip().rsplit("\n\n", 1)
    history, segment = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
    checkpoint = {"problem_id": task["id"], "task": task,
                  "prompt_ids": list(prompt), "retained_ids": retained,
                  "initial": initial.to_dict(), "discarded_prefix_tail_tokens": 0,
                  "retained_features": retained_statistics(initial, len(retained)),
                  "calculation_marker": calculation_marker(text),
                  "state": judge_state(task["prompt"], history, segment)}
    checkpoint["sha256"] = hashlib.sha256(json.dumps(
        {"prompt": prompt, "retained": retained}, sort_keys=True).encode()).hexdigest()
    return checkpoint, None


class WalltimeExceeded(RuntimeError):
    pass


class LoggedBackend:
    """Durable completed-call records survive an interrupted multi-call action."""
    def __init__(self, backend, output, deadline):
        self.backend, self.output, self.deadline = backend, output, deadline
        self.context = {}
        self.events = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, prefix, max_tokens, seed, timeout_s=180, **kwargs):
        remaining_seconds = self.deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise WalltimeExceeded("Cooperative run deadline reached")
        event = {**self.context, "sequence": len(self.events), "started_unix": time.time(),
                 "prefix_ids": list(prefix), "max_tokens": max_tokens, "seed": seed}
        # A hard process kill cannot execute finally. An unmatched start is
        # explicit evidence of a call with unknown emitted/computed work.
        append_jsonl(self.output / "generation_started.jsonl", event)
        try:
            result = self.backend.generate(prefix, max_tokens, seed,
                                           timeout_s=min(timeout_s, remaining_seconds), **kwargs)
        except BaseException as exc:
            event.update({"status": "error", "error_type": type(exc).__name__,
                          "generated_work_unknown": True})
            append_jsonl(self.output / "generation_events.jsonl", event)
            self.events.append(event)
            raise
        event.update({"status": "complete", "generation": result.to_dict()})
        append_jsonl(self.output / "generation_events.jsonl", event)
        self.events.append(event)
        return result


def summarize(args, rows, checkpoints, skipped, events, started, status, client=None):
    generations = [event["generation"] for event in events if event["status"] == "complete"]
    return {"kind": "gated_all_arm_mechanism_not_policy_evaluation", "status": status,
            "attempted_problems": args.problems, "recorded_problems": len(checkpoints) + len(skipped),
            "eligible_problems": len(checkpoints), "skipped_problems": len(skipped),
            "skip_reasons": dict(Counter(row["reason"] for row in skipped)), "episodes": len(rows),
            "planned_episodes": args.problems * len(ACTIONS) * args.repeats,
            "actions": {a: {"n": sum(r["action"] == a for r in rows),
                             "successes": sum(r["outcome"]["success"] for r in rows if r["action"] == a)} for a in ACTIONS},
            "continuation_generated_tokens": sum(r["generated_tokens"] for r in rows),
            "continuation_seconds": sum(r["elapsed_seconds"] for r in rows),
            "actual_shared_prefix_tokens": sum(r["initial"]["generated_tokens"] for r in checkpoints + skipped),
            "durable_completed_call_generated_tokens": sum(g["generated_tokens"] for g in generations),
            "durable_completed_call_seconds": sum(g["elapsed_seconds"] for g in generations),
            "generation_finish_reasons": dict(Counter(g["finish_reason"] for g in generations)),
            "failed_calls_with_unknown_work": sum(event["status"] == "error" for event in events),
            "wall_elapsed_seconds": time.monotonic() - started,
            "jev_budget": client.status() if client else None,
            "interpretation": "All-arm collection conditional on checkpoint eligibility; no learned-policy claim. Repeats are clustered within problems. Incomplete actions remain in the durable call ledger."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--problems", type=int, default=24)
    parser.add_argument("--seed", type=int, default=491027)
    parser.add_argument("--repeats", type=int, choices=[4], default=4)
    parser.add_argument("--budget", type=int, choices=[1024], default=1024)
    parser.add_argument("--checkpoint-target", type=int, choices=[256], default=256)
    parser.add_argument("--checkpoint-cap", type=int, choices=[384], default=384)
    parser.add_argument("--final-reserve", type=int, choices=[96], default=96)
    parser.add_argument("--quantize-bits", type=int, choices=[0, 4], default=4,
                        help="0 loads an already quantized local snapshot without requantization")
    parser.add_argument("--max-walltime-seconds", type=float, default=7200)
    parser.add_argument("--jev", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.start_index < 24 or not 1 <= args.problems <= 24 - args.start_index:
        parser.error("Use a nonempty shard of the fixed indices 0 through 23")
    if args.seed < 0 or not math.isfinite(args.max_walltime_seconds) or args.max_walltime_seconds <= 0:
        parser.error("Seed must be nonnegative and wall-clock limit finite and positive")
    if args.output.exists():
        parser.error("Output already exists; partial runs are preserved and never silently replayed")
    return args


def main(argv=None):
    args = parse_args(argv)
    started, started_unix = time.monotonic(), time.time()
    args.output.mkdir(parents=True)
    hashes = freeze_sources(args.output)
    planned = planned_tasks(args)
    write_json(args.output / "schedule.json", planned)
    write_json(args.output / "rubric.json", {"schema": SCHEMA_VERSION, "questions": QUESTIONS})
    packages = {}
    for package in ("mlx", "mlx-lm", "numpy", "transformers", "httpx", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {**vars(args), "output": str(args.output.resolve()),
                "kind": "gated_all_arm_mechanism_not_policy_evaluation", "protocol": PROTOCOL,
                "schema": SCHEMA_VERSION, "implementation_version": PROTOCOL,
                "started_unix": started_unix, "status": "initializing", "source_sha256": hashes,
                "schedule_sha256": file_sha256(args.output / "schedule.json"),
                "rubric_sha256": file_sha256(args.output / "rubric.json"),
                "package_versions": packages, "python_version": sys.version,
                "task_seed": args.seed, "sampling_seed": args.seed, "actions": list(ACTIONS),
                "selector": "mean_base_logprob", "prompt_suffix": PROMPT_SUFFIX,
                "temperature": .7, "top_p": .9, "per_call_timeout_seconds": 180,
                "walltime_note": "Cooperative checks before calls and after emitted tokens; prefill or active kernels need an external hard deadline.",
                "interruption_accounting": "An external kill can prevent finalization. Match generation_started.jsonl against generation_events.jsonl by sequence; an unmatched start has unknown generated/computed work. A running manifest or missing final summary is not a completed ledger.",
                "source_snapshot_note": "Complete runner/dependencies/analyzer/local-judge/config source snapshots under source/ with original relative paths.",
                "jev": args.jev, "jev_cumulative_stage_cap_usd": 1.0,
                "state_segmentation": STATE_SEGMENTATION,
                "interpretation": "Development-gated mechanism collection; no learned-policy claim or common-RNG claim."}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "skipped.json", [])
    for name in ("checkpoints.jsonl", "checkpoint_attempts.jsonl", "outcomes.jsonl", "generation_events.jsonl", "generation_started.jsonl"):
        (args.output / name).touch()
    rows, checkpoints, skipped = [], [], []
    backend = client = None
    status = "failed"
    try:
        manifest["model_identity"] = model_identity(args.model)
        from jev_control.jev import MODEL as JEV_MODEL, DEFAULT_DB
        manifest["jev_model"] = JEV_MODEL
        manifest["jev_ledger"] = str(DEFAULT_DB)
        write_json(args.output / "manifest.json", manifest)
        if time.monotonic() - started >= args.max_walltime_seconds:
            raise WalltimeExceeded("Deadline reached during model audit")
        raw_backend = MLXBackend(args.model, quantization_bits=args.quantize_bits or None)
        manifest["actual_model_config"] = raw_backend.model_config
        manifest["actual_quantization_config"] = raw_backend.quantization_config
        manifest["status"] = "running"
        write_json(args.output / "manifest.json", manifest)
        backend = LoggedBackend(raw_backend, args.output, started + args.max_walltime_seconds)
        if args.jev:
            from jev_control.jev import JevClient
            client = JevClient(stage_cap_usd=1.0)
        for item in planned:
            index, task = item["index"], item["task"]
            backend.context = {"phase": "initial", "problem_id": task["id"], "index": index}
            prompt = backend.encode_chat([{"role": "user", "content": task["prompt"]}])
            initial = backend.generate(prompt, args.checkpoint_cap, item["initial_seed"], timeout_s=180,
                                       stop_when=lambda ids: boundary_now(backend, ids, args.checkpoint_target))
            checkpoint, reason = checkpoint_from_initial(backend, task, prompt, initial,
                                                        args.checkpoint_target, args.checkpoint_cap)
            if reason is not None:
                skipped.append({"problem_id": task["id"], "task": task, "index": index,
                                "prompt_ids": prompt, "initial": initial.to_dict(), "reason": reason,
                                "initial_outcome": verify(task, initial.text)})
                write_json(args.output / "skipped.json", skipped)
                print(json.dumps({"index": index, "status": "ineligible", "reason": reason}), flush=True)
                continue
            checkpoint["index"] = index
            checkpoints.append(checkpoint)
            # Keep the observed state even if an external kill occurs while
            # the optional paid judgment is in flight.
            append_jsonl(args.output / "checkpoint_attempts.jsonl", checkpoint)
            try:
                if client:
                    if time.monotonic() >= backend.deadline:
                        raise WalltimeExceeded("Deadline reached before optional judgment")
                    checkpoint["jev"] = client.evaluate(checkpoint["state"], QUESTIONS)
            except BaseException as exc:
                checkpoint["jev_error"] = {"type": type(exc).__name__, "retry": False}
                raise
            finally:
                append_jsonl(args.output / "checkpoints.jsonl", checkpoint)
            for scheduled in item["schedule"]:
                backend.context = {"phase": "continuation", "problem_id": task["id"],
                                   "index": index, "action": scheduled["action"], "repeat": scheduled["repeat"]}
                row = rollout(backend, task, list(prompt), list(checkpoint["retained_ids"]),
                              scheduled["action"], scheduled["seed"],
                              args.budget - initial.generated_tokens, args.final_reserve)
                row.update({"problem_id": task["id"], "family": task["family"], "index": index,
                            "checkpoint_sha256": checkpoint["sha256"], "repeat": scheduled["repeat"],
                            "shared_prefix_generated_tokens": initial.generated_tokens})
                assert row["generated_tokens"] + initial.generated_tokens <= args.budget
                rows.append(row)
                append_jsonl(args.output / "outcomes.jsonl", row)
                print(json.dumps({"index": index, "action": row["action"], "repeat": row["repeat"],
                                  "success": row["outcome"]["success"],
                                  "generated_tokens": row["generated_tokens"]}), flush=True)
        status = "complete"
    except BaseException as exc:
        status = "walltime_exceeded" if isinstance(exc, WalltimeExceeded) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest.update({"status": status, "finished_unix": time.time(),
                         "wall_elapsed_seconds": time.monotonic() - started})
        write_json(args.output / "manifest.json", manifest)
        summary = summarize(args, rows, checkpoints, skipped, backend.events if backend else [],
                            started, status, client)
        write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
