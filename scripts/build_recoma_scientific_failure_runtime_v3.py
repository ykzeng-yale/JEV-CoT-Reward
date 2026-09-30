#!/usr/bin/env python3
"""Build a separate, immutable v3 source tree from the frozen v2 runtime.

This modifies accounting/error handling only. No generated action is retried,
repaired, resampled, or substituted. A typed model-format failure ends its task;
unknown exceptions abort the run. Existing model/prompt/controller parameters
and simulator source remain unchanged.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


ACCOUNTING_SOURCE = '''"""Typed scientific failures and durable task accounting, without retries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


class ScientificTaskFailure(ValueError):
    """Only a declared model-output failure; never a simulator/runtime error."""

    CODES = frozenset({"missing_action_json", "invalid_action_json",
                       "non_object_action_json", "invalid_submit_arguments"})

    def __init__(self, code, raw_output):
        if code not in self.CODES or not isinstance(raw_output, str):
            raise TypeError("scientific failure requires a declared code and text")
        self.code = code
        self.output_sha256 = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        # No raw output/private state in exception text.
        super().__init__(code)

    def record(self):
        return {"code": self.code, "output_sha256": self.output_sha256,
                "retry_or_resample": False}


def parse_generated_action(payload, raw_output):
    """Preserve upstream JSON extraction; validate only its crash-prone shape.

    Valid object actions, including environment-denied actions and dialog
    objects, are returned unchanged. None means upstream found no JSON. Other
    unexpected internal payload types are infrastructure errors, not failures.
    """
    if payload is None:
        raise ScientificTaskFailure("missing_action_json", raw_output)
    if not isinstance(payload, str):
        raise TypeError("unexpected internal action payload type")
    try:
        action = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ScientificTaskFailure("invalid_action_json", raw_output) from exc
    if not isinstance(action, dict):
        raise ScientificTaskFailure("non_object_action_json", raw_output)
    if action.get("action") == "SUBMIT" and (
            not isinstance(action.get("arg1"), str)
            or not isinstance(action.get("thought", ""), str)):
        raise ScientificTaskFailure("invalid_submit_arguments", raw_output)
    return action


def prediction_record(prediction):
    """Record an ended task without mutating its example or official scorecard."""
    row = dict(prediction.example.__dict__)
    row["task_id"] = prediction.example.unique_id
    try:
        value = json.loads(prediction.prediction)
    except (json.JSONDecodeError, TypeError):
        value = prediction.prediction
    row["predicted"] = value
    metadata = dict(prediction.final_state.data)
    failure = metadata.get("scientific_task_failure")
    row["scientific_task_status"] = "scientific_failure" if failure else "completed"
    row["runtime_contract"] = "recoma_scientific_failure_accounting_v3"
    row["runtime_contract_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    card = metadata.get("final_scorecard")
    if not isinstance(card, list) or len(card) != 1:
        raise RuntimeError("ended task lacks a unique terminal scorecard")
    # Preserve official partial progress; do not turn a model-format failure
    # into success or silently zero its actual progress score.
    observed_success = card[0].get("completedSuccessfully")
    if type(observed_success) is not bool:
        raise RuntimeError("ended task lacks an official success Boolean")
    row["failure_adjusted_completed_successfully"] = observed_success and failure is None
    row["metadata"] = metadata
    return row


class DurableJSONL:
    def __init__(self, path, byte_cap):
        self.path = Path(path)
        if type(byte_cap) is not int or byte_cap <= 0:
            raise ValueError("positive output byte cap required")
        self.byte_cap = byte_cap
        self.bytes_written = 0
        # Refuse an existing ledger instead of overwriting/resuming/retrying.
        with self.path.open("x", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, row):
        raw = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\\n"
        size = len(raw.encode("utf-8"))
        if self.bytes_written + size > self.byte_cap:
            raise RuntimeError("durable accounting output byte cap exceeded")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.bytes_written += size


def collect_durable_predictions(reader, search, input_file, output_dir):
    """Persist each ended task before starting another; abort unknown errors."""
    root = Path(output_dir)
    tasks = DurableJSONL(root / "task_records.jsonl",
                         int(os.environ.get("RECOMA_TASK_RECORDS_MAX_BYTES", "33554432")))
    events = DurableJSONL(root / "task_events.jsonl",
                          int(os.environ.get("RECOMA_TASK_EVENTS_MAX_BYTES", "4194304")))
    predictions, seen = [], set()
    for example in reader.get_examples(input_file):
        task_id = example.unique_id
        if task_id in seen:
            raise RuntimeError("duplicate task identity before inference")
        seen.add(task_id)
        started = time.monotonic()
        events.append({"event": "task_started", "task_id": task_id,
                       "monotonic_seconds": time.monotonic()})
        try:
            prediction = search.predict(example)
            row = prediction_record(prediction)
            if row["task_id"] != task_id:
                raise RuntimeError("prediction identity differs from requested task")
            row["task_wall_seconds"] = time.monotonic() - started
            tasks.append(row)
        except Exception as exc:
            events.append({"event": "infrastructure_abort", "task_id": task_id,
                           "exception_type": type(exc).__name__,
                           "monotonic_seconds": time.monotonic()})
            raise
        predictions.append(prediction)
        events.append({"event": "task_ended", "task_id": task_id,
                       "scientific_task_status": row["scientific_task_status"],
                       "monotonic_seconds": time.monotonic()})
    return predictions
'''


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("frozen-source replacement anchor is missing or ambiguous")
    return text.replace(old, new, 1)


def patched_sources(source_root: Path) -> dict[str, str]:
    controller_path = "discoveryworld/agents/recoma/react_controller.py"
    controller = (source_root / controller_path).read_text()
    controller = replace_once(controller, "import re\n", "import re\n\nfrom recoma.utils.task_accounting import parse_generated_action\n")
    controller = replace_once(controller, '''            action_json = self.extract_json_output(last_child)
            try:
                formatted_json = json.loads(action_json)
                current_history.append(Action(action_str=last_child.output, action_json=formatted_json))
            except json.JSONDecodeError:
                raise ValueError("Failed to decode JSON from action output: {}".format(last_child.output))
''', '''            raw_output = last_child.output
            action_json = self.extract_json_output(last_child)
            formatted_json = parse_generated_action(action_json, raw_output)
            current_history.append(Action(action_str=last_child.output, action_json=formatted_json))
''')
    search_path = "recoma/recoma/search/search.py"
    search = (source_root / search_path).read_text()
    search = replace_once(search, "import logging\n", "import logging\n\nfrom recoma.utils.task_accounting import ScientificTaskFailure\n")
    search = replace_once(search, '''            for new_state in self.execute(current_state):
''', '''            try:
                next_states = self.execute(current_state)
            except ScientificTaskFailure as exc:
                # The model output is retained; this task ends without another
                # call, repair, fallback, environment action, or resampling.
                current_state.data["scientific_task_failure"] = exc.record()
                current_state.data["scientific_task_status"] = "scientific_failure"
                answer = self.answerer.generate_answer(current_state)
                return ExamplePrediction(example=example, prediction=answer,
                                         final_state=current_state)
            for new_state in next_states:
''')
    runner_path = "recoma/recoma/run_inference.py"
    runner = (source_root / runner_path).read_text()
    runner = replace_once(runner, "from typing import List\n", "from typing import List\n\nfrom recoma.utils.task_accounting import collect_durable_predictions\n")
    runner = replace_once(runner, '''    for example in reader.get_examples(args.input):
        example_predictions.append(search_algo.predict(example))
    dump_predictions(args, example_predictions)
''', '''    example_predictions = collect_durable_predictions(
        reader, search_algo, args.input, args.output_dir)
    dump_predictions(args, example_predictions)
''')
    backend_path = "recoma/recoma/models/impl/hf_torch_generator.py"
    backend = (source_root / backend_path).read_text()
    backend = replace_once(backend, '''        self.model_path = str(Path(model_path).resolve())
''', '''        self.model_path = str(Path(model_path).resolve())
        self.model_revision = Path(self.model_path).name
''')
    backend = replace_once(backend, '''        start = time.monotonic()
        try:
''', '''        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
''')
    backend = replace_once(backend, '''            elapsed = time.monotonic() - start
''', '''            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
''')
    backend = replace_once(backend, '''                "status": "ok", "task_id": state.example.unique_id,
                "model": self.model_id, "prompt_sha256": prompt_hash,
''', '''                "status": "ok", "task_id": state.example.unique_id,
                "model": self.model_id, "model_revision": self.model_revision,
                "prompt_sha256": prompt_hash,
                "controller_output_sha256": hashlib.sha256(text.lstrip().encode("utf-8")).hexdigest(),
                "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved()),
''')
    return {
        controller_path: controller,
        search_path: search,
        runner_path: runner,
        backend_path: backend,
        "recoma/recoma/utils/task_accounting.py": ACCOUNTING_SOURCE,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_root: Path, output: Path, patch_path: Path) -> dict:
    if output.exists() or patch_path.exists():
        raise FileExistsError("v3 build refuses existing output or patch")
    changes = patched_sources(source_root)
    shutil.copytree(source_root, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    diffs = []
    for relative, content in changes.items():
        baseline = source_root / relative
        old = baseline.read_text() if baseline.exists() else ""
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        (output / relative).write_text(content)
        compile(content, relative, "exec")
        diffs.append("diff --git a/" + relative + " b/" + relative + "\n")
        if not baseline.exists():
            diffs.append("new file mode 100644\n")
        diffs.extend(difflib.unified_diff(old.splitlines(keepends=True), content.splitlines(keepends=True),
                                        fromfile="a/" + relative if baseline.exists() else "/dev/null",
                                        tofile="b/" + relative))
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text("".join(diffs))
    files = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {str(path.relative_to(output)): digest(path) for path in files}
    (output / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in manifest.items()))
    receipt = {
        "runtime": "recoma_scientific_failure_accounting_v3",
        "baseline_source": str(source_root.resolve()), "output_source": str(output.resolve()),
        "patch_sha256": digest(patch_path), "source_manifest_sha256": digest(output / "SHA256SUMS"),
        "source_files": len(files),
        "changed_files": {name: {"before_sha256": digest(source_root / name) if (source_root / name).exists() else None,
                                  "after_sha256": digest(output / name)} for name in changes},
        "scope": "Accounting-only model-format failure handling; no retries, policy repair, model or parameter changes.",
    }
    (output.parent / "build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.source_root, args.output, args.patch)
    print(json.dumps({key: receipt[key] for key in ("runtime", "source_files", "patch_sha256", "source_manifest_sha256")}))


if __name__ == "__main__":
    main()
