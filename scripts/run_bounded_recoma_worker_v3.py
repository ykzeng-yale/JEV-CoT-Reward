#!/usr/bin/env python3
"""Run an exact frozen ReCoMA task shard, preserving upstream control and v3 receipts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def selected_examples(reader, expected_ids: list[str]):
    if len(expected_ids) != len(set(expected_ids)) or not expected_ids:
        raise ValueError("frozen worker task IDs must be nonempty and unique")
    expected = set(expected_ids)
    examples = [example for example in reader.get_examples("") if example.unique_id in expected]
    actual = [example.unique_id for example in examples]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"reader task coverage mismatch: actual={actual}, expected={expected_ids}")
    return examples


class FrozenReader:
    def __init__(self, examples):
        self.examples = examples

    def get_examples(self, _):
        yield from self.examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.config = args.config.resolve()
    manifest = json.loads(args.manifest.read_text())
    worker = manifest["workers"][args.worker_index]
    expected_ids = worker["task_ids"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runtime = {"status": "starting", "worker_index": args.worker_index,
               "expected_task_ids": expected_ids, "model_revision": manifest["model_revision"],
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "performed_generation": False}
    try:
        import torch
        import transformers
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("each worker requires exactly one allocated visible GPU")
        if torch.__version__.split("+")[0] != manifest["runtime"]["torch_version"]:
            raise RuntimeError("pinned PyTorch mismatch")
        if transformers.__version__ != manifest["runtime"]["transformers_version"]:
            raise RuntimeError("pinned Transformers mismatch")
        model_path = Path(os.environ["MODEL_PATH"]).resolve(strict=True)
        if model_path.name != manifest["model_revision"]:
            raise RuntimeError("snapshot revision mismatch")
        runtime.update(cuda_device_count=1, cuda_device=torch.cuda.get_device_name(0),
                       cuda_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                       torch=torch.__version__, transformers=transformers.__version__,
                       model_path=str(model_path))
        if manifest["audit_scope"] != "runtime_qualification":
            qualified_gpu = manifest["qualified_gpu"]
            if runtime["cuda_device"] != qualified_gpu["name"] or runtime["cuda_total_memory_bytes"] != qualified_gpu["total_memory_bytes"]:
                raise RuntimeError("allocated GPU differs from measured qualification device")
        torch.cuda.reset_peak_memory_stats()
        from recoma.utils.class_utils import import_module_and_submodules
        import_module_and_submodules("agents.recoma.discoveryworld_env_models")
        import_module_and_submodules("agents.recoma.discoveryworld_promptlm")
        import _jsonnet
        config = json.loads(_jsonnet.evaluate_file(str(args.config), ext_vars=os.environ.copy()))
        from recoma.datasets.reader import DatasetReader
        examples = selected_examples(DatasetReader.from_dict(copy.deepcopy(config["reader"])), expected_ids)
        generator = config["models"]["action"]["generator_params"]
        required = {"type": "hf_torch", "model": manifest["model"],
                    "max_tokens": manifest["max_new_tokens_per_generation"],
                    "temperature": manifest["temperature"], "seed": manifest["generator_seed"],
                    "stop": manifest["stop_sequences"]}
        if any(generator.get(k) != v for k, v in required.items()):
            raise RuntimeError("compiled generator differs from frozen manifest")
        if config["search"]["stopping_conditions"] != [
            {"type": "max_env_calls", "max_env_calls": manifest["max_environment_actions_per_episode"]},
            {"type": "max_llm_calls", "max_llm_calls": manifest["max_llm_calls_per_episode"]},
        ]:
            raise RuntimeError("compiled episode limits differ from manifest")
        selection = {"expected_task_ids": expected_ids,
                     "executed_reader_order": [x.unique_id for x in examples],
                     "selection_basis": "frozen IDs only; no task outcomes read"}
        (args.output_dir / "task_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        config_path = args.output_dir / "executed_config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        workdir = args.output_dir / "runtime_workdir"
        workdir.mkdir()
        # Official API writes relative video frames. Keep them in each worker's
        # monitored outputs, never in the shared immutable source tree.
        (workdir / "agents").symlink_to(args.manifest.parent / "source/discoveryworld/agents", target_is_directory=True)
        os.chdir(workdir)
        from recoma.run_inference import build_configurable_systems, inference_mode
        systems = build_configurable_systems(str(config_path), str(args.output_dir))
        systems.reader = FrozenReader(examples)
        # History length alone cannot bound the dynamically injected scene.
        # Refuse an unqualified larger context; do not truncate or resample.
        action_generator = systems.search.model_list["action"].generator
        original_generate = action_generator.generate
        def within_qualified_context(input_str, state):
            messages = action_generator.extract_role_messages(input_str)
            token_ids = action_generator.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            if len(token_ids) > manifest["qualified_context_token_ceiling"]:
                raise RuntimeError("prompt exceeds prospectively frozen hardware context envelope")
            return original_generate(input_str, state)
        action_generator.generate = within_qualified_context
        runtime["performed_generation"] = True
        # Full model prompts/outputs are durably logged by RECOMA_USAGE_LOG.
        # Upstream duplicate prompt dumps assume QAExample.qid; DiscoveryWorld
        # implements only the required unique_id interface.
        inference_mode(SimpleNamespace(input="", output_dir=str(args.output_dir), dump_prompts=False), systems)
        calls = [json.loads(line) for line in Path(os.environ["RECOMA_USAGE_LOG"]).read_text().splitlines() if line]
        episode_allocated = max([torch.cuda.max_memory_allocated()] + [c.get("peak_allocated_gpu_bytes", 0) for c in calls])
        episode_reserved = max([torch.cuda.max_memory_reserved()] + [c.get("peak_reserved_gpu_bytes", 0) for c in calls])
        if manifest["audit_scope"] == "runtime_qualification":
            from recoma_context_stress_v3 import run_context_stress
            stress = run_context_stress(
                action_generator,
                args.output_dir / "context_stress.json",
                target_tokens=manifest["context_stress_input_tokens"],
                completion_tokens=manifest["context_stress_output_tokens"],
            )
            runtime["context_stress"] = {
                k: v for k, v in stress.items()
                if k not in {"input_token_ids", "output_token_ids", "output_text"}
            }
        torch.cuda.synchronize()
        runtime.update(status="completed", max_memory_allocated_bytes=max(episode_allocated, torch.cuda.max_memory_allocated()),
                       max_memory_reserved_bytes=max(episode_reserved, torch.cuda.max_memory_reserved()))
    except BaseException as exc:
        runtime.update(status="infrastructure_failure", exception_type=type(exc).__name__,
                       exception_message=str(exc))
        (args.output_dir / "infrastructure_traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        runtime["elapsed_seconds"] = time.monotonic() - started
        (args.output_dir / "runtime_report.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
