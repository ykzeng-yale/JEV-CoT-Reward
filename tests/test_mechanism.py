"""Prospective all-arm runner checks: fake CPU generation, no paid requests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from jev_control.mlx_backend import Generation
from jev_control.features import QUESTIONS
import jev_control.jev as jev_module


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("run_mechanism", SCRIPTS / "run_mechanism.py")
    mechanism = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mechanism)
    analyzer_spec = importlib.util.spec_from_file_location("mechanism_analyzer", SCRIPTS / "analyze_screen.py")
    analyzer = importlib.util.module_from_spec(analyzer_spec)
    analyzer_spec.loader.exec_module(analyzer)
finally:
    sys.path.remove(str(SCRIPTS))


TASK = {"id": "test", "family": "weighted_path", "prompt": "Find a path",
        "data": {"n": 2, "edges": [[0, 1, 1]]}}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class FakeBackend:
    def __init__(self, initial_scripts=None):
        self.model_config = {"model_type": "fake", "quantization": {"bits": 4, "group_size": 64}}
        self.quantization_config = self.model_config["quantization"]
        self.initial_scripts = list(initial_scripts or [])
        self.calls = []
        self.fail_call = None
        self.continuation_finish = "stop"

    def encode_chat(self, messages):
        return [5000] + list(map(ord, messages[0]["content"]))

    def encode_text(self, text):
        return list(map(ord, text))

    def decode(self, ids):
        return "".join("COMPOUND_TOKEN" if token == 5000 else chr(token) for token in ids)

    def choose_candidate(self, candidates):
        return 0

    def generate(self, prefix, max_tokens, seed, timeout_s, stop_when=None):
        self.calls.append({"prefix": list(prefix), "max_tokens": max_tokens, "seed": seed,
                           "timeout_s": timeout_s, "online": stop_when is not None})
        if len(self.calls) == self.fail_call:
            raise RuntimeError("fake failure")
        if stop_when is not None:
            text, finish = (self.initial_scripts.pop(0) if self.initial_scripts else
                            ("x" * 252 + "1+1\nFUTURE_FINAL: A->B", "length"))
        else:
            text, finish = "FINAL: A->B", self.continuation_finish
        emitted = []
        for token in self.encode_text(text)[:max_tokens]:
            emitted.append(token)
            if len(emitted) == len(text) and finish in {"stop", "timeout"}:
                break
            if stop_when is not None and stop_when(list(emitted)):
                finish = "checkpoint"
                break
        return Generation(emitted, self.decode(emitted), len(prefix), len(emitted), -1., 2.,
                          .125, finish, 0., "fake", seed,
                          token_logprobs=[-1.] * len(emitted), token_entropies=[2.] * len(emitted))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    model = tmp_path / "snapshots" / "revision-abc"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"fake","quantization":{"bits":4}}')
    (model / "model.safetensors").write_bytes(b"not a real model; CPU-only fixture")
    (model / "tokenizer.json").write_text('{"fake":true}')
    backend = FakeBackend()
    loaded = []

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return backend

    monkeypatch.setattr(mechanism, "MLXBackend", load)
    monkeypatch.setattr(mechanism.importlib.metadata, "version", lambda _: "test-version")
    output = tmp_path / "run"
    argv = ["--model", str(model), "--output", str(output), "--problems", "1"]
    return backend, model, output, argv, loaded


def test_complete_schedule_pinned_sources_and_analyzer_compatibility(prepared):
    backend, model, output, argv, loaded = prepared
    mechanism.main(argv)
    manifest = json.loads((output / "manifest.json").read_text())
    plan = json.loads((output / "schedule.json").read_text())
    checkpoints = read_lines(output / "checkpoints.jsonl")
    rows = read_lines(output / "outcomes.jsonl")
    events = read_lines(output / "generation_events.jsonl")
    starts = read_lines(output / "generation_started.jsonl")
    assert manifest["status"] == "complete"
    assert (manifest["seed"], manifest["repeats"], manifest["budget"]) == (491027, 4, 1024)
    assert manifest["actual_quantization_config"] == backend.quantization_config
    assert manifest["actual_model_config"] == backend.model_config
    assert manifest["model_identity"]["snapshot_revision"] == "revision-abc"
    assert manifest["model_identity"]["file_sha256"]["model.safetensors"] == hashlib.sha256((model / "model.safetensors").read_bytes()).hexdigest()
    assert loaded[0][1] == {"quantization_bits": 4}
    assert manifest["finished_unix"] >= manifest["started_unix"]
    assert manifest["wall_elapsed_seconds"] >= 0
    for relative, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((output / "source" / relative).read_bytes()).hexdigest() == digest
    assert {"scripts/run_screen.py", "scripts/run_development.py", "scripts/local_judge.py",
            "scripts/analyze_screen.py", "src/jev_control/features.py", "configs/mechanism_v1.json"} <= set(manifest["source_sha256"])
    assert manifest["rubric_sha256"] == mechanism.file_sha256(output / "rubric.json")
    assert manifest["schedule_sha256"] == mechanism.file_sha256(output / "schedule.json")
    assert len(plan) == len(checkpoints) == 1
    assert len(rows) == 12
    assert [(row["action"], row["repeat"], row["seed"]) for row in rows] == [
        (entry["action"], entry["repeat"], entry["seed"]) for entry in plan[0]["schedule"]]
    cp = checkpoints[0]
    assert cp["retained_ids"] == cp["initial"]["token_ids"]
    assert len(cp["retained_ids"]) == cp["initial"]["generated_tokens"] == 256
    assert cp["discarded_prefix_tail_tokens"] == 0
    assert cp["retained_features"] == {"retained_stat_token_count": 256,
                                        "retained_mean_logprob": -1., "retained_mean_entropy": 2.}
    assert "FUTURE" not in json.dumps(cp)
    assert set(cp["state"]) == {"task", "history", "latest_segment"}
    assert len(events) == len(backend.calls) == 17  # Shared initial + 4/4/8 arm calls.
    assert [event["sequence"] for event in starts] == [event["sequence"] for event in events]
    assert "unknown" in manifest["interruption_accounting"]
    for row in rows:
        assert row["generated_tokens"] + row["shared_prefix_generated_tokens"] <= 1024
        assert row["checkpoint_sha256"] == cp["sha256"]
    for event in events:
        if event.get("action") in {"continue", "branch"}:
            assert event["prefix_ids"] == cp["prompt_ids"] + cp["retained_ids"]
    result = analyzer.analyze(output, draws=100)
    assert result["enrollment"]["complete_all_arm_problems"] == 1
    assert result["enrollment"]["expected_repeats_per_action"] == 4
    assert result["costs"]["ledger_issues"] == []
    assert result["crossfit"]["status"] == "not_run"


def test_schedule_is_prespecified_before_any_model_generation(prepared, monkeypatch):
    backend, _, output, argv, _ = prepared
    generate = backend.generate

    def checked(*args, **kwargs):
        plan = json.loads((output / "schedule.json").read_text())
        assert len(plan) == 1 and len(plan[0]["schedule"]) == 12
        assert (output / "rubric.json").exists()
        assert (output / "source/scripts/run_screen.py").exists()
        return generate(*args, **kwargs)

    monkeypatch.setattr(backend, "generate", checked)
    mechanism.main(argv)


def test_online_checkpoint_requires_actual_stop_and_never_trims_retroactively():
    backend = FakeBackend()
    ids = list(map(ord, "x" * 255 + "\n" + "future"))
    initial = Generation(ids, backend.decode(ids), 1, len(ids), -1, 1, .1, "length", 0, "", 1)
    cp, reason = mechanism.checkpoint_from_initial(backend, TASK, [5000], initial, 256, 384)
    assert cp is None
    assert reason == "no_boundary_before_cap"
    assert initial.token_ids == ids


@pytest.mark.parametrize("finish,text,reason", [
    ("stop", "FINAL: A->B", "completed_before_checkpoint"),
    ("timeout", "working", "timeout_before_checkpoint"),
    ("length", "FINAL: partial", "answer_phase_reached_before_checkpoint"),
    ("length", "x" * 384, "no_boundary_before_cap"),
])
def test_skips_remain_enrolled_and_never_generate_arms(prepared, finish, text, reason):
    backend, _, output, argv, _ = prepared
    backend.initial_scripts = [(text, finish)]
    mechanism.main(argv)
    skipped = json.loads((output / "skipped.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert len(skipped) == len(backend.calls) == 1
    assert skipped[0]["reason"] == reason
    assert skipped[0]["initial"]["text"] == text
    assert not read_lines(output / "checkpoints.jsonl")
    assert not read_lines(output / "outcomes.jsonl")
    assert summary["skipped_problems"] == 1
    assert summary["actual_shared_prefix_tokens"] == len(text)
    assert summary["generation_finish_reasons"] == {finish: 1}
    assert analyzer.analyze(output, draws=100)["enrollment"]["skipped_problems"] == 1


def test_checkpoint_at_cap_is_eligible(prepared):
    backend, _, output, argv, _ = prepared
    backend.initial_scripts = [("x" * 383 + "\n", "length")]
    mechanism.main(argv)
    cp = read_lines(output / "checkpoints.jsonl")[0]
    assert len(cp["retained_ids"]) == 384
    assert cp["initial"]["finish_reason"] == "checkpoint"


def test_checkpoint_preserves_noncanonical_ids_and_coherent_paragraph_state():
    backend = FakeBackend()
    ids = [5000, 10] + [ord("x")] * 253 + [10]
    initial = Generation(ids, backend.decode(ids), 1, len(ids), -1, 2, .1, "checkpoint", 0, "", 1,
                         [-1] * 256, [2] * 256)
    cp, reason = mechanism.checkpoint_from_initial(backend, TASK, [99], initial, 256, 384)
    assert reason is None
    assert cp["retained_ids"] == ids
    assert cp["state"]["history"] == ""
    assert cp["state"]["latest_segment"] == "COMPOUND_TOKEN\n" + "x" * 253


def test_state_partition_covers_retained_paragraphs_without_future_text():
    backend = FakeBackend()
    text = "Previous paragraph.\n\nEarlier calculations.\n\n" + "x" * 204 + "\nClosing line\n"
    ids = backend.encode_text(text)
    initial = Generation(ids, text, 1, len(ids), -1, 2, .1, "checkpoint", 0, "", 1,
                         [-1] * len(ids), [2] * len(ids))
    cp, reason = mechanism.checkpoint_from_initial(backend, TASK, [99], initial, 256, 384)
    assert reason is None
    assert cp["state"]["history"] == "Previous paragraph.\n\nEarlier calculations."
    assert cp["state"]["latest_segment"] == "x" * 204 + "\nClosing line"
    assert cp["state"]["history"] + "\n\n" + cp["state"]["latest_segment"] == text.rstrip()


def test_disjoint_shards_keep_original_tasks_and_seed_schedule():
    def plan(start, count):
        return mechanism.planned_tasks(mechanism.parse_args([
            "--model", "unused", "--output", "/nonexistent-mechanism-test-output",
            "--start-index", str(start), "--problems", str(count)]))

    full, left, right = plan(0, 24), plan(0, 12), plan(12, 12)
    assert full == left + right
    assert {item["task"]["id"] for item in left}.isdisjoint(item["task"]["id"] for item in right)
    assert len({entry["seed"] for item in full for entry in item["schedule"]}) == 288
    for item in full:
        if item["task"]["family"] == "arithmetic_construction":
            assert len(item["task"]["data"]["numbers"]) == 6
        assert item["task"]["prompt"].endswith(mechanism.PROMPT_SUFFIX)


def test_zero_quantization_option_does_not_requantize(prepared):
    _, _, output, argv, loaded = prepared
    mechanism.main(argv + ["--quantize-bits", "0"])
    assert loaded[0][1] == {"quantization_bits": None}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["quantize_bits"] == 0
    assert manifest["actual_quantization_config"]["bits"] == 4


def test_jev_is_one_batched_request_per_checkpoint_on_central_stage_cap(prepared, monkeypatch):
    _, _, output, argv, _ = prepared
    requests, constructed = [], []

    class FakeJev:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def evaluate(self, state, questions):
            requests.append((state, questions))
            assert not (output / "outcomes.jsonl").read_text()
            return {"response": {"answers": {key: {"noul": .5} for key in questions}}}

        def status(self):
            return {"accounted_usd": 0.0, "stage_cap_usd": 1.0}

    monkeypatch.setattr(jev_module, "JevClient", FakeJev)
    mechanism.main(argv + ["--jev"])
    assert constructed == [{"stage_cap_usd": 1.0}]
    assert len(requests) == 1
    assert requests[0][1] == QUESTIONS
    assert set(requests[0][0]) == {"task", "history", "latest_segment"}
    assert "FUTURE" not in json.dumps(requests[0])
    assert "jev" in read_lines(output / "checkpoints.jsonl")[0]


def test_jev_disabled_never_constructs_client(prepared, monkeypatch):
    _, _, _, argv, _ = prepared
    monkeypatch.setattr(jev_module, "JevClient", lambda **_: pytest.fail("Unrequested Jev call"))
    mechanism.main(argv)


def test_jev_failure_keeps_checkpoint_without_retry_or_arms(prepared, monkeypatch):
    backend, _, output, argv, _ = prepared
    requests = []

    class FailingJev:
        def __init__(self, **kwargs):
            pass

        def evaluate(self, *args):
            requests.append(args)
            raise RuntimeError("simulated service failure")

        def status(self):
            return {"accounted_usd": .01}

    monkeypatch.setattr(jev_module, "JevClient", FailingJev)
    with pytest.raises(RuntimeError, match="simulated service failure"):
        mechanism.main(argv + ["--jev"])
    assert len(requests) == len(backend.calls) == 1
    cp = read_lines(output / "checkpoints.jsonl")[0]
    assert cp["jev_error"] == {"type": "RuntimeError", "retry": False}
    assert read_lines(output / "checkpoint_attempts.jsonl")[0]["retained_ids"] == cp["retained_ids"]
    assert not read_lines(output / "outcomes.jsonl")
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"


def test_failed_mid_action_retains_completed_call_ledger(prepared):
    backend, _, output, argv, _ = prepared
    backend.fail_call = 3
    with pytest.raises(RuntimeError, match="fake failure"):
        mechanism.main(argv)
    events = read_lines(output / "generation_events.jsonl")
    assert [event["status"] for event in events] == ["complete", "complete", "error"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["durable_completed_call_generated_tokens"] == 256 + len("FINAL: A->B")
    assert summary["failed_calls_with_unknown_work"] == 1
    assert read_lines(output / "checkpoints.jsonl")


def test_expired_deadline_never_starts_a_generation(tmp_path, monkeypatch):
    backend = FakeBackend()
    wrapper = mechanism.LoggedBackend(backend, tmp_path, deadline=10)
    monkeypatch.setattr(mechanism.time, "monotonic", lambda: 11)
    with pytest.raises(mechanism.WalltimeExceeded):
        wrapper.generate([1], 100, 1)
    assert not backend.calls
    assert not wrapper.events


def test_call_timeout_is_capped_by_remaining_walltime(tmp_path, monkeypatch):
    backend = FakeBackend()
    wrapper = mechanism.LoggedBackend(backend, tmp_path, deadline=20)
    monkeypatch.setattr(mechanism.time, "monotonic", lambda: 11)
    wrapper.generate([1], 100, 1, timeout_s=180)
    assert backend.calls[0]["timeout_s"] == 9


def test_generation_start_is_durable_before_entering_backend(tmp_path):
    backend = FakeBackend()
    wrapper = mechanism.LoggedBackend(backend, tmp_path, deadline=float("inf"))

    def interrupted(*args, **kwargs):
        starts = read_lines(tmp_path / "generation_started.jsonl")
        assert starts[0]["sequence"] == 0
        assert starts[0]["prefix_ids"] == [1, 2]
        assert not (tmp_path / "generation_events.jsonl").exists()
        raise KeyboardInterrupt

    backend.generate = interrupted
    with pytest.raises(KeyboardInterrupt):
        wrapper.generate([1, 2], 100, 1)
    assert read_lines(tmp_path / "generation_events.jsonl")[0]["generated_work_unknown"]


def test_weight_revision_changes_with_contents_even_when_filename_is_same(prepared):
    _, model, _, _, _ = prepared
    first = mechanism.model_identity(model)
    (model / "model.safetensors").write_bytes(b"changed fake weights")
    second = mechanism.model_identity(model)
    assert first["weight_revision_sha256"] != second["weight_revision_sha256"]
    assert first["file_sha256"]["config.json"] == second["file_sha256"]["config.json"]


@pytest.mark.parametrize("extra", [
    ["--start-index", "-1"], ["--start-index", "24"], ["--problems", "0"],
    ["--start-index", "12", "--problems", "13"], ["--repeats", "3"],
    ["--budget", "2048"], ["--checkpoint-target", "128"], ["--checkpoint-cap", "512"],
    ["--max-walltime-seconds", "nan"], ["--max-walltime-seconds", "0"],
])
def test_bounds_reject_before_creating_output(prepared, extra):
    backend, _, output, argv, loaded = prepared
    with pytest.raises(SystemExit) as exc:
        mechanism.main(argv + extra)
    assert exc.value.code == 2
    assert not output.exists()
    assert not loaded and not backend.calls


def test_existing_output_is_preserved(prepared):
    _, _, output, argv, loaded = prepared
    output.mkdir()
    (output / "important").write_text("partial run")
    with pytest.raises(SystemExit):
        mechanism.main(argv)
    assert (output / "important").read_text() == "partial run"
    assert not loaded


def test_config_and_default_protocol_match():
    args = mechanism.parse_args(["--model", "unused", "--output", "/nonexistent-mechanism-test-output"])
    config = json.loads(mechanism.CONFIG.read_text())
    for key in ("seed", "start_index", "problems", "repeats", "budget", "checkpoint_target",
                "checkpoint_cap", "final_reserve", "quantize_bits", "max_walltime_seconds"):
        assert config[key] == getattr(args, key)
    assert config["actions"] == list(mechanism.ACTIONS)
    assert config["state_segmentation"] == mechanism.STATE_SEGMENTATION


@pytest.mark.parametrize("action", mechanism.ACTIONS)
def test_inherited_rollout_never_resumes_after_a_timeout(action):
    backend = FakeBackend()
    backend.continuation_finish = "timeout"
    row = mechanism.rollout(backend, TASK, [5000], list(map(ord, "x\n")), action, 123, 768, 96)
    assert len(backend.calls) == (2 if action == "branch" else 1)
    assert all(call["finish_reason"] == "timeout" for call in row["calls"])
