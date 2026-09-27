"""CPU-only protocol checks for the prospective development diagnostic."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from jev_control.mlx_backend import Generation


SCRIPTS = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("run_development", SCRIPTS / "run_development.py")
development = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(development)
finally:
    sys.path.remove(str(SCRIPTS))


TASK = {"id": "test-path", "family": "weighted_path",
        "data": {"n": 2, "edges": [[0, 1, 1]]}}


class FakeBackend:
    """Token scripts with online stopping and deliberately noncanonical IDs."""
    pieces = {1000: "cost=5", 1001: "FUTURE_CALCULATION_2+2=4"}

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.encoded = []

    def encode_text(self, text):
        self.encoded.append(text)
        return list(map(ord, text))

    def decode(self, ids):
        return "".join(self.pieces.get(i, chr(i)) for i in ids)

    def generate(self, prefix, max_tokens, seed, timeout_s, stop_when=None):
        self.calls.append({"prefix": list(prefix), "max_tokens": max_tokens,
                           "seed": seed, "stop_when": stop_when})
        scripted, finish = self.responses.pop(0)
        scripted = list(map(ord, scripted)) if isinstance(scripted, str) else scripted
        ids = []
        for token in scripted[:max_tokens]:
            ids.append(token)
            terminal = len(ids) == len(scripted) and finish in {"stop", "timeout"}
            if terminal:
                break
            if stop_when is not None and stop_when(list(ids)):
                finish = "checkpoint"
                break
        return Generation(ids, self.decode(ids), len(prefix), len(ids), -1., 1.,
                          .25, finish, 0., "fake", seed)


def run(backend, condition="sham", **kwargs):
    return development.episode(backend, TASK, [71, 72], condition, 20,
                               **{"budget": 40, "final_reserve": 12,
                                  "checkpoint_target": 3, "checkpoint_cap": 8, **kwargs})


def test_sham_resumes_exact_ids_and_charges_all_calls_without_losing_tail():
    backend = FakeBackend([([1000, 32, 10, 1001], "length"),
                           ("continued", "length"), ("A->B", "stop")])
    row = run(backend)
    retained = [1000, 32, 10]
    assert backend.calls[1]["prefix"] == [71, 72] + retained
    assert backend.calls[1]["max_tokens"] == 25
    assert [call["seed"] for call in backend.calls] == [20, 21, 22]
    assert backend.calls[2]["prefix"] == ([71, 72] + retained + list(map(ord, "continued"))
                                          + list(map(ord, development.FINAL)))
    assert row["generated_tokens"] == 3 + len("continued") + len("A->B")
    assert backend.calls[2]["max_tokens"] == 40 - 3 - len("continued")
    assert row["prompt_tokens_processed"] == sum(len(call["prefix"]) for call in backend.calls)
    assert row["elapsed_seconds"] == .75
    assert row["checkpoint"]["token_ids"] == retained
    assert row["checkpoint"]["prefix_sha256"] == hashlib.sha256(b"[71,72,1000,32,10]").hexdigest()
    assert row["checkpoint_status"] == "eligible"
    assert row["checkpoint"]["calculation_marker"]
    assert "FUTURE" not in row["text"]
    assert "continued" in row["text"]
    assert row["outcome"]["success"]
    assert backend.encoded == [development.FINAL]
    assert row["overhead"] == [{"kind": "finalization_instruction", "tokens": len(development.FINAL)}]


def test_checkpoint_label_cannot_see_future_calculation_or_final_answer():
    backend = FakeBackend([("abc\n2+2=4\nFINAL: A->B", "length"), ("FINAL: A->B", "stop")])
    row = run(backend)
    assert row["checkpoint"]["text"] == "abc\n"
    assert not row["checkpoint"]["calculation_marker"]
    assert row["checkpoint_status"] == "eligible"
    assert row["outcome"]["success"]


def test_default_1024_token_budget_and_reserve_match_between_conditions():
    for condition, responses in [
        ("baseline", [("x" * 928, "length"), ("A->B", "stop")]),
        ("sham", [("x" * 255 + "\nignored", "length"), ("y" * 672, "length"), ("A->B", "stop")]),
    ]:
        backend = FakeBackend(responses)
        row = development.episode(backend, TASK, [1], condition, 10)
        assert backend.calls[-1]["max_tokens"] == 96
        assert row["generated_tokens"] == 932
        assert row["outcome"]["success"]
        if condition == "sham":
            assert row["checkpoint"]["position"] == 256
        else:
            assert len(backend.calls) == 2
            assert backend.calls[0]["stop_when"] is None
            assert row["checkpoint_status"] == "not_requested"


@pytest.mark.parametrize("condition", ["baseline", "sham"])
@pytest.mark.parametrize("finish", ["stop", "timeout"])
def test_early_eos_or_timeout_never_resumes_or_injects(condition, finish):
    backend = FakeBackend([("hi", finish)])
    row = run(backend, condition)
    assert len(backend.calls) == 1
    assert not backend.encoded
    assert row["generated_tokens"] == 2
    expected = "not_requested" if condition == "baseline" else "completed_early" if finish == "stop" else "timeout"
    assert row["checkpoint_status"] == expected


@pytest.mark.parametrize("finish", ["stop", "timeout"])
def test_termination_after_checkpoint_prevents_final_reserve_call(finish):
    backend = FakeBackend([("abc\n", "length"), ("done", finish)])
    row = run(backend)
    assert len(backend.calls) == 2
    assert not backend.encoded
    assert row["generated_tokens"] == 8
    assert row["checkpoint_status"] == "eligible"


@pytest.mark.parametrize("initial,status", [
    ("abcdefgh", "no_boundary_before_cap"),
    ("FINAL:\nX", "answer_phase_before_checkpoint"),
])
def test_no_eligible_boundary_still_preserves_all_emitted_ids(initial, status):
    backend = FakeBackend([(initial, "length"), ("tail", "stop")])
    row = run(backend)
    assert row["checkpoint"] is None
    assert row["checkpoint_status"] == status
    assert backend.calls[1]["prefix"] == [71, 72] + list(map(ord, initial))
    assert row["text"] == initial + "tail"
    assert row["generated_tokens"] == len(initial + "tail")


@pytest.mark.parametrize("condition", ["baseline", "sham"])
def test_partial_final_marker_is_continued_without_instruction_injection(condition):
    responses = ([("FINAL: A->", "length"), ("B", "stop")] if condition == "baseline"
                 else [("abc\n", "length"), ("FINAL: A->", "length"), ("B", "stop")])
    backend = FakeBackend(responses)
    row = run(backend, condition)
    assert not backend.encoded
    assert not row["overhead"]
    assert row["outcome"]["success"]


@pytest.mark.parametrize("text,minimum,expected", [
    ("x\n", 3, False), ("ab\n", 3, True), ("a\nb", 3, False),
    ("FINAL:\n", 3, False), ("xyz", 3, False),
])
def test_boundary_is_current_eligible_newline_only(text, minimum, expected):
    assert development.boundary_now(FakeBackend([]), list(map(ord, text)), minimum) is expected


@pytest.mark.parametrize("kwargs", [
    {"condition": "jev"}, {"checkpoint_target": 0}, {"checkpoint_target": 9},
    {"checkpoint_cap": 28}, {"final_reserve": -1}, {"budget": 20, "final_reserve": 12},
])
def test_invalid_episode_envelopes_reject_before_generation(kwargs):
    backend = FakeBackend([])
    with pytest.raises(ValueError):
        run(backend, **kwargs)
    assert not backend.calls


@pytest.mark.parametrize("option,value", [
    ("--problems", "0"), ("--problems", "25"), ("--repeats", "0"),
    ("--repeats", "5"), ("--budget", "480"), ("--checkpoint-target", "385"),
    ("--final-reserve", "-1"),
])
def test_cli_bounds_fail_before_output_creation(tmp_path, monkeypatch, option, value):
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_development", "--model", "unused", "--output", str(output), option, value])
    monkeypatch.setattr(development, "MLXBackend", lambda *a, **k: pytest.fail("Invalid args loaded model"))
    with pytest.raises(SystemExit) as exc:
        development.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_existing_run_is_preserved(tmp_path, monkeypatch):
    sentinel = tmp_path / "outcomes.jsonl"
    sentinel.write_text("partial run\n")
    monkeypatch.setattr(sys, "argv", ["run_development", "--model", "unused", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        development.main()
    assert exc.value.code == 2
    assert sentinel.read_text() == "partial run\n"


def test_manifest_and_schedule_record_fresh_seeded_development_design(tmp_path, monkeypatch):
    output = tmp_path / "run"
    backend = FakeBackend([("x", "stop")] * 48)
    backend.encode_chat = lambda messages: backend.encode_text(messages[0]["content"])
    monkeypatch.setattr(development, "MLXBackend", lambda *a, **k: backend)
    monkeypatch.setattr(development.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(sys, "argv", ["run_development", "--model", "local-snapshot", "--output", str(output)])
    development.main()
    manifest = json.loads((output / "manifest.json").read_text())
    tasks = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    assert (manifest["problems"], manifest["repeats"], manifest["budget"]) == (12, 2, 1024)
    assert (manifest["checkpoint_target"], manifest["checkpoint_cap"], manifest["final_reserve"]) == (256, 384, 96)
    assert manifest["task_seed"] == 391027
    assert manifest["sampling_seed"] == 271027
    assert len(tasks) == len({task["task"]["id"] for task in tasks}) == 12
    assert len(rows) == len(backend.calls) == 48
    for index, item in enumerate(tasks):
        assert item["task"]["prompt"].endswith(development.PROMPT_SUFFIX)
        subset = [row for row in rows if row["problem_id"] == item["task"]["id"]]
        assert {(r["condition"], r["repeat"], r["seed"]) for r in subset} == {
            (condition, repeat, 271027 + index * 1000 + repeat * 10)
            for condition in ("baseline", "sham") for repeat in (0, 1)
        }
    for path, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((output / "source" / Path(path).name).read_bytes()).hexdigest() == expected
    assert summary["episodes"] == 48
    assert summary["independent_problems"] == 12
    assert summary["conditions"]["baseline"]["n"] == summary["conditions"]["sham"]["n"] == 24


def test_summary_counts_problems_once_and_keeps_conditions_and_families_separate():
    def row(condition, family, problem, success, tokens, status):
        return {"condition": condition, "family": family, "problem_id": problem,
                "outcome": {"success": success}, "generated_tokens": tokens,
                "elapsed_seconds": .5, "checkpoint_status": status,
                "checkpoint": {"calculation_marker": True} if status == "eligible" else None}

    result = development.summarize([
        row("baseline", "path", "p1", True, 10, "not_requested"),
        row("baseline", "path", "p1", False, 20, "not_requested"),
        row("sham", "path", "p1", True, 30, "eligible"),
        row("sham", "arithmetic", "p2", False, 40, "timeout"),
    ])
    assert result["episodes"] == 4
    assert result["independent_problems"] == 2
    assert result["conditions"]["baseline"]["generated_tokens"] == 30
    assert result["conditions"]["sham"]["generated_tokens"] == 70
    assert result["conditions"]["sham"]["successes"] == 1
    assert result["conditions"]["sham"]["checkpoint_status"] == {"eligible": 1, "timeout": 1}
    assert result["conditions"]["sham"]["calculation_marker_checkpoints"] == 1
    assert result["conditions"]["sham"]["families"] == {
        "arithmetic": {"n": 1, "successes": 0}, "path": {"n": 1, "successes": 1}}
    assert "Not an equivalence test" in result["warning"]
