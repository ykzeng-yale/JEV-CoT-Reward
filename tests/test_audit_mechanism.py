"""Completed-run integrity audits using synthetic CPU-only generation records."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from jev_control.mlx_backend import Generation


ROOT = Path(__file__).parents[1]
MODEL_CONFIG = {"model_type": "audit_fixture", "quantization": {
    "bits": 4, "group_size": 64, "mode": "affine"}}
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import run_mechanism as mechanism
    import audit_mechanism as auditor
finally:
    sys.path.remove(str(ROOT / "scripts"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prefix_hash(ids):
    return digest(json.dumps(ids, separators=(",", ":")).encode())


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


class Tokenizer:
    """A deterministic tokenizer, including a noncanonical chat prefix token."""

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join("<assistant>" if token == 5000 else chr(token) for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        return [5000] + self.encode(messages[0]["content"])


def correct_answer(item, seed):
    task, index = item["task"], item["index"]
    if task["family"] == "arithmetic_construction":
        rng = random.Random(seed + index)
        a, b, c, d, e, f = [rng.randint(1, 12) for _ in range(6)]
        return f"({a}+{b})*{c}+{d}*{e}-{f}"
    best = {0: (0, [0])}
    for a, b, weight in task["data"]["edges"]:
        if a in best and (b not in best or best[a][0] + weight < best[b][0]):
            best[b] = best[a][0] + weight, best[a][1] + [b]
    return "->".join(chr(65 + node) for node in best[task["data"]["n"] - 1][1])


def response(text, finish="stop", logprob=-1.0, *, fill=False, suffix=""):
    return {"text": text, "finish": finish, "logprob": logprob,
            "fill": fill, "suffix": suffix}


def scripted_responses(plan, seed, initial_modes):
    responses = []
    for item, initial_mode in zip(plan, initial_modes):
        answer = correct_answer(item, seed)
        initial = {
            "checkpoint": response("x" * 255 + "\n", "length"),
            "at_cap": response("x" * 383 + "\n", "length"),
            "stop": response("FINAL: " + answer),
            "timeout": response("Still working", "timeout"),
            "final": response("FINAL: unfinished ", "length", fill=True),
            "no_boundary": response("x", "length", fill=True),
        }[initial_mode]
        responses.append(initial)
        if initial_mode not in {"checkpoint", "at_cap"}:
            continue
        for entry in item["schedule"]:
            action, repeat = entry["action"], entry["repeat"]
            if action != "branch":
                if repeat == 0:
                    responses.append(response("\nFINAL: " + answer))
                elif repeat == 1:
                    responses.extend([response("calculating ", "length", fill=True),
                                      response(answer)])
                elif repeat == 2:
                    responses.extend([response("checking ", "length", fill=True,
                                               suffix="\nFINAL: "), response(answer)])
                else:
                    responses.append(response("Unfinished reasoning", "timeout"))
            elif repeat == 0:
                responses.extend([response("loser ", "length", -2., fill=True),
                                  response("\nFINAL: " + answer, "stop", -.5)])
            elif repeat == 1:
                responses.extend([response("winner ", "length", -.5, fill=True),
                                  response("loser ", "length", -2., fill=True),
                                  response("calculating ", "length", fill=True),
                                  response(answer)])
            elif repeat == 2:
                responses.extend([response("first tie ", "length", -1., fill=True),
                                  response("second tie ", "length", -1., fill=True),
                                  response("\nFINAL: " + answer)])
            else:
                responses.extend([response("Unfinished winner", "timeout", -.5),
                                  response("\nFINAL: " + answer, "stop", -2.)])
    return responses


class FakeBackend(Tokenizer):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model_config = deepcopy(MODEL_CONFIG)
        self.quantization_config = self.model_config["quantization"]

    encode_text = Tokenizer.encode
    encode_chat = Tokenizer.apply_chat_template

    @staticmethod
    def choose_candidate(candidates):
        return max(range(len(candidates)), key=lambda index:
                   candidates[index].mean_logprob if candidates[index].mean_logprob is not None
                   and math.isfinite(candidates[index].mean_logprob) else -math.inf)

    def generate(self, prefix, max_tokens, seed, timeout_s, stop_when=None):
        spec = self.responses.pop(0)
        text, finish = spec["text"], spec["finish"]
        if spec["fill"]:
            body_size = max_tokens - len(spec["suffix"])
            text = (text * (max_tokens + 1))[:body_size] + spec["suffix"]
        scripted = self.encode(text)
        ids = []
        for token in scripted[:max_tokens]:
            ids.append(token)
            if len(ids) == len(scripted) and finish == "stop":
                break
            if stop_when is not None and stop_when(list(ids)):
                finish = "checkpoint"
                break
            if len(ids) == len(scripted) and finish == "timeout":
                break
        logprobs = [spec["logprob"]] * len(ids)
        entropies = [2.] * len(ids)
        result = Generation(ids, self.decode(ids), len(prefix), len(ids),
                            sum(logprobs) / len(ids), sum(entropies) / len(ids),
                            .125, finish, .01, prefix_hash(prefix), seed,
                            token_logprobs=logprobs, token_entropies=entropies)
        self.calls.append(result)
        return result


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    """Freeze real runner sources and tiny model metadata without loading a model."""
    model = tmp_path / "snapshots" / "fixture-revision"
    model.mkdir(parents=True)
    write_json(model / "config.json", MODEL_CONFIG)
    write_json(model / "tokenizer.json", {"fixture": "character tokenizer"})
    (model / "model.safetensors").write_bytes(b"synthetic test fixture; never loaded")
    tokenizer = Tokenizer()
    tokenizer.audit_sha256 = digest((model / "tokenizer.json").read_bytes())
    monkeypatch.setattr(mechanism.importlib.metadata, "version", lambda _: "fixture-version")
    serial = 0

    def build(initial_modes=("checkpoint", "checkpoint")):
        nonlocal serial
        output = tmp_path / f"run-{serial}"
        serial += 1
        argv = ["--model", str(model), "--output", str(output),
                "--problems", str(len(initial_modes)), "--quantize-bits", "0"]
        args = mechanism.parse_args(argv)
        plan = mechanism.planned_tasks(args)
        backend = FakeBackend(scripted_responses(plan, args.seed, initial_modes))
        monkeypatch.setattr(mechanism, "MLXBackend", lambda *a, **kw: backend)
        mechanism.main(argv)
        assert backend.responses == []
        return SimpleNamespace(path=output, model=model, tokenizer=tokenizer, backend=backend)

    return build


@pytest.fixture
def completed(make_run):
    return make_run()


def audit(run):
    return auditor.audit(run.path, tokenizer_dir=run.model, tokenizer=run.tokenizer)


def rows(run):
    return read_lines(run.path / "outcomes.jsonl")


def reject(run):
    with pytest.raises(ValueError):
        audit(run)


def test_frozen_old_analyzer_is_hashed_but_never_executed(completed):
    name = "scripts/analyze_screen.py"
    source = completed.path / "source" / name
    source.write_text('raise RuntimeError("A saved analyzer is evidence, not executable audit code")\n')
    manifest = read_json(completed.path / "manifest.json")
    manifest["source_sha256"][name] = digest(source.read_bytes())
    write_json(completed.path / "manifest.json", manifest)
    result = audit(completed)
    assert result["ready_for_statistical_analysis"]
    assert result["provenance"]["launch_source_sha256"][name] == digest(source.read_bytes())


def test_current_task_functions_cannot_replace_frozen_task_evidence(completed, monkeypatch):
    import analyze_development
    import jev_control.tasks

    def unexpected(*args, **kwargs):
        pytest.fail("Audit invoked a current task function instead of approved frozen source")

    monkeypatch.setattr(analyze_development, "make_task", unexpected)
    monkeypatch.setattr(analyze_development, "verify", unexpected)
    monkeypatch.setattr(jev_control.tasks, "make_task", unexpected)
    monkeypatch.setattr(jev_control.tasks, "verify", unexpected)
    assert audit(completed)["ready_for_statistical_analysis"]


def mutate_call(run, action, repeat, call_index, mutation):
    """Keep duplicated call/start ledgers coherent to test semantic reconstruction."""
    outcomes = rows(run)
    row = next(row for row in outcomes if row["index"] == 0
               and row["action"] == action and row["repeat"] == repeat)
    events = read_lines(run.path / "generation_events.jsonl")
    selected = [event for event in events if event.get("phase") == "continuation"
                and event["problem_id"] == row["problem_id"]
                and event["action"] == action and event["repeat"] == repeat]
    event = selected[call_index]
    mutation(event)
    row["calls"][call_index] = deepcopy(event["generation"])
    starts = read_lines(run.path / "generation_started.jsonl")
    start = next(start for start in starts if start["sequence"] == event["sequence"])
    for key in ("prefix_ids", "seed", "max_tokens"):
        start[key] = deepcopy(event[key])
    write_lines(run.path / "outcomes.jsonl", outcomes)
    write_lines(run.path / "generation_events.jsonl", events)
    write_lines(run.path / "generation_started.jsonl", starts)


def test_completed_all_arm_run_reconstructs_every_call_and_charges_losers(completed):
    before = {str(p.relative_to(completed.path)): digest(p.read_bytes())
              for p in completed.path.rglob("*") if p.is_file()}
    result = audit(completed)
    generations = completed.backend.calls
    outcomes = rows(completed)
    loser_tokens = sum(row["calls"][1 - row["overhead"][0]["winner"]]["generated_tokens"]
                       for row in outcomes if row["action"] == "branch")
    assert result["status"] == "passed_integrity_audit"
    assert result["ready_for_statistical_analysis"] is True
    expected_counts = {"planned_problems": 2, "eligible_problems": 2,
                       "skipped_problems": 0, "episodes": 24,
                       "durable_calls": len(generations)}
    assert {key: result["counts"][key] for key in expected_counts} == expected_counts
    assert result["accounting"]["generated_tokens"] == sum(g.generated_tokens for g in generations)
    assert result["accounting"]["prompt_tokens_processed"] == sum(g.prompt_tokens for g in generations)
    assert result["accounting"]["elapsed_seconds"] == pytest.approx(sum(g.elapsed_seconds for g in generations))
    assert result["accounting"]["discarded_branch_generated_tokens"] == loser_tokens > 0
    assert result["outcome_reverification"]["disagreement_count"] == 0
    assert result["outcome_reverification"]["examples"] == []
    assert {row["outcome"]["success"] for row in outcomes} == {True, False}
    assert {row["overhead"][0]["winner"] for row in outcomes if row["action"] == "branch"} == {0, 1}
    for row in outcomes:
        finalizations = [item for item in row["overhead"] if item["kind"] == "finalization_instruction"]
        if row["action"] != "branch":
            assert bool(finalizations) is (row["repeat"] == 1)
    assert before == {str(p.relative_to(completed.path)): digest(p.read_bytes())
                      for p in completed.path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("action,repeat,call_index", [
    ("continue", 0, 0), ("repair", 0, 0), ("branch", 1, 2),
    ("continue", 1, 1), ("continue", 2, 1), ("branch", 1, 3),
])
def test_exact_prefix_reconstruction_rejects_coherently_rehashed_forgery(completed, action, repeat, call_index):
    def change(event):
        event["prefix_ids"][-1] += 1
        event["generation"]["prefix_sha256"] = prefix_hash(event["prefix_ids"])
    mutate_call(completed, action, repeat, call_index, change)
    reject(completed)


@pytest.mark.parametrize("field,value", [("seed", 17), ("max_tokens", 1)])
def test_prespecified_call_seed_and_cap_are_reconstructed(completed, field, value):
    def change(event):
        event[field] += value
        if field == "seed":
            event["generation"]["seed"] = event[field]
    mutate_call(completed, "continue", 0, 0, change)
    reject(completed)


@pytest.mark.parametrize("field,value", [
    ("text", "forged decoded text"), ("generated_tokens", 1),
    ("prompt_tokens", 1), ("mean_logprob", -.25), ("mean_entropy", 3.),
    ("prefix_sha256", "0" * 64),
])
def test_generation_token_statistics_and_hash_are_checked(completed, field, value):
    mutate_call(completed, "continue", 0, 0,
                lambda event: event["generation"].__setitem__(field, value))
    reject(completed)


@pytest.mark.parametrize("repeat", [0, 2])
def test_branch_selection_checks_likelihood_and_first_candidate_tie(completed, repeat):
    outcomes = rows(completed)
    row = next(row for row in outcomes if row["action"] == "branch" and row["repeat"] == repeat)
    row["overhead"][0]["winner"] = 1 - row["overhead"][0]["winner"]
    write_lines(completed.path / "outcomes.jsonl", outcomes)
    reject(completed)


def test_discarded_candidate_work_cannot_be_removed_from_episode_cost(completed):
    outcomes = rows(completed)
    row = next(row for row in outcomes if row["action"] == "branch")
    loser = 1 - row["overhead"][0]["winner"]
    row["generated_tokens"] -= row["calls"][loser]["generated_tokens"]
    write_lines(completed.path / "outcomes.jsonl", outcomes)
    reject(completed)


@pytest.mark.parametrize("filename", ["generation_started.jsonl", "generation_events.jsonl", "outcomes.jsonl"])
def test_duplicate_durable_or_episode_records_are_rejected(completed, filename):
    path = completed.path / filename
    records = read_lines(path)
    write_lines(path, records + [records[-1]])
    reject(completed)


def test_unmatched_start_is_unknown_work_even_with_complete_manifest(completed):
    path = completed.path / "generation_started.jsonl"
    starts = read_lines(path)
    extra = deepcopy(starts[-1])
    extra["sequence"] += 1
    write_lines(path, starts + [extra])
    reject(completed)


def test_completed_event_requires_matching_durable_start(completed):
    path = completed.path / "generation_started.jsonl"
    write_lines(path, read_lines(path)[:-1])
    reject(completed)


@pytest.mark.parametrize("filename,status", [("manifest.json", "running"),
                                             ("manifest.json", "interrupted"),
                                             ("summary.json", "failed")])
def test_only_finalized_complete_runs_are_accepted(completed, filename, status):
    path = completed.path / filename
    record = read_json(path)
    record["status"] = status
    write_json(path, record)
    reject(completed)


@pytest.mark.parametrize("filename", ["summary.json", "checkpoint_attempts.jsonl"])
def test_missing_required_completion_artifact_is_rejected(completed, filename):
    (completed.path / filename).unlink()
    reject(completed)


@pytest.mark.parametrize("mode,reason", [
    ("stop", "completed_before_checkpoint"), ("timeout", "timeout_before_checkpoint"),
    ("final", "answer_phase_reached_before_checkpoint"), ("no_boundary", "no_boundary_before_cap"),
])
def test_skipped_problem_remains_enrolled_and_its_initial_work_is_charged(make_run, mode, reason):
    run = make_run(("checkpoint", mode))
    skipped = read_json(run.path / "skipped.json")
    assert skipped[0]["reason"] == reason
    result = audit(run)
    assert result["ready_for_statistical_analysis"] is True
    assert result["counts"]["planned_problems"] == 2
    assert result["counts"]["eligible_problems"] == result["counts"]["skipped_problems"] == 1
    assert result["counts"]["episodes"] == 12
    assert result["accounting"]["generated_tokens"] == sum(g.generated_tokens for g in run.backend.calls)


def test_checkpoint_at_exact_cap_is_eligible(make_run):
    run = make_run(("at_cap",))
    assert len(read_lines(run.path / "checkpoints.jsonl")[0]["retained_ids"]) == 384
    assert audit(run)["counts"]["eligible_problems"] == 1


@pytest.mark.parametrize("field,value", [("state", {"task": "leaked future", "history": "", "latest_segment": ""}),
                                        ("discarded_prefix_tail_tokens", 1),
                                        ("retained_features", {"retained_stat_token_count": 0,
                                                               "retained_mean_logprob": None,
                                                               "retained_mean_entropy": None})])
def test_checkpoint_content_is_derived_only_from_actual_initial_emissions(completed, field, value):
    for filename in ("checkpoints.jsonl", "checkpoint_attempts.jsonl"):
        records = read_lines(completed.path / filename)
        records[0][field] = value
        write_lines(completed.path / filename, records)
    reject(completed)


def test_skip_reason_cannot_be_relabeled(make_run):
    run = make_run(("timeout",))
    skipped = read_json(run.path / "skipped.json")
    skipped[0]["reason"] = "completed_before_checkpoint"
    write_json(run.path / "skipped.json", skipped)
    reject(run)


@pytest.mark.parametrize("filename", ["source/scripts/run_screen.py", "schedule.json"])
def test_recorded_source_and_schedule_hashes_detect_tampering(completed, filename):
    path = completed.path / filename
    path.write_bytes(path.read_bytes() + b"\n")
    reject(completed)


def test_resealed_source_must_still_match_recognized_rollout_semantics(completed):
    relative = "scripts/run_screen.py"
    path = completed.path / "source" / relative
    source = path.read_text()
    assert "winner=backend.choose_candidate(candidates)" in source
    path.write_text(source.replace("winner=backend.choose_candidate(candidates)", "winner=0"))
    manifest = read_json(completed.path / "manifest.json")
    manifest["source_sha256"][relative] = digest(path.read_bytes())
    write_json(completed.path / "manifest.json", manifest)
    reject(completed)


def test_resealed_schedule_still_must_follow_prespecified_assignment(completed):
    path = completed.path / "schedule.json"
    plan = read_json(path)
    plan[0]["schedule"][0]["seed"] += 1
    write_json(path, plan)
    manifest = read_json(completed.path / "manifest.json")
    manifest["schedule_sha256"] = digest(path.read_bytes())
    write_json(completed.path / "manifest.json", manifest)
    reject(completed)


def test_completed_status_does_not_excuse_missing_scheduled_episode(completed):
    write_lines(completed.path / "outcomes.jsonl", rows(completed)[:-1])
    reject(completed)


def test_tokenizer_asset_must_match_pinned_model_identity(completed):
    (completed.model / "tokenizer.json").write_text('{"fixture":"changed tokenizer"}\n')
    reject(completed)


def test_original_outcome_disagreement_is_reported_without_rewriting_records(completed):
    outcomes = rows(completed)
    original = outcomes[0]["outcome"]["success"]
    outcomes[0]["outcome"]["success"] = not original
    write_lines(completed.path / "outcomes.jsonl", outcomes)
    summary = read_json(completed.path / "summary.json")
    summary["actions"][outcomes[0]["action"]]["successes"] += -1 if original else 1
    write_json(completed.path / "summary.json", summary)
    before = (completed.path / "outcomes.jsonl").read_bytes()
    result = audit(completed)
    assert result["ready_for_statistical_analysis"] is False
    assert result["outcome_reverification"]["disagreement_count"] == 1
    assert len(result["outcome_reverification"]["examples"]) == 1
    assert (completed.path / "outcomes.jsonl").read_bytes() == before


def test_cli_writes_report_using_requested_local_tokenizer(completed, monkeypatch):
    output = completed.path.parent / "audit-report.json"
    original_audit = auditor.audit
    observed = []

    def injected(run, *, tokenizer_dir=None, tokenizer=None):
        observed.append((Path(run), Path(tokenizer_dir)))
        return original_audit(run, tokenizer_dir=tokenizer_dir, tokenizer=completed.tokenizer)

    monkeypatch.setattr(auditor, "audit", injected)
    monkeypatch.setattr(sys, "argv", ["audit_mechanism", str(completed.path),
                                     "--output", str(output), "--tokenizer", str(completed.model)])
    auditor.main()
    assert observed == [(completed.path, completed.model)]
    result = read_json(output)
    assert result["status"] == "passed_integrity_audit"
    assert result["ready_for_statistical_analysis"] is True
