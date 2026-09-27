"""Synthetic analysis/provenance audits; never read measured runs or load models."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from jev_control.features import QUESTIONS
from jev_control.mlx_backend import Generation

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    def load(name):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    analysis = load("analyze_screen")
    judge = load("local_judge")
finally:
    sys.path.remove(str(SCRIPTS))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def lines(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_integrity_gate_rejects_stale_data_and_incomplete_audits(tmp_path):
    names = ("manifest.json", "summary.json", "schedule.json", "rubric.json", "checkpoint_attempts.jsonl",
             "checkpoints.jsonl", "skipped.json", "outcomes.jsonl", "generation_started.jsonl", "generation_events.jsonl")
    for name in names:
        (tmp_path / name).write_text("{}\n")
    report = {"status": "passed_integrity_audit", "ready_for_statistical_analysis": True,
              "outcome_reverification": {"disagreement_count": 0}, "accounting": {"unknown_work_calls": 0},
              "provenance": {"input_sha256": {name: digest(tmp_path / name) for name in names}}}
    target = tmp_path / "audit.json"
    write(target, report)
    assert analysis.validate_integrity_report(tmp_path, target) == digest(target)
    (tmp_path / "outcomes.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="does not match"):
        analysis.validate_integrity_report(tmp_path, target)
    (tmp_path / "outcomes.jsonl").write_text("{}\n")
    report["provenance"]["input_sha256"].pop("rubric.json")
    write(target, report)
    with pytest.raises(ValueError, match="incomplete"):
        analysis.validate_integrity_report(tmp_path, target)
    report["ready_for_statistical_analysis"] = False
    write(target, report)
    with pytest.raises(ValueError, match="passed independent"):
        analysis.validate_integrity_report(tmp_path, target)


def make_run(path, n=24, mechanism=False):
    path.mkdir()
    cps, rows, plan, events = [], [], [], []
    for index in range(n):
        pid, family = f"p{index:02}", "left" if index % 2 == 0 else "right"
        task = {"id": pid, "family": family, "prompt": f"Observable {family} task", "data": {"GOLD_SECRET": 1}}
        initial = Generation([65, 10], "A\n", 2, 2, -1, 1, .1, "checkpoint", 0, "", 100 + index).to_dict()
        cp = {"problem_id": pid, "task": task, "prompt_ids": [7, 8], "retained_ids": [65, 10],
              "initial": initial, "discarded_prefix_tail_tokens": 0,
              "state": {"task": task["prompt"], "history": "Past work", "latest_segment": "Current step"}}
        cp["sha256"] = hashlib.sha256(json.dumps({"prompt": cp["prompt_ids"], "retained": cp["retained_ids"]}, sort_keys=True).encode()).hexdigest()
        cps.append(cp)
        events.append({"sequence": len(events), "phase": "initial", "prefix_ids": [7, 8], "status": "complete", "generation": initial})
        schedule = []
        for ai, action in enumerate(analysis.ACTIONS):
            for repeat in range(2):
                seed = 1000 + index * 100 + ai * 10 + repeat
                schedule.append({"action": action, "repeat": repeat, "seed": seed})
                calls = [Generation([1, 2], "FUTURE_SECRET", 4, 2, -1, 1, .2, "stop", 0, "", seed + k).to_dict()
                         for k in range(2 if action == "branch" else 1)]
                success = action == ("continue" if family == "left" else "repair")
                row = {"problem_id": pid, "family": family, "checkpoint_sha256": cp["sha256"], "action": action,
                       "repeat": repeat, "seed": seed, "outcome": {"success": success}, "text": "FUTURE_SECRET",
                       "generated_tokens": 2 * len(calls), "elapsed_seconds": .2 * len(calls),
                       "prompt_tokens_processed": 4 * len(calls), "calls": calls,
                       "overhead": [{"kind": "branch_selection", "winner": 0}] if action == "branch" else []}
                rows.append(row)
                for g in calls:
                    events.append({"sequence": len(events), "phase": "continuation", "prefix_ids": [7, 8, 65, 10], "status": "complete", "generation": g})
        plan.append({"index": index, "task": task, "initial_seed": 100 + index, "schedule": schedule})
    manifest = {"problems": n, "repeats": 2, "budget": 100, "seed": 99, "status": "complete"}
    summary = {"status": "complete", "episodes": len(rows), "eligible_problems": n}
    if mechanism:
        manifest["protocol"] = "mechanism-v1-online-emitted-checkpoint"
        (path / "source").mkdir()
        (path / "source/frozen.py").write_text("# synthetic frozen source\n")
        manifest["source_sha256"] = {"frozen.py": digest(path / "source/frozen.py")}
        write(path / "schedule.json", plan)
        write(path / "rubric.json", {"schema": "semantic-v1", "questions": QUESTIONS})
        manifest["schedule_sha256"] = digest(path / "schedule.json")
        manifest["rubric_sha256"] = digest(path / "rubric.json")
        lines(path / "generation_events.jsonl", events)
        lines(path / "generation_started.jsonl", [{"sequence": event["sequence"]} for event in events])
    write(path / "manifest.json", manifest)
    write(path / "summary.json", summary)
    write(path / "skipped.json", [])
    lines(path / "checkpoints.jsonl", cps)
    lines(path / "outcomes.jsonl", rows)
    return cps, rows, manifest


def test_grouped_fit_has_no_problem_leakage_and_family_baseline_controls_confounding(tmp_path, monkeypatch):
    cps, rows, _ = make_run(tmp_path / "run")
    seen = []

    def fake_fit(train_docs, test_docs, train_num, test_num, train_rows, train_ids, test_ids, alpha):
        assert set(train_ids).isdisjoint(test_ids)
        assert {row["problem_id"] for row in train_rows} == set(train_ids)
        assert len(train_rows) == 6 * len(train_ids)
        assert all("FUTURE_SECRET" not in doc and "GOLD_SECRET" not in doc for doc in train_docs + test_docs)
        seen.extend(test_ids)
        return np.zeros((len(test_ids), 3)), 1

    monkeypatch.setattr(analysis, "fit_predict", fake_fit)
    result = analysis.analyze(tmp_path / "run", learned=True, draws=100)
    fit = result["crossfit"]
    assert sorted(seen) == sorted(cp["problem_id"] for cp in cps)
    assert fit["policies"]["training_selected_family_constant"]["success"]["mean"] == 1
    assert fit["policies"]["training_selected_constant"]["success"]["mean"] < 1
    assert set(fit["family_stratified"]) == {"left", "right"}
    assert "cheap_tfidf_minus_training_selected_family_constant" in fit["family_stratified"]["left"]["paired_policy_differences"]
    for fold in fit["folds"]:
        assert fold["training_selected_family_constants"] == {"left": "continue", "right": "repair"}


def test_real_fit_handles_empty_vocabulary_and_all_missing_judge_columns():
    train_ids, test_ids = ["a", "b"], ["c"]
    rows = [{"problem_id": pid, "action": action, "outcome": {"success": pid == "a"}}
            for pid in train_ids for action in analysis.ACTIONS]
    prediction, vocabulary = analysis.fit_predict(["", ""], ["heldoutonly"], np.full((2, 3), np.nan),
                                                  np.array([[.9, .8, .7]]), rows, train_ids, test_ids, 10)
    assert vocabulary == 0
    assert prediction.shape == (1, 3)
    assert np.isfinite(prediction).all()


def test_observable_features_ignore_gold_outcomes_future_and_unretained_logits(tmp_path):
    cps, _, _ = make_run(tmp_path / "run", n=1)
    cp = cps[0]
    before = analysis.observable_features(cp, 100)
    cp["task"]["data"] = {"secret": "different gold"}
    cp["outcome"] = {"success": False}
    cp["future"] = "new future"
    after = analysis.observable_features(cp, 100)
    assert before[0] == after[0]
    np.testing.assert_equal(before[1], after[1])
    cp["discarded_prefix_tail_tokens"] = 10
    assert np.isnan(analysis.observable_features(cp, 100)[1][6:8]).all()
    cp["discarded_prefix_tail_tokens"] = 0
    cp["initial"]["generated_tokens"] = 12
    assert np.isnan(analysis.observable_features(cp, 100)[1][6:8]).all()


@pytest.mark.parametrize("record", [None, {}, {"response": None}, {"error": "malformed_schema", "probabilities": {key: .8 for key in QUESTIONS}}])
def test_missing_and_failed_features_are_missing_not_zero(record):
    assert np.isnan(analysis.semantic_features(record)).all()


def test_local_records_outside_run_do_not_inflate_costs(tmp_path):
    cps, _, _ = make_run(tmp_path / "run", n=1)
    result = analysis.analyze(tmp_path / "run", draws=10, local_features={
        cps[0]["problem_id"]: {"generation": {"generated_tokens": 5, "prompt_tokens": 10, "elapsed_seconds": 1}},
        "elsewhere": {"generation": {"generated_tokens": 999, "prompt_tokens": 999, "elapsed_seconds": 999}}})
    costs = result["costs"]["local_judge_acquisition"]
    assert costs["generated_tokens"] == 5
    assert costs["outside_run_records_excluded"] == 1
    assert "historical" in result["implementation"]["local_judge_audit"]["identity_policy"]


@pytest.mark.parametrize("mutation", ["source", "schedule", "checkpoint", "seed", "duplicate", "overlap"])
def test_mechanism_integrity_rejects_corrupt_inputs(tmp_path, mutation):
    run = tmp_path / "run"
    cps, rows, manifest = make_run(run, n=1, mechanism=True)
    if mutation == "source":
        (run / "source/frozen.py").write_text("changed")
    elif mutation == "schedule":
        write(run / "schedule.json", [])
    elif mutation == "checkpoint":
        cps[0]["retained_ids"] = [9]
        lines(run / "checkpoints.jsonl", cps)
    elif mutation == "seed":
        rows[0]["seed"] += 1
        lines(run / "outcomes.jsonl", rows)
    elif mutation == "duplicate":
        lines(run / "outcomes.jsonl", rows + rows[:1])
    else:
        write(run / "skipped.json", [{"task": cps[0]["task"], "initial": cps[0]["initial"]}])
    with pytest.raises(ValueError):
        analysis.analyze(run, draws=10)


def test_partial_mechanism_cannot_fit_even_with_24_complete_observed_problems(tmp_path):
    run = tmp_path / "run"
    _, _, manifest = make_run(run, mechanism=True)
    manifest["status"] = "running"
    write(run / "manifest.json", manifest)
    result = analysis.analyze(run, learned=True, draws=10)
    assert result["crossfit"]["status"] == "not_run"
    assert not result["enrollment"]["mechanism_run_complete"]


def test_budget_violation_blocks_learned_fit_but_keeps_descriptive_costs(tmp_path):
    run = tmp_path / "run"
    _, _, manifest = make_run(run)
    manifest["budget"] = 3
    write(run / "manifest.json", manifest)
    result = analysis.analyze(run, learned=True, draws=10)
    assert result["costs"]["ledger_issues"]
    assert result["crossfit"]["status"] == "not_run"
    assert result["enrollment"]["complete_all_arm_problems"] == 24


def test_cheap_jev_and_local_use_same_problem_folds_and_keep_failed_rows(tmp_path, monkeypatch):
    run = tmp_path / "run"
    cps, _, _ = make_run(run)
    local = {}
    for i, cp in enumerate(cps):
        cp["jev"] = {"response": {"answers": {key: {"noul": .6} for key in QUESTIONS}}}
        local[cp["problem_id"]] = {"probabilities": {key: .4 for key in QUESTIONS}} if i else {"error": "malformed_schema"}
    lines(run / "checkpoints.jsonl", cps)
    calls = []
    def fake_fit(train_docs, test_docs, train_num, test_num, train_rows, train_ids, test_ids, alpha):
        calls.append((tuple(train_ids), tuple(test_ids), train_num.shape[1]))
        return np.zeros((len(test_ids), 3)), 1
    monkeypatch.setattr(analysis, "fit_predict", fake_fit)
    result = analysis.analyze(run, learned=True, draws=10, local_features=local)
    assert set(result["crossfit"]["problem_weighted_bernoulli_brier"]) == {"cheap_tfidf", "cheap_tfidf_plus_jev", "cheap_tfidf_plus_local"}
    for start in range(0, len(calls), 3):
        group = calls[start:start + 3]
        assert len({(train, test) for train, test, _ in group}) == 1
        assert [columns for _, _, columns in group] == [14, 21, 21]
    assert result["crossfit"]["n_problems"] == 24
    assert result["feature_quality"]["local_complete_checkpoints"] == 23
    assert result["crossfit"]["policies"]["cheap_tfidf_plus_jev"]["costs"]["jev_uncached_equivalent_usd_per_problem"] is None


def test_durable_orphan_calls_and_unmatched_starts_are_not_lost(tmp_path):
    run = tmp_path / "run"
    make_run(run, n=1, mechanism=True)
    baseline = analysis.analyze(run, draws=10)["costs"]["collection_total_generated_tokens"]
    events = read_lines(run / "generation_events.jsonl")
    starts = read_lines(run / "generation_started.jsonl")
    extra = copy.deepcopy(events[-1])
    extra["sequence"] = len(events)
    extra["generation"]["seed"] = 999999
    events.append(extra)
    starts += [{"sequence": len(events) - 1}, {"sequence": len(events)}]
    lines(run / "generation_events.jsonl", events)
    lines(run / "generation_started.jsonl", starts)
    costs = analysis.analyze(run, draws=10)["costs"]
    assert costs["collection_total_generated_tokens"] == baseline + 2
    assert costs["completed_calls_outside_recorded_checkpoints_or_outcomes"] == 1
    assert costs["totals_are_lower_bounds"]
    assert costs["ledger_issues"]


def test_public_release_explicit_unused_adapter_omission_is_narrowly_supported(tmp_path):
    run = tmp_path / "run"
    _, _, manifest = make_run(run, n=1)
    (run / "source").mkdir()
    source = run / "source/tasks.py"
    source.write_text("# retained verifier\n")
    manifest.update({"original_run_id": "original", "source_sha256": {"tasks.py": digest(source), "jev.py": "a" * 64}})
    write(run / "manifest.json", manifest)
    sources = {"files": [{"released_path": "source/tasks.py", "sha256": digest(source), "original_declared_sha256": digest(source),
                          "status": "captured_run_source", "source_run_id": "original"}],
               "omitted_original_sources": [{"original_filename": "jev.py", "original_declared_sha256": "a" * 64}]}
    write(run / "source_manifest.json", sources)
    names = ["manifest.json", "checkpoints.jsonl", "outcomes.jsonl", "summary.json", "source_manifest.json", "source/tasks.py"]
    release = {"format_version": "public-local-diagnostics-v1", "original_run_id": "original",
               "released_files_sha256": {name: digest(run / name) for name in names}}
    write(run / "release_manifest.json", release)
    result = analysis.analyze(run, draws=10)
    assert result["implementation"]["source_audit"]["declared_unverified_public_omissions"] == ["jev.py"]
    source.write_text("changed")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        analysis.analyze(run, draws=10)


class FakeJudgeBackend:
    model_config = {"model_type": "fake", "quantization": {"bits": 4}}
    quantization_config = {"bits": 4}

    def __init__(self):
        self.prompts, self.calls = [], []
        self.text = json.dumps({key: .5 for key in QUESTIONS})
        self.finish = "stop"
        self.fail = False

    def encode_chat(self, messages):
        self.prompts.append(messages[0]["content"])
        return list(map(ord, messages[0]["content"]))

    def generate(self, prefix, max_tokens, seed, timeout_s):
        self.calls.append((max_tokens, seed, timeout_s))
        if self.fail:
            raise RuntimeError("fake error")
        ids = list(map(ord, self.text))
        return Generation(ids, self.text, len(prefix), len(ids), -1, 1, .25, self.finish, 0, "", seed)


@pytest.fixture
def local_prepared(tmp_path, monkeypatch):
    run = tmp_path / "run"
    cps, _, _ = make_run(run, n=1, mechanism=True)
    cps[0]["state"]["DO_NOT_SEND"] = "GOLD_OR_JEV_SECRET"
    lines(run / "checkpoints.jsonl", cps)
    model = tmp_path / "model"
    model.mkdir()
    write(model / "config.json", {"model_type": "fake", "quantization": {"bits": 4}})
    (model / "model.safetensors").write_bytes(b"fake CPU-only fixture")
    backend, loaded = FakeJudgeBackend(), []
    def factory(*args, **kwargs):
        loaded.append(kwargs)
        return backend
    monkeypatch.setattr(judge, "MLXBackend", factory)
    monkeypatch.setattr(judge.importlib.metadata, "version", lambda _: "fake-version")
    return run, backend, loaded, ["--model", str(model), "--run", str(run), "--max-tokens", "512", "--quantize-bits", "0"]


def test_local_judge_identity_allowlist_quantization_and_costs(local_prepared):
    run, backend, loaded, argv = local_prepared
    judge.main(argv)
    assert loaded == [{"temperature": 0, "quantization_bits": None}]
    assert "GOLD_OR_JEV_SECRET" not in backend.prompts[0]
    assert "FUTURE_SECRET" not in backend.prompts[0]
    records = read_lines(run / "local_judge.jsonl")
    record = records[0]
    provenance = json.loads((run / "local_judge_provenance.json").read_text())
    assert record["provenance_sha256"] == digest(run / "local_judge_provenance.json")
    assert provenance["actual_quantization_config"] == {"bits": 4}
    assert provenance["model_identity"]["weight_revision_sha256"]
    assert provenance["source_sha256"]["scripts/local_judge.py"]
    assert record["generation"]["generated_tokens"] == len(backend.text)
    result = analysis.analyze(run, draws=10, local_features={record["problem_id"]: record})
    assert result["feature_quality"]["local_complete_checkpoints"] == 1
    record["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        analysis.analyze(run, draws=10, local_features={record["problem_id"]: record})


@pytest.mark.parametrize("kind,error", [("malformed", "malformed_schema"), ("timeout", "generation_timeout"), ("exception", "generation_error")])
def test_local_judge_failures_retain_costs_or_mark_unknown(local_prepared, kind, error):
    run, backend, _, argv = local_prepared
    if kind == "malformed":
        backend.text = "not JSON"
    elif kind == "timeout":
        backend.finish = "timeout"
    else:
        backend.fail = True
    if kind == "exception":
        with pytest.raises(RuntimeError):
            judge.main(argv)
    else:
        judge.main(argv)
    record = read_lines(run / "local_judge.jsonl")[0]
    assert record["error"] == error
    assert record["probabilities"] is None
    assert np.isnan(analysis.semantic_features(record)).all()
    if kind == "exception":
        assert record["generated_work_unknown"]
        assert record["generation"] is None
    else:
        assert record["generation"]["generated_tokens"] == len(backend.text)
    assert len(read_lines(run / "local_judge_started.jsonl")) == 1


def test_mechanism_rejects_unidentified_local_features_but_historical_input_accepts(tmp_path):
    run = tmp_path / "run"
    cps, _, manifest = make_run(run, n=1, mechanism=True)
    local = {cps[0]["problem_id"]: {"probabilities": {key: .5 for key in QUESTIONS}}}
    with pytest.raises(ValueError, match="identity"):
        analysis.analyze(run, draws=10, local_features=local)
    manifest.pop("protocol")
    write(run / "manifest.json", manifest)
    assert analysis.analyze(run, draws=10, local_features=local)["feature_quality"]["local_complete_checkpoints"] == 1


def test_local_parser_rejects_duplicate_keys_and_booleans():
    text = json.dumps({key: .5 for key in QUESTIONS})
    assert judge.parse_probabilities(text)
    with pytest.raises(ValueError, match="Duplicate"):
        judge.parse_probabilities(text[:-1] + ', "hypothesis": 0.9}')
    with pytest.raises(ValueError, match="Invalid"):
        judge.parse_probabilities(json.dumps({key: True for key in QUESTIONS}))
