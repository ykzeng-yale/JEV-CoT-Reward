"""Offline integrity tests; synthetic records, no model or network calls."""
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

from jev_control.mlx_backend import Generation
from jev_control.tasks import make_task

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import run_development as development
    import analyze_development as analysis
    import export_public_runs as public_export
finally:
    sys.path.remove(str(ROOT / "scripts"))


class Tokenizer:
    audit_sha256 = "0" * 64

    def encode(self, text):
        return list(map(ord, text))

    def decode(self, ids):
        return "".join(map(chr, ids))


class Backend(Tokenizer):
    def __init__(self, responses):
        self.responses = list(responses)

    encode_text = Tokenizer.encode

    def generate(self, prefix, max_tokens, seed, timeout_s, stop_when=None):
        text, finish = self.responses.pop(0)
        scripted = self.encode(text)
        ids = []
        for token in scripted[:max_tokens]:
            ids.append(token)
            if len(ids) == len(scripted) and finish == "stop":
                break
            if stop_when is not None and stop_when(list(ids)):
                finish = "checkpoint"
                break
        return Generation(ids, self.decode(ids), len(prefix), len(ids), -1., 1.,
                          .25, finish, .01, analysis.prefix_hash(prefix), seed)


def answer(task, index, seed):
    if task["family"] == "arithmetic_construction":
        rng = random.Random(seed + index)
        a, b, c, d, e, f = [rng.randint(1, 12) for _ in range(6)]
        return f"({a}+{b})*{c}+{d}*{e}-{f}"
    best = {0: (0, [0])}
    for a, b, weight in task["data"]["edges"]:
        if a in best and (b not in best or best[a][0] + weight < best[b][0]):
            best[b] = (best[a][0] + weight, best[a][1] + [b])
    return "->".join(chr(65 + x) for x in best[task["data"]["n"] - 1][1])


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fixture_run(tmp_path, *, full_default=False, mismatch=False, terminal=None):
    plan = json.loads((ROOT / "configs/development_v1.json").read_text())
    if not full_default:
        plan.update(problems=2, repeats=2, budget=128, checkpoint_target=16, checkpoint_cap=48, final_reserve=32)
    source = tmp_path / "source"
    source.mkdir()
    hashes = {}
    for name in ("scripts/run_development.py", "scripts/run_screen.py",
                 "src/jev_control/tasks.py", "src/jev_control/mlx_backend.py"):
        data = (ROOT / name).read_bytes()
        (source / Path(name).name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {k: plan[k] for k in analysis.PLAN_FIELDS}
    manifest.update(source_sha256=hashes, model="/Users/private_account/private_model", output="/Users/private_account/private_run")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    items, rows = [], []
    for index in range(plan["problems"]):
        task = make_task(index, plan["task_seed"])
        task["prompt"] += development.PROMPT_SUFFIX
        prompt = Tokenizer().encode(task["prompt"])
        items.append({"task": task, "prompt_ids": prompt})
        for repeat in range(plan["repeats"]):
            for condition in analysis.CONDITIONS:
                initial = ("loss=2+3" if mismatch and condition == "sham" else "cost=2+3")
                initial += "x" * (plan["checkpoint_target"] - len(initial) - 1) + "\n"
                tail = "z" * (plan["budget"] - plan["final_reserve"] - len(initial))
                final = answer(task, index, plan["task_seed"]) if condition == "baseline" or repeat == 0 else "0"
                if terminal:
                    responses = [("FINAL: " + final if terminal == "stop" else "bad", terminal)]
                elif condition == "baseline":
                    responses = [(initial + tail, "length"), (final, "stop")]
                else:
                    responses = [(initial, "length"), (tail, "length"), (final, "stop")]
                seed = plan["sampling_seed"] + index * 1000 + repeat * 10
                row = development.episode(Backend(responses), task, prompt, condition, seed,
                                          plan["budget"], plan["final_reserve"], plan["checkpoint_target"], plan["checkpoint_cap"])
                row["repeat"] = repeat
                rows.append(row)
    write_jsonl(tmp_path / "tasks.jsonl", items)
    write_jsonl(tmp_path / "outcomes.jsonl", rows)
    (tmp_path / "summary.json").write_text(json.dumps(development.summarize(rows)))
    return plan, rows, items


def analyze(tmp_path, plan, **kwargs):
    return analysis.analyze(tmp_path, plan=plan, tokenizer=Tokenizer(), draws=100, **kwargs)


def test_full_default_plan_counts_independent_problems_and_failed_work(tmp_path):
    plan, rows, _ = fixture_run(tmp_path, full_default=True)
    result = analysis.analyze(tmp_path, tokenizer=Tokenizer(), draws=100)
    assert result["counts"]["observed_episodes"] == 48
    assert result["counts"]["recorded_tasks"] == 12
    assert result["counts"]["paired_problem_repeat_cells"] == 24
    assert result["conditions"]["baseline"]["successes"] == 24
    assert result["conditions"]["sham"]["successes"] == 12
    assert result["fixed_control_comparison"]["independent_problems"] == 12
    assert result["fixed_control_comparison"]["difference"] == -.5
    assert result["fixed_control_comparison"]["strata"] == {"weighted_path": 6, "arithmetic_construction": 6}
    assert result["accounting"]["generated_tokens"] == sum(c["generated_tokens"] for r in rows for c in r["calls"])
    assert result["accounting"]["prompt_tokens_processed"] == sum(c["prompt_tokens"] for r in rows for c in r["calls"])
    assert result["conditions"]["sham"]["failed_outcome_costs"]["generation_calls"] == 36
    assert result["prepause_matching"]["full_sham_prefix_agreements"] == 24
    assert result["eligibility_and_markers"]["eligible_with_marker"]["episodes"] == 24
    assert not result["scientific_final"]
    serialized = json.dumps(result, allow_nan=False)
    assert "private_account" not in serialized and "/Users/" not in serialized
    assert result["provenance"]["audit_time_tokenizer_json_sha256"] == "0" * 64


def test_explicit_partial_has_no_final_comparison_and_preserves_run(tmp_path):
    plan, rows, _ = fixture_run(tmp_path)
    write_jsonl(tmp_path / "outcomes.jsonl", rows[:-1])
    (tmp_path / "summary.json").unlink()
    before = (tmp_path / "outcomes.jsonl").read_bytes()
    with pytest.raises(ValueError, match="Incomplete expected plan"):
        analyze(tmp_path, plan)
    result = analyze(tmp_path, plan, partial=True)
    assert result["status"] == "provisional_partial_diagnostics"
    assert result["counts"]["missing_episodes"] == 1
    assert result["fixed_control_comparison"] is None
    assert not result["scientific_final"] and not result["expected_plan_complete"]
    assert (tmp_path / "outcomes.jsonl").read_bytes() == before


def test_partial_flag_stays_provisional_even_with_complete_data(tmp_path):
    plan, _, _ = fixture_run(tmp_path)
    result = analyze(tmp_path, plan, partial=True)
    assert result["expected_plan_complete"]
    assert result["fixed_control_comparison"] is None
    assert result["status"] == "provisional_partial_diagnostics"


def test_default_plan_not_inferred_downward_from_small_manifest(tmp_path):
    fixture_run(tmp_path)
    with pytest.raises(ValueError, match="expected plan: problems"):
        analysis.analyze(tmp_path, tokenizer=Tokenizer())


@pytest.mark.parametrize("kind", ["episode", "task"])
def test_duplicate_observations_are_refused(tmp_path, kind):
    plan, rows, items = fixture_run(tmp_path)
    if kind == "episode":
        write_jsonl(tmp_path / "outcomes.jsonl", rows + [rows[0]])
    else:
        write_jsonl(tmp_path / "tasks.jsonl", items + [items[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(tmp_path, plan, partial=True)


@pytest.mark.parametrize("mutate,message", [
    (lambda r: r.update(generated_tokens=r["generated_tokens"] + 1), "differs from call sum"),
    (lambda r: r.update(prompt_tokens_processed=r["prompt_tokens_processed"] + 1), "differs from call sum"),
    (lambda r: r.update(elapsed_seconds=-1), "finite nonnegative"),
    (lambda r: r["calls"][0].update(generated_tokens=1), "raw token IDs"),
    (lambda r: r["calls"][-1].update(prefix_sha256="0" * 64), "prefix hash mismatch"),
    (lambda r: r["calls"][-1].update(prompt_tokens=1), "prompt count mismatch"),
    (lambda r: r["calls"][-1].update(seed=3), "call seed mismatch"),
    (lambda r: r["calls"][0].update(text="tampered"), "text/token decode mismatch"),
    (lambda r: r["overhead"][0].update(tokens=1), "FINAL instruction overhead mismatch"),
    (lambda r: r.update(family="wrong_family"), "family mismatch"),
    (lambda r: r.update(problem_id="missing"), "missing or unknown task"),
    (lambda r: r.update(repeat=2), "Repeat outside"),
    (lambda r: r.update(seed=1), "Episode seed differs"),
    (lambda r: r.update(outcome=None), "malformed recorded outcome"),
    (lambda r: r["outcome"].update(private_path="/Users/private_account"), "unexpected recorded outcome fields"),
    (lambda r: r.update(generated_tokens=float(r["generated_tokens"])), "expected integer"),
])
def test_malformed_call_and_join_data_refused(tmp_path, mutate, message):
    plan, rows, _ = fixture_run(tmp_path)
    mutate(rows[0])
    write_jsonl(tmp_path / "outcomes.jsonl", rows)
    with pytest.raises(ValueError, match=message):
        analyze(tmp_path, plan)


def test_raw_token_cap_violation_is_refused(tmp_path):
    plan, rows, _ = fixture_run(tmp_path)
    call = rows[0]["calls"][0]
    call["token_ids"].append(ord("x"))
    call["generated_tokens"] += 1
    write_jsonl(tmp_path / "outcomes.jsonl", rows)
    with pytest.raises(ValueError, match="call token cap exceeded"):
        analyze(tmp_path, plan)


def test_checkpoint_hash_and_successive_call_chain_are_independently_checked(tmp_path):
    plan, rows, _ = fixture_run(tmp_path)
    row = next(r for r in rows if r["condition"] == "sham")
    row["checkpoint"]["prefix_sha256"] = "0" * 64
    write_jsonl(tmp_path / "outcomes.jsonl", rows)
    with pytest.raises(ValueError, match="checkpoint prefix hash mismatch"):
        analyze(tmp_path, plan)


def test_changed_source_snapshot_and_malformed_json_are_refused(tmp_path):
    plan, _, _ = fixture_run(tmp_path)
    p = tmp_path / "source/run_screen.py"
    original = p.read_bytes()
    p.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Frozen source hash mismatch"):
        analyze(tmp_path, plan)
    p.write_bytes(original)
    with (tmp_path / "outcomes.jsonl").open("a") as out:
        out.write('{"unfinished":')
    with pytest.raises(ValueError, match="malformed JSON"):
        analyze(tmp_path, plan, partial=True)


def test_outcomes_reverified_without_rewriting_logged_records(tmp_path):
    plan, rows, _ = fixture_run(tmp_path)
    rows[0]["outcome"]["success"] = False
    write_jsonl(tmp_path / "outcomes.jsonl", rows)
    before = (tmp_path / "outcomes.jsonl").read_bytes()
    result = analyze(tmp_path, plan)
    assert len(result["outcome_reverification"]["disagreements"]) == 1
    assert result["conditions"]["baseline"]["successes"] == 4
    assert (tmp_path / "outcomes.jsonl").read_bytes() == before


def test_prepause_disagreement_is_retained_diagnostic_not_global_bitwise_claim(tmp_path):
    plan, _, _ = fixture_run(tmp_path, mismatch=True)
    result = analyze(tmp_path, plan)
    assert result["prepause_matching"]["full_sham_prefix_agreements"] == 0
    assert len(result["prepause_matching"]["disagreements"]) == 4
    assert all(x["first_disagreement_position_zero_based"] == 0 for x in result["prepause_matching"]["disagreements"])
    assert result["fixed_control_comparison"]["difference"] == -.5


@pytest.mark.parametrize("terminal", ["stop", "timeout"])
def test_terminal_calls_keep_costs_and_skip_injection(tmp_path, terminal):
    plan, rows, _ = fixture_run(tmp_path, terminal=terminal)
    result = analyze(tmp_path, plan)
    assert result["accounting"]["generation_calls"] == len(rows)
    assert result["accounting"]["inserted_instruction_tokens"] == 0
    if terminal == "timeout":
        assert result["conditions"]["sham"]["timeout_episodes"] == 4
        assert result["conditions"]["sham"]["failed_outcome_costs"]["generated_tokens"] == 12


def test_cli_refuses_existing_output_before_analysis(tmp_path, monkeypatch):
    output = tmp_path / "analysis.json"
    output.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["analyze_development", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        analysis.main()
    assert exc.value.code == 2
    assert output.read_text() == "preserve"


def public_fixture(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    plan, _, _ = fixture_run(original)
    manifest = json.loads((original / "manifest.json").read_text())
    # Match a fully captured launch inventory, including the unused adapter.
    for name in ("src/jev_control/__init__.py", "src/jev_control/features.py", "src/jev_control/jev.py"):
        content = (ROOT / name).read_bytes()
        (original / "source" / Path(name).name).write_bytes(content)
        manifest["source_sha256"][name] = analysis.digest(content)
    (original / "manifest.json").write_text(json.dumps(manifest))
    public = tmp_path / "public"
    public_export.export_run(original, public, mode="development-v1")
    return original, public, plan


def refresh_public_file_hash(public, name):
    """Exercise semantic provenance checks beyond simple file hash mismatch."""
    path = public / "release_manifest.json"
    release = json.loads(path.read_text())
    release["released_files_sha256"][name] = analysis.digest((public / name).read_bytes())
    path.write_text(json.dumps(release))


def test_public_release_reanalysis_matches_private_scientific_fields(tmp_path):
    original, public, plan = public_fixture(tmp_path)
    left = analyze(original, plan)
    right = analyze(public, plan)
    assert {k: v for k, v in left.items() if k != "provenance"} == {k: v for k, v in right.items() if k != "provenance"}
    assert right["provenance"]["run_source_sha256"] == left["provenance"]["run_source_sha256"]
    omitted = right["provenance"]["source_validation"]["omitted_original_sources"]
    assert len(omitted) == 1 and omitted[0]["original_repository_path"] == "src/jev_control/jev.py"
    assert omitted[0]["content_verified"] is False
    assert left["provenance"]["source_validation"]["omitted_original_sources"] == []


def test_missing_private_source_still_fails_without_public_declaration(tmp_path):
    original, _, plan = public_fixture(tmp_path)
    (original / "source/jev.py").unlink()
    with pytest.raises(ValueError, match="Frozen source hash mismatch: jev.py"):
        analyze(original, plan)


def test_public_release_rejects_undeclared_missing_sources(tmp_path):
    _, public, plan = public_fixture(tmp_path)
    provenance = json.loads((public / "source_manifest.json").read_text())
    provenance["omitted_original_sources"] = []
    (public / "source_manifest.json").write_text(json.dumps(provenance))
    refresh_public_file_hash(public, "source_manifest.json")
    with pytest.raises(ValueError, match="Undeclared missing source"):
        analyze(public, plan)


def test_public_release_cannot_exempt_executed_helper(tmp_path):
    _, public, plan = public_fixture(tmp_path)
    manifest = json.loads((public / "manifest.json").read_text())
    provenance = json.loads((public / "source_manifest.json").read_text())
    provenance["omitted_original_sources"].append({"original_repository_path": "src/jev_control/tasks.py",
                                                  "original_declared_sha256": manifest["source_sha256"]["src/jev_control/tasks.py"]})
    (public / "source_manifest.json").write_text(json.dumps(provenance))
    refresh_public_file_hash(public, "source_manifest.json")
    with pytest.raises(ValueError, match="may omit only the unused Jev adapter"):
        analyze(public, plan)


def test_public_release_rejects_tampered_data_and_borrowed_source_provenance(tmp_path):
    _, public, plan = public_fixture(tmp_path)
    outcome = public / "outcomes.jsonl"
    original = outcome.read_bytes()
    outcome.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Public release file hash mismatch: outcomes.jsonl"):
        analyze(public, plan)
    outcome.write_bytes(original)
    provenance = json.loads((public / "source_manifest.json").read_text())
    provenance["files"][0]["source_run_id"] = "another-version"
    (public / "source_manifest.json").write_text(json.dumps(provenance))
    refresh_public_file_hash(public, "source_manifest.json")
    with pytest.raises(ValueError, match="this run's captured launch file"):
        analyze(public, plan)
