"""Fault-injection tests of the exact rendered v3 source, without model/Java."""
from __future__ import annotations

import ast
import hashlib
import heapq
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/bouchet-recoma-discoveryworld-full-v2-control/artifacts/inputs/source"


def builder_module():
    spec = importlib.util.spec_from_file_location(
        "recoma_v3_builder", ROOT / "scripts/build_recoma_scientific_failure_runtime_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source():
    if not BASELINE.exists():
        pytest.skip("frozen external ReCoMA source bundle unavailable")
    return builder_module().patched_sources(BASELINE)


@pytest.fixture
def accounting(tmp_path):
    # Execute the very bytes the builder puts into the production adapter.
    namespace = {"__file__": str(tmp_path / "task_accounting.py")}
    text = builder_module().ACCOUNTING_SOURCE
    Path(namespace["__file__"]).write_text(text)
    exec(compile(text, namespace["__file__"], "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize(("payload", "code"), [
    (None, "missing_action_json"), ("{bad", "invalid_action_json"),
    ("null", "non_object_action_json"), ("[]", "non_object_action_json"),
    ('"EAT"', "non_object_action_json"),
    ('{"action":"SUBMIT"}', "invalid_submit_arguments"),
    ('{"action":"SUBMIT","arg1":1}', "invalid_submit_arguments"),
    ('{"action":"SUBMIT","arg1":"answer","thought":false}', "invalid_submit_arguments"),
])
def test_typed_model_failures_have_exact_raw_binding_and_no_retry(accounting, payload, code):
    raw = "RAW MODEL OUTPUT"
    with pytest.raises(accounting.ScientificTaskFailure) as failure:
        accounting.parse_generated_action(payload, raw)
    assert failure.value.record() == {
        "code": code, "output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "retry_or_resample": False,
    }
    assert str(failure.value) == code


@pytest.mark.parametrize("value", [
    {"action": "EAT", "arg1": 987}, {}, {"action": "NO_SUCH_ACTION"},
    {"chosen_dialog_option_int": 2},
    {"action": "SUBMIT", "arg1": "answer", "thought": "visible reasoning"},
])
def test_valid_objects_and_environment_denials_preserve_upstream_behavior(accounting, value):
    assert accounting.parse_generated_action(json.dumps(value), "raw") == value


def test_unexpected_internal_type_is_not_mislabeled_scientific_failure(accounting):
    with pytest.raises(TypeError):
        accounting.parse_generated_action(123, "raw")
    with pytest.raises(TypeError):
        accounting.ScientificTaskFailure("cuda_oom", "raw")


def test_actual_upstream_dump_predictions_retains_discoveryworld_records_without_qid(source, tmp_path):
    """Run the unmodified upstream dump function; discovery-world examples have no qid."""
    tree=ast.parse(source["recoma/recoma/run_inference.py"])
    definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="dump_predictions")
    program=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),
                             definition],type_ignores=[])
    namespace={"Path":Path,"json":json}
    exec(compile(ast.fix_missing_locations(program),"actual-upstream-dump-predictions","exec"),namespace)
    class DiscoveryWorldLikeExample:
        def __init__(self,seed):
            self.scenario="Combinatorial Chemistry";self.difficulty="Easy";self.random_seed=seed
        @property
        def unique_id(self):return f"{self.scenario}_{self.difficulty}_{self.random_seed}"
        @property
        def label(self):return self.unique_id
    records=[]
    for seed in (0,1):
        example=DiscoveryWorldLikeExample(seed)
        assert not hasattr(example,"qid")
        metadata={"final_scorecard":[{"scoreNormalized":.25,"completed":False,"completedSuccessfully":False}],
                  "num_steps":3,"num_calls":4,"completion_tokens":20,"prompt_tokens":100}
        records.append(SimpleNamespace(example=example,prediction=json.dumps({"answer":"controller output"}),
                                       final_state=SimpleNamespace(data=metadata)))
    namespace["dump_predictions"](SimpleNamespace(output_dir=str(tmp_path),dump_prompts=False),records)
    dumped=[json.loads(line) for line in (tmp_path/"all_data.jsonl").read_text().splitlines()]
    assert len(dumped)==2 and set(json.loads((tmp_path/"predictions.json").read_text()))=={r.example.unique_id for r in records}
    for row,record in zip(dumped,records):
        assert row["metadata"]==record.final_state.data
        # The generic string EM field is not the official scoreNormalized or task-success endpoint.
        assert row["correct"]=="0" and row["metadata"]["final_scorecard"][0]["scoreNormalized"]==.25
    with pytest.raises(AttributeError,match="qid"):
        namespace["dump_predictions"](SimpleNamespace(output_dir=str(tmp_path),dump_prompts=True),records)


def test_production_wrapper_disables_upstream_qid_prompt_dump_without_removing_usage_trace():
    path=ROOT/"scripts/run_bounded_recoma_worker_v3.py"
    tree=ast.parse(path.read_text())
    settings=[keyword.value for node in ast.walk(tree) if isinstance(node,ast.Call)
              for keyword in node.keywords if keyword.arg=="dump_prompts"]
    assert len(settings)==1 and isinstance(settings[0],ast.Constant) and settings[0].value is False
    assert "RECOMA_USAGE_LOG" in path.read_text()


def actual_controller(source, accounting):
    tree = ast.parse(source["discoveryworld/agents/recoma/react_controller.py"])
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "DiscoveryWorldReactController")
    definition.bases, definition.decorator_list = [], []
    namespace = {"json": json, "re": __import__("re"),
                 "parse_generated_action": accounting.parse_generated_action,
                 "Action": lambda **kw: SimpleNamespace(**kw),
                 "Observation": lambda **kw: SimpleNamespace(**kw)}
    program = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                               definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(program), "rendered-react-controller", "exec"), namespace)
    cls = namespace["DiscoveryWorldReactController"]
    controller = cls.__new__(cls)
    controller.partial_code_regex = r".*```json\n(.*)"
    controller.full_code_regex = r"```json\n(.*?)```"
    controller.ignore_multiple_jsons = True
    controller.OBSERVATION = 2
    controller.get_step = lambda node: 1
    return controller


@pytest.mark.parametrize("text", ["nonsense", "```json\n{bad", "null", "[]"])
def test_actual_controller_stops_bad_json_with_typed_failure(source, accounting, text):
    controller = actual_controller(source, accounting)
    with pytest.raises(accounting.ScientificTaskFailure) as failure:
        controller.append_message_to_history([], SimpleNamespace(output=text))
    assert failure.value.output_sha256 == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("text", ['{"action":"EAT","arg1":99}',
                                '```json\n{"action":"EAT","arg1":99}\n```\n'])
def test_actual_controller_retains_valid_extraction(source, accounting, text):
    controller = actual_controller(source, accounting)
    history = []
    controller.append_message_to_history(history, SimpleNamespace(output=text))
    assert history[0].action_json == {"action": "EAT", "arg1": 99}


def actual_search_class(source, accounting):
    tree = ast.parse(source["recoma/recoma/search/search.py"])
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("SearchAlgo", "BestFirstSearch")]
    for node in classes:
        node.decorator_list = []
        if node.name == "SearchAlgo":
            node.bases = []

    class State:
        def __init__(self, example, data):
            self.example, self.data, self.open = example, data, True

        def add_next_step(self, next_step_model, **kwargs):
            self.model = next_step_model

        def get_open_node(self):
            return SimpleNamespace(target_model=lambda: self.model, tag="test")

        def has_open_node(self):
            return self.open

        def to_str_tree(self):
            return "state"

    namespace = {"SearchState": State, "heapq": heapq, "logger": logging.getLogger(__name__),
                 "ExamplePrediction": lambda **kw: SimpleNamespace(**kw),
                 "ScientificTaskFailure": accounting.ScientificTaskFailure}
    program = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                               *classes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(program), "rendered-search", "exec"), namespace)
    return namespace["BestFirstSearch"]


def test_actual_search_retains_failed_task_and_all_usage_without_resampling(source, accounting):
    cls = actual_search_class(source, accounting)
    search = cls.__new__(cls)
    search.start_model, search.output_dir, search.renderers, search.stopping_conditions = "policy", None, [], []
    calls, evaluations = [], []

    def model(state):
        calls.append(state.example.unique_id)
        state.data.update({"hf_torch.model.calls": 1, "hf_torch.model.prompt_tokens": 17,
                           "hf_torch.model.completion_tokens": 3})
        accounting.parse_generated_action(None, "bad output")
        raise AssertionError("no continuation after scientific failure")

    def endpoint(state):
        evaluations.append(True)
        state.data.update(final_scorecard=[{"scoreNormalized": .4, "completedSuccessfully": False}], num_steps=2)
        return "0.4"

    search.model_list = {"policy": model}
    search.answerer = SimpleNamespace(generate_answer=endpoint)
    prediction = search.predict(SimpleNamespace(unique_id="task-1", task="instructions"))
    row = accounting.prediction_record(prediction)
    assert calls == ["task-1"] and len(evaluations) == 1
    assert row["predicted"] == .4 and row["metadata"]["final_scorecard"][0]["scoreNormalized"] == .4
    assert row["scientific_task_status"] == "scientific_failure"
    assert row["failure_adjusted_completed_successfully"] is False
    assert row["metadata"]["hf_torch.model.calls"] == 1
    assert row["metadata"]["hf_torch.model.prompt_tokens"] == 17
    assert row["metadata"]["hf_torch.model.completion_tokens"] == 3


def test_actual_search_preserves_normal_success_path(source, accounting):
    cls = actual_search_class(source, accounting)
    search = cls.__new__(cls)
    search.start_model, search.output_dir, search.renderers, search.stopping_conditions = "policy", None, [], []
    calls, evaluations = [], []

    def model(state):
        calls.append(state.example.unique_id)
        state.open = False
        state.data["hf_torch.model.calls"] = 1
        return [state]

    def endpoint(state):
        evaluations.append(True)
        state.data.update(final_scorecard=[{"scoreNormalized": 1.0, "completedSuccessfully": True}], num_steps=2)
        return "1.0"

    search.model_list = {"policy": model}
    search.answerer = SimpleNamespace(generate_answer=endpoint)
    row = accounting.prediction_record(search.predict(SimpleNamespace(unique_id="task-1", task="instructions")))
    assert calls == ["task-1"] and len(evaluations) == 1
    assert row["scientific_task_status"] == "completed"
    assert row["failure_adjusted_completed_successfully"] is True
    assert row["predicted"] == 1.0 and "scientific_task_failure" not in row["metadata"]


@pytest.mark.parametrize("exception", [RuntimeError("CUDA OOM"), TypeError("internal bug"), ValueError("simulator error")])
def test_actual_search_propagates_infrastructure_errors(source, accounting, exception):
    cls = actual_search_class(source, accounting)
    search = cls.__new__(cls)
    search.start_model, search.output_dir, search.renderers = "policy", None, []
    search.model_list = {"policy": lambda _: (_ for _ in ()).throw(exception)}
    search.answerer = SimpleNamespace(generate_answer=lambda _: pytest.fail("no scientific outcome on infrastructure error"))
    with pytest.raises(type(exception), match=str(exception)):
        search.predict(SimpleNamespace(unique_id="task-1", task="instructions"))


def prediction(task_id, failure=None, success=False):
    example = SimpleNamespace(unique_id=task_id)
    data = {"final_scorecard": [{"scoreNormalized": .4, "completedSuccessfully": success}], "num_steps": 2}
    if failure:
        data["scientific_task_failure"] = failure
    return SimpleNamespace(example=example, prediction="0.4", final_state=SimpleNamespace(data=data))


def test_every_task_is_durable_before_next_inference_and_failed_tasks_remain(accounting, tmp_path):
    examples = [SimpleNamespace(unique_id="a"), SimpleNamespace(unique_id="b")]
    calls = []
    failure = accounting.ScientificTaskFailure("missing_action_json", "bad").record()

    def predict(example):
        if example.unique_id == "b":
            saved = [json.loads(line) for line in (tmp_path / "task_records.jsonl").read_text().splitlines()]
            assert saved[0]["task_id"] == "a" and saved[0]["scientific_task_status"] == "scientific_failure"
        calls.append(example.unique_id)
        return prediction(example.unique_id, failure if example.unique_id == "a" else None)

    result = accounting.collect_durable_predictions(SimpleNamespace(get_examples=lambda _: examples),
                                                    SimpleNamespace(predict=predict), None, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "task_records.jsonl").read_text().splitlines()]
    assert len(result) == len(rows) == 2 and calls == ["a", "b"]
    assert all(row["task_wall_seconds"] >= 0 for row in rows)
    assert rows[0]["metadata"]["scientific_task_failure"]["retry_or_resample"] is False


def test_infrastructure_abort_preserves_prior_task_and_no_new_outcome(accounting, tmp_path):
    calls = []

    def predict(example):
        calls.append(example.unique_id)
        if example.unique_id == "b":
            raise RuntimeError("hardware failure")
        return prediction(example.unique_id)

    examples = [SimpleNamespace(unique_id=value) for value in ("a", "b", "c")]
    with pytest.raises(RuntimeError, match="hardware failure"):
        accounting.collect_durable_predictions(SimpleNamespace(get_examples=lambda _: examples),
                                               SimpleNamespace(predict=predict), None, tmp_path)
    assert calls == ["a", "b"]
    assert len((tmp_path / "task_records.jsonl").read_text().splitlines()) == 1
    events = [json.loads(line) for line in (tmp_path / "task_events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "infrastructure_abort"
    assert events[-1]["exception_type"] == "RuntimeError"


def test_ledgers_refuse_overwrite_and_enforce_output_cap(accounting, tmp_path):
    ledger = accounting.DurableJSONL(tmp_path / "records.jsonl", 4)
    with pytest.raises(RuntimeError, match="byte cap"):
        ledger.append({"long": "record"})
    with pytest.raises(FileExistsError):
        accounting.DurableJSONL(tmp_path / "records.jsonl", 4)


def test_failure_does_not_inherit_official_success_as_deployed_success(accounting):
    failure = accounting.ScientificTaskFailure("missing_action_json", "bad").record()
    row = accounting.prediction_record(prediction("a", failure, success=True))
    assert row["metadata"]["final_scorecard"][0]["completedSuccessfully"] is True
    assert row["failure_adjusted_completed_successfully"] is False


def test_rendered_runner_uses_durable_collection_before_summary(source):
    runner = source["recoma/recoma/run_inference.py"]
    assert "example_predictions = collect_durable_predictions(" in runner
    assert runner.index("example_predictions = collect_durable_predictions(") < runner.index("    dump_predictions(args, example_predictions)")


def test_built_patch_applies_and_matches_all_rendered_source(source, tmp_path):
    builder = builder_module()
    baseline = tmp_path / "baseline"
    for relative in source:
        old = BASELINE / relative
        if old.exists():
            target = baseline / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old.read_bytes())
    output, patch = tmp_path / "output", tmp_path / "runtime.patch"
    receipt = builder.build(baseline, output, patch)
    assert receipt["changed_files"]["recoma/recoma/utils/task_accounting.py"]["before_sha256"] is None
    subprocess.run(["git", "init", "-q"], cwd=baseline, check=True)
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=baseline, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=baseline, check=True)
    for relative, text in source.items():
        assert (baseline / relative).read_text() == (output / relative).read_text() == text
    with pytest.raises(FileExistsError):
        builder.build(baseline, output, patch)
