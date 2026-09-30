from __future__ import annotations

from types import SimpleNamespace

import pytest

import copy
import hashlib
import json

from jev_control.scienceworld_conductivity_action_audit import audit, evaluate_terminal, training_label


class JavaClass:
    def __init__(self, name):
        self.name = name

    def getSimpleName(self):
        return self.name


class Typed:
    def __init__(self, kind):
        self.kind = kind

    def getClass(self):
        return JavaClass(self.kind)


class Modifier(Typed):
    def __init__(self, key, value, kind="TaskValueStr"):
        super().__init__(kind)
        self._key, self._value = key, value

    def key(self):
        return self._key

    def value(self):
        return self._value


class Option:
    def __init__(self, value):
        self.value = value

    def isDefined(self):
        return self.value is not None

    def get(self):
        assert self.isDefined()
        return self.value


class Object:
    def __init__(self, name, parent=None):
        self._name, self.parent = name, parent

    def name(self):
        return self._name

    def getContainer(self):
        return Option(self.parent)


class ScalaCollection:
    def __init__(self, items):
        self.items = items

    def iterator(self):
        items = iter(self.items)
        remaining = len(self.items)

        def has_next():
            return remaining > 0

        def get_next():
            nonlocal remaining
            remaining -= 1
            return next(items)

        return SimpleNamespace(hasNext=has_next, next=get_next)


@pytest.fixture
def state():
    target = Object("unknown substance B", Object("red box"))
    state = SimpleNamespace(
        target=target, objects=[target], complete=True, failed=False,
        modifiers=[
            Typed("TaskObject"),  # No key()/value(): heterogeneous Scala list.
            Modifier("unknownSubstance", "unknown substance B"),
            Modifier("unknownIsConductive", True, "TaskValueBool"),
            Modifier("conductive", "red box"),
            Modifier("nonconductive", "blue box"),
        ],
        goal_classes=["GoalFind", "GoalObjectInContainerByName"],
    )
    goals = SimpleNamespace(
        subgoals=lambda: [Typed(kind) for kind in state.goal_classes],
        isCompleted=lambda: state.complete, isFailed=lambda: state.failed,
    )
    task = SimpleNamespace(taskModifiers=lambda: state.modifiers, goalSequence=lambda: goals)
    universe = SimpleNamespace(getContainedObjectsRecursive=lambda: ScalaCollection(state.objects))
    interface = SimpleNamespace(task=lambda: task, universe=lambda: universe)

    # There is deliberately no score/reward/done/gold-path API in this fixture.
    state.env = SimpleNamespace(server=SimpleNamespace(agentInterface=lambda: Option(interface)))
    return state


def run(state, prediction=True, box="red box"):
    return evaluate_terminal(state.env, "unknown substance B", box, prediction)


def test_valid_classification_placement_and_ordered_success(state):
    result = run(state)
    assert result["classification_correct"]
    assert result["placement_correct"]
    assert result["ordered_goal_complete"]
    assert result["task_success"]
    assert set(result.values()) <= {False, True}
    assert "source_label" not in result


def test_requested_correct_box_does_not_count_when_move_did_not_happen(state):
    state.target.parent = Object("inventory")
    result = run(state)
    assert result["classification_correct"]
    assert not result["placement_correct"]
    assert not result["chosen_box_matches_placement"]
    assert not result["task_success"]


def test_optional_progress_or_correct_placement_does_not_replace_ordered_goals(state):
    state.complete = False
    assert run(state)["placement_correct"]
    assert not run(state)["task_success"]


def test_failure_overrides_goal_completion(state):
    state.failed = True
    assert not run(state)["task_success"]


def test_classification_and_actual_action_are_distinct_outcomes(state):
    result = run(state, prediction=False, box="blue box")
    assert not result["classification_correct"]
    assert not result["chosen_box_matches_placement"]
    assert result["placement_correct"]
    assert result["task_success"]


def test_nested_container_matches_simulator_ancestry_rule(state):
    state.target.parent = Object("small jar", Object("red box"))
    assert run(state)["placement_correct"]


def test_wrong_container_ancestor_takes_precedence(state):
    state.target.parent = Object("red box", Object("blue box"))
    assert not run(state)["placement_correct"]


def test_container_search_has_simulator_twenty_ancestor_bound(state):
    ancestor = Object("red box")
    for _ in range(20):
        ancestor = Object("nested jar", ancestor)
    state.target.parent = ancestor
    assert not run(state)["placement_correct"]


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_runtime_target_fails_closed(state, count):
    state.objects = [state.target] * count
    result = run(state)
    assert not result["target_unique"]
    assert not result["task_success"]


def test_unresolved_policy_prediction_counts_as_incorrect(state):
    assert not run(state, prediction=None, box=None)["classification_correct"]


def test_source_target_mismatch_is_rejected(state):
    with pytest.raises(ValueError, match="target differs"):
        evaluate_terminal(state.env, "unknown substance C", "red box", True)


def test_ambiguous_source_label_is_rejected(state):
    state.modifiers.append(Modifier("unknownIsConductive", False, "TaskValueBool"))
    with pytest.raises(ValueError, match="ambiguous"):
        run(state)


def test_nonboolean_prediction_and_label_rejected(state):
    with pytest.raises(TypeError, match="prediction"):
        run(state, prediction=1)
    state.modifiers[2] = Modifier("unknownIsConductive", "true", "TaskValueBool")
    with pytest.raises(TypeError, match="label"):
        run(state)


def test_changed_goal_contract_rejected(state):
    state.goal_classes = ["GoalFindAnswerBox"]
    with pytest.raises(ValueError, match="goal structure"):
        run(state)


def test_training_calibration_requires_frozen_training_membership_and_loaded_identity(state):
    state.env.variationIdx = 0
    assert training_label(state.env, 0, [0, 1]) is True
    with pytest.raises(ValueError, match="outside"):
        training_label(state.env, 2, [0, 1])
    with pytest.raises(ValueError, match="loaded"):
        training_label(state.env, 1, [0, 1])


@pytest.fixture
def panel(state):
    protocol = {
        "task": "test-conductivity-of-unknown-substances",
        "splits": {"train": {"start_inclusive": 0, "end_exclusive": 2},
                   "dev": {"start_inclusive": 2, "end_exclusive": 4}},
        "policies": ["measurement", "masked_measurement", "continue_prior", "random_measurement"],
        "limits": {"episode_action_budget": 32},
    }
    description = "Your task is to determine if unknown substance B is electrically conductive."
    legal = ["wait", "move unknown substance B to red box"]

    def digest(obs):
        data = {"task_description": description, "observation": obs, "legal_actions": sorted(legal)}
        return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    episodes = []
    for variation in (2, 3):
        for policy in protocol["policies"]:
            episodes.append({
                "variation_id": variation, "split": "dev", "policy": policy,
                "initial_view_sha256": digest("OBS0"),
                "initial_look_sha256": hashlib.sha256(b"OBS0").hexdigest(),
                "trace": [{"action": "wait", "view_sha256": digest("OBS1")},
                          {"action": legal[1], "view_sha256": digest("OBS2")}],
                "prediction": True, "chosen_box": "red box", "endpoint": run(state),
                "actions": 2, "wall_seconds": 1.25, "failure": None, "decision_action_index": 1,
                "target_group": "unknown substance B", "intrinsic_action_count": 2,
                "circuit_actions": 0, "reading_sha256": None,
            })
    input_census = [{
        "split": "train" if variation < 2 else "dev", "variation_id": variation,
        "target_group": "unknown substance B",
        "task_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "look_sha256": hashlib.sha256(b"OBS0").hexdigest(), "full_sha256": digest("OBS0"),
        "look_label_matches_first": True, "full_label_matches_first": True,
    } for variation in range(4)]
    state.closed = False
    state.steps = 0

    def load(task, *, variationIdx, simplificationStr, generateGoldPath):
        assert task == protocol["task"] and simplificationStr == "" and generateGoldPath is False
        assert variationIdx in (0, 1, 2, 3)
        state.steps = 0

    def step(action):
        assert action in legal
        state.steps += 1
        return f"OBS{state.steps}", 47, True, {"score": 47}

    state.env.load = load
    state.env.reset = lambda: ("OBS0", {"score": 0})
    state.env.step = step
    state.env.taskdescription = lambda: description
    state.env.get_valid_action_object_combinations = lambda: list(reversed(legal))
    state.env.close = lambda: setattr(state, "closed", True)
    return protocol, {"episodes": episodes, "input_census": input_census,
                      "calibration": {"train_positive": 2, "train_total": 2,
                                      "selected_prior": True, "tie_rule": False}}


def test_record_validation_alone_cannot_certify_outcomes(panel):
    report = audit(*panel)
    assert report["record_checks_pass"]
    assert not report["passes"]
    assert not report["replay_performed"]


def test_independent_replay_recomputes_views_and_private_endpoints(panel, state):
    report = audit(*panel, env_factory=lambda: state.env)
    assert report["passes"]
    assert report["replay_episodes_verified"] == 8
    assert report["replay_census_rows_verified"] == 4
    assert state.closed
    assert report["policy_counts"]["measurement"]["actions"] == 4


def test_false_but_internally_consistent_classification_rejected_by_replay(panel, state):
    protocol, result = copy.deepcopy(panel)
    result["episodes"][0]["endpoint"]["classification_correct"] = False
    assert audit(protocol, result)["record_checks_pass"]
    report = audit(protocol, result, env_factory=lambda: state.env)
    assert not report["passes"]
    assert report["replay_error"]
    assert state.closed


@pytest.mark.parametrize("change", ["duplicate", "missing", "foreign_id", "wrong_split"])
def test_incomplete_or_contaminated_panel_rejected(panel, change):
    protocol, result = copy.deepcopy(panel)
    if change == "duplicate":
        result["episodes"].append(copy.deepcopy(result["episodes"][0]))
    elif change == "missing":
        result["episodes"].pop()
    elif change == "foreign_id":
        result["episodes"][0]["variation_id"] = 12345
    else:
        result["episodes"][0]["split"] = "test"
    assert not audit(protocol, result)["record_checks_pass"]


def test_action_matched_arms_require_identical_prefix_views_and_actions(panel):
    protocol, result = copy.deepcopy(panel)
    result["episodes"][1]["trace"][0]["view_sha256"] = "0" * 64
    assert not audit(protocol, result)["checks"]["measurement_prefixes_equal"]


def test_pairs_require_identical_initial_controller_inputs(panel):
    protocol, result = copy.deepcopy(panel)
    result["episodes"][1]["initial_view_sha256"] = "0" * 64
    assert not audit(protocol, result)["checks"]["paired_initial_views_equal"]


@pytest.mark.parametrize("change", ["fake_success", "bad_steps", "nan_cost", "string_bool", "label_leak"])
def test_invalid_outcome_or_cost_schema_rejected(panel, change):
    protocol, result = copy.deepcopy(panel)
    row = result["episodes"][0]
    if change == "fake_success":
        row["endpoint"]["placement_correct"] = False
    elif change == "bad_steps":
        row["actions"] = 1
        row["intrinsic_action_count"] = 1
    elif change == "nan_cost":
        row["wall_seconds"] = float("nan")
    elif change == "string_bool":
        row["endpoint"]["classification_correct"] = "False"
    else:
        row["source_label"] = True
    assert not audit(protocol, result)["record_checks_pass"]


def test_failed_episodes_are_retained_in_denominators_and_costs(panel):
    protocol, result = copy.deepcopy(panel)
    for row in result["episodes"][:4]:
        row["trace"] = row["trace"][:1]
        row["actions"] = 1
        row["intrinsic_action_count"] = 1
        row["decision_action_index"] = None
        row["prediction"] = row["chosen_box"] = None
        row["failure"] = "measurement unavailable"
        row["endpoint"].update(classification_correct=False, placement_correct=False,
                               chosen_box_matches_placement=False, ordered_goal_complete=False,
                               task_success=False)
    report = audit(protocol, result)
    assert report["record_checks_pass"]
    for policy in protocol["policies"]:
        counts = report["policy_counts"][policy]
        assert counts["episodes"] == 2
        assert counts["failures"] == 1
        assert counts["task_success"] == 1
        assert counts["actions"] == 3


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "test", "label_leak", "first_false"])
def test_census_requires_exact_ordered_train_dev_safe_records(panel, change):
    protocol, result = copy.deepcopy(panel)
    rows = result["input_census"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "reordered":
        rows.reverse()
    elif change == "test":
        rows[0]["split"] = "test"
    elif change == "label_leak":
        rows[0]["source_label"] = True
    else:
        rows[0]["full_label_matches_first"] = False
    assert not audit(protocol, result)["record_checks_pass"]


def test_census_relative_label_flag_independently_verified(panel, state):
    protocol, result = copy.deepcopy(panel)
    result["input_census"][1]["full_label_matches_first"] = False
    assert audit(protocol, result)["record_checks_pass"]
    report = audit(protocol, result, env_factory=lambda: state.env)
    assert not report["passes"]
    assert report["replay_error"]


def test_calibration_counts_require_runtime_replay_not_just_majority_logic(panel, state):
    protocol, result = copy.deepcopy(panel)
    result["calibration"].update(train_positive=1, selected_prior=False)
    assert audit(protocol, result)["record_checks_pass"]
    assert not audit(protocol, result, env_factory=lambda: state.env)["passes"]


def test_episodes_initial_inputs_must_match_independent_census(panel):
    protocol, result = copy.deepcopy(panel)
    for row in result["episodes"]:
        row["initial_view_sha256"] = "0" * 64
    assert not audit(protocol, result)["checks"]["episodes_match_census"]


def test_aggregate_distinguishes_look_overlap_from_full_controller_overlap(panel):
    protocol, result = copy.deepcopy(panel)
    # Synthetic source contract: dev has a different target and full view while
    # its room-only observation still repeats training exactly.
    for row in result["input_census"]:
        if row["split"] == "dev":
            row["target_group"] = "unknown substance C"
            row["full_sha256"] = "f" * 64
            row["task_description_sha256"] = "d" * 64
    for row in result["episodes"]:
        row["target_group"] = "unknown substance C"
        row["initial_view_sha256"] = "f" * 64
    report = audit(protocol, result)
    assert report["record_checks_pass"]
    assert report["input_census"]["look"]["shared_train_dev_groups"] == 1
    assert report["input_census"]["full"]["shared_train_dev_groups"] == 0
    assert report["input_census"]["target_groups"]["shared"] == 0
