from __future__ import annotations

import pytest

from jev_control.scienceworld_conductivity_policy import (
    CircuitKit, ConductivityTask, Endpoint, choose_kit, indicator_on,
    make_controller_view, measurement_connections, parse_task, select_answer,
    select_connection, select_focus,
    resolve_target_ref,
)


def description(letter="B", conductive="red", nonconductive="green"):
    target = "unknown substance " + letter
    return (f"Your task is to determine if {target} is electrically conductive. "
            f"The {target} is located around the workshop. First, focus on the {target}. "
            f"If it is electrically conductive, place it in the {conductive} box. "
            f"If it is electrically nonconductive, place it in the {nonconductive} box. ")


def view(actions=(), text="A shared initial room", desc=None):
    return make_controller_view(task_description=desc or description(), observation=text,
                                legal_actions=actions)


def test_full_view_distinguishes_equal_looks_with_different_task_instructions():
    assert view(desc=description("B")).sha256() != view(desc=description("O")).sha256()
    assert view(actions=["wait1", "look around"]).sha256() == view(actions=["look around", "wait1"]).sha256()
    assert view(actions=["wait1"]).sha256() != view(actions=[]).sha256()


def test_explicit_fields_reject_metadata_and_mutable_or_nontext_payloads():
    with pytest.raises(TypeError):
        make_controller_view(task_description=description(), observation="room", legal_actions=[], info={"score": 1})
    with pytest.raises(TypeError):
        make_controller_view(task_description=description(), observation={}, legal_actions=[])
    with pytest.raises(TypeError):
        make_controller_view(task_description=description(), observation="room", legal_actions=[{"score": 1}])
    with pytest.raises(TypeError):
        make_controller_view(task_description=description(), observation="room", legal_actions="look around")


def test_task_parser_extracts_class_mapping_but_no_hidden_target():
    assert parse_task(description()) == ConductivityTask("unknown substance B", "workshop", "red box", "green box")
    assert parse_task("Task Description:\n" + description()) == parse_task(description())
    with pytest.raises(ValueError):
        parse_task(description().replace("First, focus on the unknown substance B", "First, focus on the unknown substance C"))
    with pytest.raises(ValueError):
        parse_task(description(conductive="red", nonconductive="red"))


def test_source_combination_structure_makes_official_train_dev_descriptions_disjoint():
    # Public source ordering only, no simulator/seed/hidden target reconstruction.
    letters = "BCDEFGHJKLMNOPQRSTUVWXYZ"
    colors = ("red", "green", "blue", "orange", "yellow", "purple")
    rows = [description(letter, colors[b], colors[b + 1])
            for letter in letters for _ in range(5) for b in range(5)]
    assert len(rows) == 600
    assert set(rows[:300]).isdisjoint(rows[300:450])
    assert len(set(rows[:300])) == 60
    assert len(set(rows[300:450])) == 30


def _kit_view():
    refs = [f"{t} in {name}" for name, ts in [
        ("battery in workshop", ["anode", "cathode"]),
        ("blue light bulb in table", ["anode", "cathode"]),
        *[(w + " wire in table", ["terminal 1", "terminal 2"]) for w in ["red", "blue", "green"]]
    ] for t in ts]
    return view([f"connect {a} to {b}" for a in refs for b in refs if a != b])


def test_circuit_kit_uses_only_visible_free_legal_terminal_referents():
    kit = choose_kit(_kit_view())
    assert kit == CircuitKit("battery in workshop", "blue light bulb in table",
                             ("blue wire in table", "green wire in table", "red wire in table"))
    with pytest.raises(ValueError):
        choose_kit(view(["look around"]))


def test_series_circuit_consumes_each_device_terminal_once_and_two_sample_contacts():
    kit = choose_kit(_kit_view())
    pairs = measurement_connections(parse_task(description()), kit)
    endpoints = [endpoint for pair in pairs for endpoint in pair]
    assert len(pairs) == 6
    sample = Endpoint("unknown substance B", None)
    assert endpoints.count(sample) == 2
    assert len({ep for ep in endpoints if ep != sample}) == 10
    assert pairs[-1][1] == Endpoint(kit.battery, "cathode")
    assert pairs[2][0] == Endpoint(kit.indicator, "anode")


def test_connection_selects_actual_legal_action_and_fails_when_unavailable():
    a, b = Endpoint("battery in workshop", "anode"), Endpoint("blue wire in table", "terminal 1")
    action = "connect terminal 1 in blue wire in table to anode in battery in workshop"
    assert select_connection(view([action]), (a, b)) == action
    with pytest.raises(ValueError):
        select_connection(view([action.replace("anode", "cathode")]), (a, b))


def test_indicator_parser_rejects_ambiguous_or_wrong_device_and_retains_negative_reading():
    assert indicator_on("a blue light bulb, which is on.", "blue light bulb in table") is True
    assert indicator_on("a blue light bulb, which is off.", "blue light bulb") is False
    assert indicator_on("a red light bulb, which is on.", "blue light bulb") is None
    assert indicator_on("a blue light bulb, which is off. a blue light bulb, which is on", "blue light bulb") is None


def test_endpoint_actions_use_visible_target_and_instruction_box_mapping():
    task = parse_task(description())
    observation = view(["focus on unknown substance B in workshop", "move unknown substance B to red box", "move unknown substance B to green box"])
    assert select_focus(observation, task) == "focus on unknown substance B in workshop"
    assert select_answer(observation, task, True).endswith("red box")
    assert select_answer(observation, task, False).endswith("green box")
    with pytest.raises(TypeError):
        select_answer(observation, task, 1)


def test_actual_lowercase_legal_referents_preserve_uppercase_instruction_target():
    task = parse_task(description())
    v = view(["focus on unknown substance b in workshop", "move unknown substance b to red box",
              "connect unknown substance b to terminal 1 in blue wire"])
    assert select_focus(v, task) == "focus on unknown substance b in workshop"
    assert select_answer(v, task, True) == "move unknown substance b to red box"
    assert select_connection(v, (Endpoint(task.target, None), Endpoint("blue wire", "terminal 1"))) == "connect unknown substance b to terminal 1 in blue wire"


def test_unique_generic_alias_executes_focus_connect_and_place_without_changing_target():
    task = parse_task(description())
    v = view(["focus on unknown substance", "move unknown substance to red box",
              "connect unknown substance to terminal 1 in blue wire"])
    assert resolve_target_ref(v, task.target) == "unknown substance"
    assert select_focus(v, task) == "focus on unknown substance"
    assert select_answer(v, task, True) == "move unknown substance to red box"
    assert select_connection(v, (Endpoint(task.target, None), Endpoint("blue wire", "terminal 1"))) == "connect unknown substance to terminal 1 in blue wire"
    assert task.target == "unknown substance B"


@pytest.mark.parametrize("refs", [
    ["unknown substance", "unknown substance c"],
    ["unknown substance in table", "unknown substance on desk"],
    ["unknown substance c"],
    [],
])
def test_generic_target_alias_fails_closed_for_competing_or_missing_unknowns(refs):
    v = view(["focus on " + ref for ref in refs])
    with pytest.raises(ValueError):
        resolve_target_ref(v, "unknown substance B")


def test_specific_target_is_valid_among_multiple_unknowns_but_never_an_arbitrary_one():
    v = view(["focus on unknown substance b", "focus on unknown substance c"])
    assert resolve_target_ref(v, "unknown substance B") == "unknown substance b"
    with pytest.raises(ValueError):
        resolve_target_ref(v, "unknown substance D")
