"""Evaluator-only terminal checks for the ScienceWorld conductivity task.

Call ``evaluate_terminal`` only after the policy has irrevocably ended. This
module reads hidden task modifiers and runtime objects; its result must never
be supplied to a policy. It intentionally does not use step reward, rounded
score, ``done``, or a gold action sequence as an outcome label.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterator


ENDPOINT_FIELDS = frozenset({
    "classification_correct", "placement_correct", "chosen_box_matches_placement",
    "ordered_goal_complete", "task_failed", "task_success", "target_present", "target_unique",
})
EPISODE_FIELDS = frozenset({
    "variation_id", "split", "policy", "initial_view_sha256", "initial_look_sha256",
    "trace", "prediction", "chosen_box", "endpoint", "actions", "wall_seconds",
    "failure", "decision_action_index",
    "target_group", "intrinsic_action_count", "circuit_actions", "reading_sha256",
})
CENSUS_FIELDS = frozenset({
    "split", "variation_id", "task_description_sha256", "target_group", "look_sha256",
    "full_sha256", "look_label_matches_first", "full_label_matches_first",
})


def _scala_items(collection: Any) -> Iterator[Any]:
    """Handle Scala collections through their documented iterator accessors."""
    iterator = collection.iterator()
    while iterator.hasNext():
        yield iterator.next()


def _source_values(task: Any) -> dict[str, Any]:
    required = {
        "unknownSubstance": "TaskValueStr",
        "unknownIsConductive": "TaskValueBool",
        "conductive": "TaskValueStr",
        "nonconductive": "TaskValueStr",
    }
    values: dict[str, Any] = {}
    for modifier in task.taskModifiers():
        kind = str(modifier.getClass().getSimpleName())
        if kind not in {"TaskValueStr", "TaskValueBool"}:
            continue
        key = str(modifier.key())
        if key not in required:
            continue
        if kind != required[key] or key in values:
            raise ValueError("source task modifiers are ambiguous or mistyped")
        value = modifier.value()
        if key == "unknownIsConductive":
            if type(value) is not bool:
                raise TypeError("source conductivity label is not a Boolean")
        elif not isinstance(value, str) or not value:
            raise TypeError("source task name is not a nonempty string")
        values[key] = value
    if set(values) != set(required):
        raise ValueError("source task lacks required conductivity modifiers")
    if values["conductive"].casefold() == values["nonconductive"].casefold():
        raise ValueError("conductivity answer boxes are not distinct")
    return values


def evaluate_terminal(
    env: Any,
    target_name: str,
    chosen_box: str | None,
    prediction: bool | None,
) -> dict[str, bool]:
    """Independently check the frozen policy's final classification and action.

    The caller provides only already-frozen policy outputs. Actual placement is
    checked in the runtime object tree, not inferred from the requested move or
    its response. Source ``GoalObjectInContainerByName`` examines up to twenty
    ancestors and gives an incorrect answer container precedence over a correct
    one. Ordered-goal completion and failure are checked separately, so reward
    from optional circuit-building subgoals cannot count as task success.

    Labels, their hashes, correct box names, and private object details are not
    returned. A missing/duplicate runtime target is an unsuccessful placement;
    a mismatched source target or task structure rejects the evaluator call.
    """
    if not isinstance(target_name, str) or not target_name:
        raise TypeError("target_name must be a nonempty policy-visible string")
    if chosen_box is not None and (not isinstance(chosen_box, str) or not chosen_box):
        raise TypeError("chosen_box must be a nonempty string or None")
    if prediction is not None and type(prediction) is not bool:
        raise TypeError("prediction must be Boolean or None")

    interface = env.server.agentInterface().get()
    task = interface.task()
    values = _source_values(task)
    if target_name.casefold() != values["unknownSubstance"].casefold():
        raise ValueError("policy target differs from source-defined task target")
    goals = task.goalSequence()
    goal_classes = [str(goal.getClass().getSimpleName()) for goal in goals.subgoals()]
    if goal_classes != ["GoalFind", "GoalObjectInContainerByName"]:
        raise ValueError("unexpected conductivity ordered-goal structure")

    targets = [
        obj for obj in _scala_items(interface.universe().getContainedObjectsRecursive())
        if str(obj.name()).casefold() == values["unknownSubstance"].casefold()
    ]
    ancestors: list[str] = []
    if len(targets) == 1:
        parent = targets[0].getContainer()
        # Match the simulator goal's explicit twenty-ancestor bound.
        for _ in range(20):
            if not parent.isDefined():
                break
            obj = parent.get()
            ancestors.append(str(obj.name()).casefold())
            parent = obj.getContainer()

    label = values["unknownIsConductive"]
    correct_box = values["conductive" if label else "nonconductive"].casefold()
    wrong_box = values["nonconductive" if label else "conductive"].casefold()
    placement_correct = correct_box in ancestors and wrong_box not in ancestors
    ordered_goal_complete = bool(goals.isCompleted())
    task_failed = bool(goals.isFailed())
    return {
        "classification_correct": prediction is not None and prediction == label,
        "placement_correct": placement_correct,
        "chosen_box_matches_placement": chosen_box is not None and chosen_box.casefold() in ancestors,
        "ordered_goal_complete": ordered_goal_complete,
        "task_failed": task_failed,
        "task_success": placement_correct and ordered_goal_complete and not task_failed,
        "target_present": bool(targets),
        "target_unique": len(targets) == 1,
    }


def training_label(env: Any, variation_id: int, train_ids: list[int]) -> bool:
    """Read a training-only label for frozen majority-prior calibration.

    Calibration must finish before development policies run. Caller supplies
    the protocol's official training IDs, and the loaded environment identity
    must match the requested ID. Never send this function to a policy.
    """
    if type(variation_id) is not int or variation_id not in train_ids:
        raise ValueError("label calibration requested outside frozen training IDs")
    if env.variationIdx != variation_id:
        raise ValueError("loaded environment differs from requested training ID")
    return _source_values(env.server.agentInterface().get().task())["unknownIsConductive"]


def _full_view_hash(env: Any, observation: str) -> str:
    # Authored separately from the policy adapter: compare exact canonical
    # field values rather than trusting a stored policy digest or mixed info.
    view = {
        "task_description": env.taskdescription(),
        "observation": observation,
        "legal_actions": sorted(set(env.get_valid_action_object_combinations())),
    }
    raw = json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _replay_episode(env: Any, protocol: dict, row: dict) -> dict:
    env.load(protocol["task"], variationIdx=row["variation_id"],
             simplificationStr="", generateGoldPath=False)
    observation, _ = env.reset()
    if _full_view_hash(env, observation) != row["initial_view_sha256"]:
        raise ValueError("independent replay initial full-view mismatch")
    if hashlib.sha256(observation.encode("utf-8")).hexdigest() != row["initial_look_sha256"]:
        raise ValueError("independent replay initial look mismatch")
    # Read only the policy-visible task sentence to identify its target.
    target = re.search(r"Your task is to determine if (unknown substance [A-Z]) is electrically conductive\.",
                       env.taskdescription())
    if target is None:
        raise ValueError("independent replay cannot identify visible task target")
    if target.group(1) != row["target_group"]:
        raise ValueError("independent replay target group differs")
    reading_hash = None
    for step in row["trace"]:
        if step["action"] not in env.get_valid_action_object_combinations():
            raise ValueError("independent replay action was not publicly legal")
        observation, _reward, _done, _info = env.step(step["action"])
        if _full_view_hash(env, observation) != step["view_sha256"]:
            raise ValueError("independent replay post-action view mismatch")
        if step["action"].startswith(("look at ", "examine ")):
            reading_hash = hashlib.sha256(observation.encode("utf-8")).hexdigest()
    if reading_hash != row["reading_sha256"]:
        raise ValueError("independent replay measurement-reading digest mismatch")
    return evaluate_terminal(env, target.group(1), row["chosen_box"], row["prediction"])


def _census_record_checks(protocol: dict, result: dict) -> tuple[dict, dict]:
    rows = result.get("input_census", [])
    expected = [(split, variation) for split in ("train", "dev")
                for variation in range(protocol["splits"][split]["start_inclusive"],
                                       protocol["splits"][split]["end_exclusive"])]
    calibration = result.get("calibration", {})
    checks = {"census_schema": isinstance(rows, list), "census_coverage_order": True,
              "calibration_schema_and_majority": False, "census_first_label_flags": True}
    actual = []
    first_hashes = {"look": set(), "full": set()}
    groups: dict[str, dict[str, list]] = {"look": defaultdict(list), "full": defaultdict(list)}
    for row in rows if isinstance(rows, list) else []:
        valid = (isinstance(row, dict) and set(row) == CENSUS_FIELDS
                 and type(row["variation_id"]) is int and row["split"] in ("train", "dev")
                 and isinstance(row["target_group"], str)
                 and bool(re.fullmatch(r"unknown substance [A-Z]", row["target_group"]))
                 and all(isinstance(row[key], str) and re.fullmatch(r"[0-9a-f]{64}", row[key])
                         for key in ("task_description_sha256", "look_sha256", "full_sha256"))
                 and all(type(row[key]) is bool for key in ("look_label_matches_first", "full_label_matches_first")))
        if not valid:
            checks["census_schema"] = False
            continue
        actual.append((row["split"], row["variation_id"]))
        for kind in ("look", "full"):
            h = row[kind + "_sha256"]
            if h not in first_hashes[kind]:
                checks["census_first_label_flags"] &= row[kind + "_label_matches_first"] is True
                first_hashes[kind].add(h)
            groups[kind][h].append(row)
    checks["census_coverage_order"] = actual == expected
    train_total = sum(split == "train" for split, _ in expected)
    if isinstance(calibration, dict) and set(calibration) == {"train_positive", "train_total", "selected_prior", "tie_rule"}:
        positive = calibration["train_positive"]
        checks["calibration_schema_and_majority"] = (
            type(positive) is int and 0 <= positive <= train_total
            and type(calibration["train_total"]) is int and calibration["train_total"] == train_total
            and type(calibration["selected_prior"]) is bool
            and calibration["selected_prior"] == (positive > train_total / 2)
            and calibration["tie_rule"] is False
        )
    aggregate: dict[str, Any] = {}
    if all(checks.values()):
        for kind in ("look", "full"):
            shared = [values for values in groups[kind].values() if {row["split"] for row in values} == {"train", "dev"}]
            aggregate[kind] = {
                "unique_groups": len(groups[kind]), "shared_train_dev_groups": len(shared),
                "shared_train_dev_rows": sum(len(values) for values in shared),
                "shared_dev_rows": sum(row["split"] == "dev" for values in shared for row in values),
                "conflicting_label_groups": sum(any(not row[kind + "_label_matches_first"] for row in values)
                                                for values in groups[kind].values()),
                "shared_conflicting_label_groups": sum(any(not row[kind + "_label_matches_first"] for row in values)
                                                       for values in shared),
            }
        targets = {split: {row["target_group"] for row in rows if row["split"] == split} for split in ("train", "dev")}
        aggregate["target_groups"] = {"train": len(targets["train"]), "dev": len(targets["dev"]),
                                      "shared": len(targets["train"] & targets["dev"])}
    return checks, aggregate


def _replay_census(env: Any, protocol: dict, result: dict) -> int:
    anchors: dict[str, dict[str, bool]] = {"look": {}, "full": {}}
    positives = 0
    checked = 0
    for row in result["input_census"]:
        env.load(protocol["task"], variationIdx=row["variation_id"], simplificationStr="", generateGoldPath=False)
        observation, _ = env.reset()
        text = env.taskdescription()
        target = re.search(r"Your task is to determine if (unknown substance [A-Z]) is electrically conductive\.", text)
        if target is None or target.group(1) != row["target_group"]:
            raise ValueError("independent census task target mismatch")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["task_description_sha256"]:
            raise ValueError("independent census task description mismatch")
        hashes = {"look": hashlib.sha256(observation.encode("utf-8")).hexdigest(), "full": _full_view_hash(env, observation)}
        label = _source_values(env.server.agentInterface().get().task())["unknownIsConductive"]
        positives += row["split"] == "train" and label
        for kind, h in hashes.items():
            if h != row[kind + "_sha256"]:
                raise ValueError("independent census input hash mismatch")
            if h not in anchors[kind]:
                anchors[kind][h] = label
            if (label == anchors[kind][h]) != row[kind + "_label_matches_first"]:
                raise ValueError("independent census relative label mismatch")
        checked += 1
    if positives != result["calibration"]["train_positive"]:
        raise ValueError("independent training-majority calibration mismatch")
    return checked


def audit(protocol: dict, result: dict, *, env_factory: Callable[[], Any] | None = None) -> dict:
    """Audit the complete frozen paired panel, optionally replaying every trace.

    Without ``env_factory``, ``record_checks_pass`` may be true but ``passes``
    stays false: record consistency alone cannot certify runtime outcomes.
    All failed episodes remain in each policy's denominator and resource sums.
    """
    dev = protocol["splits"]["dev"]
    ids = list(range(dev["start_inclusive"], dev["end_exclusive"]))
    policies = protocol["policies"]
    cap = protocol["limits"]["episode_action_budget"]
    episodes = result.get("episodes", [])
    checks = {"episode_schema": isinstance(episodes, list), "coverage_exact": True,
              "trace_and_cost_schema": True, "endpoint_consistency": True,
              "paired_initial_views_equal": True, "measurement_prefixes_equal": True}
    census_checks, census_aggregates = _census_record_checks(protocol, result)
    checks.update(census_checks)
    checks["episodes_match_census"] = True
    by_key: dict[tuple[int, str], dict] = {}
    hex64 = re.compile(r"[0-9a-f]{64}\Z")
    for row in episodes if isinstance(episodes, list) else []:
        if not isinstance(row, dict) or set(row) != EPISODE_FIELDS:
            checks["episode_schema"] = False
            continue
        valid_key = (type(row["variation_id"]) is int and row["variation_id"] in ids
                     and row["split"] == "dev" and row["policy"] in policies)
        if not valid_key:
            checks["coverage_exact"] = False
            continue
        key = row["variation_id"], row["policy"]
        if key in by_key:
            checks["coverage_exact"] = False
        by_key[key] = row
        trace = row["trace"]
        n = row["actions"]
        wall = row["wall_seconds"]
        decision = row["decision_action_index"]
        trace_ok = (isinstance(trace, list) and type(n) is int and 0 <= n <= cap and n == len(trace)
                    and type(wall) in (int, float) and math.isfinite(wall) and wall >= 0
                    and isinstance(row["initial_view_sha256"], str) and bool(hex64.fullmatch(row["initial_view_sha256"]))
                    and isinstance(row["initial_look_sha256"], str) and bool(hex64.fullmatch(row["initial_look_sha256"])))
        trace_ok &= (isinstance(row["target_group"], str) and bool(re.fullmatch(r"unknown substance [A-Z]", row["target_group"]))
                     and (row["intrinsic_action_count"] is None or
                          (type(row["intrinsic_action_count"]) is int and 0 <= row["intrinsic_action_count"] <= n))
                     and type(row["circuit_actions"]) is int and row["circuit_actions"] >= 0
                     and (row["reading_sha256"] is None or
                          isinstance(row["reading_sha256"], str) and bool(hex64.fullmatch(row["reading_sha256"]))))
        if trace_ok:
            trace_ok = all(isinstance(step, dict) and set(step) == {"action", "view_sha256"}
                           and isinstance(step["action"], str) and bool(step["action"])
                           and isinstance(step["view_sha256"], str) and bool(hex64.fullmatch(step["view_sha256"]))
                           for step in trace)
            trace_ok &= row["circuit_actions"] == sum(step["action"].startswith("connect ") for step in trace)
        if decision is None:
            trace_ok &= (row["prediction"] is None and row["chosen_box"] is None
                         and isinstance(row["failure"], str) and bool(row["failure"]))
        else:
            trace_ok &= (type(decision) is int and decision == n - 1 and decision >= 0
                         and type(row["prediction"]) is bool
                         and isinstance(row["chosen_box"], str) and bool(row["chosen_box"])
                         and row["failure"] is None)
        checks["trace_and_cost_schema"] &= bool(trace_ok)
        end = row["endpoint"]
        end_ok = isinstance(end, dict) and set(end) == ENDPOINT_FIELDS and all(type(x) is bool for x in end.values())
        if end_ok:
            end_ok = (
                end["task_success"] == (end["placement_correct"] and end["ordered_goal_complete"] and not end["task_failed"])
                and (not end["target_unique"] or end["target_present"])
                and (not end["placement_correct"] or end["target_unique"])
                and (row["prediction"] is not None or not end["classification_correct"])
                and (row["chosen_box"] is not None or not end["chosen_box_matches_placement"])
                and (decision is not None or not end["task_success"])
            )
        checks["endpoint_consistency"] &= bool(end_ok)
    expected = {(variation, policy) for variation in ids for policy in policies}
    checks["coverage_exact"] &= set(by_key) == expected and len(episodes) == len(expected)
    if all(checks.values()):
        census_dev = {row["variation_id"]: row for row in result["input_census"] if row["split"] == "dev"}
        for variation in ids:
            paired = [by_key[variation, policy] for policy in policies]
            census_row = census_dev[variation]
            checks["episodes_match_census"] &= all(
                row["initial_view_sha256"] == census_row["full_sha256"]
                and row["initial_look_sha256"] == census_row["look_sha256"]
                and row["target_group"] == census_row["target_group"] for row in paired)
            checks["paired_initial_views_equal"] &= len({
                (row["initial_view_sha256"], row["initial_look_sha256"]) for row in paired}) == 1
            measured = [by_key[variation, policy] for policy in
                        ("measurement", "masked_measurement", "random_measurement")]
            prefixes = [row["trace"][:row["decision_action_index"]]
                        if row["decision_action_index"] is not None else row["trace"]
                        for row in measured]
            checks["measurement_prefixes_equal"] &= prefixes[0] == prefixes[1] == prefixes[2]

    record_pass = all(checks.values())
    replay_checked = 0
    census_replayed = 0
    replay_error: str | None = None
    if record_pass and env_factory is not None:
        env = env_factory()
        try:
            census_replayed = _replay_census(env, protocol, result)
            for row in episodes:
                endpoint = _replay_episode(env, protocol, row)
                if endpoint != row["endpoint"]:
                    raise ValueError("independent replay endpoint mismatch")
                replay_checked += 1
        except Exception as exc:
            # Do not include repr of bridge objects/private source values.
            replay_error = f"{type(exc).__name__}: independent runtime replay failed after {replay_checked} episodes"
        finally:
            env.close()

    aggregates: dict[str, Any] = {}
    if record_pass:
        for policy in policies:
            rows = [by_key[variation, policy] for variation in ids]
            aggregates[policy] = {
                "episodes": len(rows), "failures": sum(row["failure"] is not None for row in rows),
                "classification_correct": sum(row["endpoint"]["classification_correct"] for row in rows),
                "placement_correct": sum(row["endpoint"]["placement_correct"] for row in rows),
                "task_success": sum(row["endpoint"]["task_success"] for row in rows),
                "actions": sum(row["actions"] for row in rows),
                "intrinsic_actions": sum(row["intrinsic_action_count"] or 0 for row in rows),
                "circuit_actions": sum(row["circuit_actions"] for row in rows),
                "wall_seconds": sum(row["wall_seconds"] for row in rows),
                "failure_types": dict(Counter(row["failure"] for row in rows if row["failure"] is not None)),
            }
    return {
        "audit": "independent ScienceWorld conductivity action panel audit",
        "checks": checks, "record_checks_pass": record_pass,
        "replay_performed": env_factory is not None,
        "replay_episodes_verified": replay_checked, "replay_error": replay_error,
        "replay_census_rows_verified": census_replayed, "input_census": census_aggregates,
        "passes": record_pass and env_factory is not None and replay_error is None and replay_checked == len(expected),
        "policy_counts": aggregates,
        "interpretation": "Development-only finite panel; no confirmatory efficacy, transfer, or Jev claim. Full pass requires environment replay.",
    }
