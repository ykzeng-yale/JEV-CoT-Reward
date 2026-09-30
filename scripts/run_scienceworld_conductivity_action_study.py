#!/usr/bin/env python3
"""Frozen non-model circuit policy census on ScienceWorld train/dev only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
from zipfile import ZipFile

from jev_control.scienceworld_conductivity_policy import (
    choose_kit, indicator_on, make_controller_view, measurement_connections,
    parse_task, select_answer, select_connection, select_focus,
)
from jev_control.scienceworld_conductivity_action_audit import evaluate_terminal


def digest(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def view_of(env, observation):
    # Never forward the mixed info dictionary, reward, score, done, or ID.
    return make_controller_view(task_description=env.taskdescription(),
                                observation=observation,
                                legal_actions=env.get_valid_action_object_combinations())


def evaluator_label(env):
    """Evaluator/calibration only; the policy never receives this object."""
    values = []
    for m in env.server.agentInterface().get().task().taskModifiers():
        if str(m.getClass().getSimpleName()) == "TaskValueBool" and str(m.key()) == "unknownIsConductive":
            values.append(m.value())
    if len(values) != 1 or type(values[0]) is not bool:
        raise ValueError("source-label contract mismatch")
    return values[0]


def read_room(text):
    match = re.search(r"(?:This room is called the |This outside location is called the |You move to the )([^\n.]+)\.", text)
    return match[1] if match else None


def navigation_action(view, target):
    # Visible doors only. Priority is a fixed public house-navigation heuristic,
    # not the simulator's gold path or private location/object tree.
    for destination in (target, "hallway", "kitchen", "outside", "greenhouse"):
        actions = [a for a in view.legal_actions if a == "go to " + destination]
        if actions:
            return actions[0]
        doors = [a for a in view.legal_actions if a.startswith("open door")
                 and re.search(r"\b" + re.escape(destination) + r"\b", a)]
        if doors:
            return doors[0]
    raise ValueError("no visible navigation action under frozen route heuristic")


def run_episode(env, protocol, variation, policy, prior, driver_event=None):
    started = time.monotonic()
    env.load(protocol["task"], variationIdx=variation, simplificationStr="", generateGoldPath=False)
    observation, _ = env.reset()
    view = view_of(env, observation)
    task = parse_task(view.task_description)
    initial = view.sha256()
    trace = []
    prediction = chosen_box = failure = decision_index = None
    circuit_actions = 0
    intrinsic_actions = None
    reading_hash = None
    step_limit = protocol["limits"]["episode_action_budget"]

    def take(action):
        nonlocal view
        if len(trace) >= step_limit:
            raise ValueError("episode action budget exhausted")
        if action not in view.legal_actions:
            raise ValueError("policy selected action outside visible legal list")
        if driver_event:
            driver_event({"event": "action_started", "action": action, "action_index": len(trace)})
        try:
            response, _, _, _ = env.step(action)
            view = view_of(env, response)
        except Exception as exc:
            # An adapter/backend failure is not a scientifically wrong policy
            # outcome. The driver preserves the attempted action and aborts.
            raise RuntimeError("environment step or visible-view adapter failed") from exc
        trace.append({"action": action, "view_sha256": view.sha256()})
        if driver_event:
            driver_event({"event": "action_completed", "action": action,
                          "action_index": len(trace) - 1, "view_sha256": view.sha256()})
        return response

    try:
        room = read_room(view.observation)
        for _ in range(protocol["limits"]["max_navigation_actions"]):
            if room == task.location:
                break
            take(navigation_action(view, task.location))
            room = read_room(view.observation) or room
        if room != task.location:
            raise ValueError("navigation budget exhausted")
        take(select_focus(view, task))
        navigation_and_focus = len(trace)
        if policy == "continue_prior":
            prediction = prior
            intrinsic_actions = navigation_and_focus + 1
            while len(trace) < step_limit - 1:
                take("wait1")
        else:
            kit = choose_kit(view)
            for pair in measurement_connections(task, kit):
                take(select_connection(view, pair))
                circuit_actions += 1
            for _ in range(protocol["limits"]["settling_wait_actions"]):
                take("wait1")
            intrinsic_actions = len(trace) + 2  # inspect + final placement
            while len(trace) < step_limit - 2:
                take("wait1")
            inspections = [prefix + kit.indicator for prefix in ("look at ", "examine ")
                           if prefix + kit.indicator in view.legal_actions]
            if not inspections:
                raise ValueError("indicator has no unambiguous visible examine action")
            inspect = inspections[0]
            reading = take(inspect)
            reading_hash = digest(reading)
            if policy == "measurement":
                prediction = indicator_on(reading, kit.indicator)
                if prediction is None:
                    raise ValueError("indicator reading absent or ambiguous")
            elif policy == "masked_measurement":
                prediction = prior  # reading is never sent to the decoder
            elif policy == "random_measurement":
                prediction = bool(int(digest(initial + ":" + str(protocol["random_seed"])), 16) % 2)
            else:
                raise ValueError("unknown policy")
        chosen_box = task.conductive_box if prediction else task.nonconductive_box
        answer = select_answer(view, task, prediction)
        decision_index = len(trace)
        take(answer)
    except (ValueError, TypeError) as exc:
        failure = str(exc)
        prediction = chosen_box = decision_index = None
        intrinsic_actions = None
    # Private state is read only after the policy has irrevocably stopped.
    endpoint = evaluate_terminal(env, task.target, chosen_box, prediction)
    return {
        "variation_id": variation, "split": "dev" if variation >= 300 else "train_smoke",
        "policy": policy, "initial_view_sha256": initial,
        "initial_look_sha256": digest(observation), "target_group": task.target,
        "trace": trace, "prediction": prediction, "chosen_box": chosen_box,
        "decision_action_index": decision_index, "endpoint": endpoint,
        "actions": len(trace), "intrinsic_action_count": intrinsic_actions,
        "circuit_actions": circuit_actions, "reading_sha256": reading_hash,
        "wall_seconds": time.monotonic() - started, "failure": failure,
    }


def census(env, protocol):
    rows = []
    seen = {"look": {}, "full": {}}
    train_positive = 0
    for split in ("train", "dev"):
        spec = protocol["splits"][split]
        for variation in range(spec["start_inclusive"], spec["end_exclusive"]):
            env.load(protocol["task"], variationIdx=variation, simplificationStr="", generateGoldPath=False)
            observation, _ = env.reset()
            view = view_of(env, observation)
            label = evaluator_label(env)
            if split == "train":
                train_positive += label
            hashes = {"look": digest(observation), "full": view.sha256()}
            row = {"split": split, "variation_id": variation,
                   "task_description_sha256": digest(view.task_description),
                   "target_group": parse_task(view.task_description).target}
            for key, h in hashes.items():
                row[key + "_sha256"] = h
                row[key + "_label_matches_first"] = label == seen[key].setdefault(h, label)
            rows.append(row)
    return rows, {"train_positive": train_positive, "train_total": 300,
                  "selected_prior": train_positive > 150, "tie_rule": False}


def prepare_runtime(protocol, source):
    for name, record in protocol["source_members"].items():
        path = source / record["path"].removeprefix("scienceworld-source/")
        if digest(path.read_bytes()) != record["sha256"]:
            raise ValueError("pinned source hash mismatch: " + name)
    archive = source / "scienceworld/scienceworld.jar"
    version_path = source / "scienceworld/version.py"
    if not version_path.exists():
        manifest = ZipFile(archive).read("META-INF/MANIFEST.MF").decode()
        version = next(line.split(": ", 1)[1] for line in manifest.splitlines()
                       if line.startswith("Specification-Version:"))
        version_path.write_text("__version__ = " + repr(version) + "\n")
    sys.path.insert(0, str(source))
    from scienceworld import ScienceWorldEnv, __version__
    if __version__ != protocol["expected_scienceworld_version"]:
        raise ValueError("runtime version mismatch")
    return ScienceWorldEnv


def save_new(path, obj, cap):
    raw = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if len(raw.encode()) > cap:
        raise ValueError("result output cap exceeded")
    with path.open("x") as f:
        f.write(raw)


def run(protocol_path, source, output):
    output.mkdir(parents=True, exist_ok=False)
    env = None
    context = {"stage": "read_protocol", "variation_id": None, "policy": None}
    event_bytes = 0
    try:
        protocol = json.loads(protocol_path.read_bytes())
        context["stage"] = "prepare_runtime"
        factory = prepare_runtime(protocol, source)
        env = factory(envStepLimit=100)
        context["stage"] = "verify_split"
        env.load(protocol["task"], variationIdx=0, simplificationStr="", generateGoldPath=False)
        if list(env.get_variations_train()) != list(range(300)) or list(env.get_variations_dev()) != list(range(300, 450)):
            raise ValueError("official train/dev split mismatch")
        context["stage"] = "input_census"
        rows, calibration = census(env, protocol)
        save_new(output / "input-census.json", {"records": rows, "calibration": calibration}, 2000000)
        events_path = output / "infrastructure-events.jsonl"
        events_path.touch(exist_ok=False)

        def driver_event(event):
            nonlocal event_bytes
            raw = json.dumps({**context, **event, "monotonic_seconds": time.monotonic()}, sort_keys=True) + "\n"
            event_bytes += len(raw.encode())
            if event_bytes > protocol["limits"].get("max_diagnostic_bytes", 32000000):
                raise RuntimeError("infrastructure diagnostic output cap exceeded")
            with events_path.open("a") as stream:
                stream.write(raw)
                stream.flush()

        # One fixed training integration gate. It checks executability only;
        # wrong scientific classifications are never grounds to retry or tune.
        smoke = []
        context["stage"] = "training_integration"
        context["policy"] = "measurement"
        for variation in protocol["training_integration_ids"]:
            context["variation_id"] = variation
            smoke.append(run_episode(env, protocol, variation, "measurement",
                                     calibration["selected_prior"], driver_event))
        save_new(output / "training-integration.json", smoke, 1000000)
        if any(row["failure"] for row in smoke):
            raise RuntimeError("training integration failed; no dev action outcomes executed")
        episodes = []
        context["stage"] = "development_episodes"
        with (output / "episodes.jsonl").open("x") as durable:
            for variation in range(300, 450):
                for policy in protocol["policies"]:
                    context.update(variation_id=variation, policy=policy)
                    row = run_episode(env, protocol, variation, policy, calibration["selected_prior"], driver_event)
                    durable.write(json.dumps(row, sort_keys=True) + "\n")
                    durable.flush()
                    episodes.append(row)
        result = {"protocol": protocol["protocol"], "protocol_sha256": digest(protocol_path.read_bytes()),
                  "input_census": rows, "calibration": calibration, "episodes": episodes,
                  "training_integration": smoke, "test_loaded": 0, "model_calls": 0,
                  "jev_calls": 0, "gold_paths_requested": False,
                  "controller_fields": ["task_description", "observation", "legal_actions"],
                  "api_score_computation_note": "The stock step adapter computes reward/score internally; neither it, done, the mixed info dictionary, nor private source labels is forwarded to the controller.",
                  "scope": "Fixed development-panel local measurement baseline; not Jev efficacy, independent-family transfer, novel theory, or confirmation."}
        save_new(output / "result.json", result, protocol["limits"]["max_result_bytes"])
        return result
    except BaseException as exc:
        # This is an infrastructure diagnostic, never a PASS or a policy row.
        # Traceback is local to the immutable run and includes exception causes.
        save_new(output / "driver-failure.json",
                 {**context, "status": "driver_failure", "exception_type": type(exc).__name__,
                  "traceback": traceback.format_exc(), "result_valid": False}, 1000000)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    r = run(args.protocol, args.source_root, args.output_dir)
    print(json.dumps({"status": "complete_uninterpreted", "episodes": len(r["episodes"]), "test_loaded": 0}))
