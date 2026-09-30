from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("sw_action_analysis", Path(__file__).parents[1] / "scripts/analyze_scienceworld_conductivity_action_study.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
analyze = module.analyze


@pytest.fixture
def files(tmp_path):
    protocol = {"protocol": "synthetic-panel", "policies": ["measurement", "masked_measurement", "continue_prior", "random_measurement"],
                "splits": {"dev": {"start_inclusive": 300, "end_exclusive": 450}}}
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(protocol))
    rows = []
    for i in range(150):
        for policy in protocol["policies"]:
            success = policy == "measurement" and i >= 25
            rows.append({"variation_id": 300 + i, "split": "dev", "policy": policy,
                         "target_group": "group" + str(i // 25),
                         "endpoint": {"task_success": success, "placement_correct": success, "classification_correct": success},
                         "failure": "synthetic failure" if i < 25 else None,
                         "decision_action_index": None if i < 25 else 31,
                         "reading_sha256": None if i < 25 or policy == "continue_prior" else "0" * 64,
                         "circuit_actions": 0 if policy == "continue_prior" else 6,
                         "actions": 32, "intrinsic_action_count": None if i < 25 else 12,
                         "wall_seconds": 0.5})
    result = {"protocol": protocol["protocol"], "protocol_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
              "episodes": rows, "test_loaded": 0, "model_calls": 0, "jev_calls": 0, "gold_paths_requested": False,
              "calibration": {"train_positive": 130, "train_total": 300, "selected_prior": False, "tie_rule": False}}
    r, a = tmp_path / "result.json", tmp_path / "audit.json"
    r.write_text(json.dumps(result))
    audit = {"passes": True, "record_checks_pass": True, "replay_performed": True,
             "replay_episodes_verified": 600, "replay_census_rows_verified": 450,
             "replay_error": None, "checks": {"synthetic_check": True},
             "result_sha256": hashlib.sha256(r.read_bytes()).hexdigest(),
             "protocol_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "input_census": {"synthetic": True}}
    a.write_text(json.dumps(audit))
    return p, r, a


def rewrite(path, mutation):
    obj = json.loads(path.read_text())
    mutation(obj)
    path.write_text(json.dumps(obj))


def rebind(files):
    p, r, a = files
    rewrite(a, lambda obj: obj.update(result_sha256=hashlib.sha256(r.read_bytes()).hexdigest(), protocol_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))


def test_full_panel_failures_pairing_and_actual_costs_retained(files):
    report = analyze(*files)
    arm = report["policy_summary"]["measurement"]
    assert arm["assigned_variations"] == arm["analyzed_variations"] == 150
    assert arm["failures"] == 25
    assert arm["outcomes"]["task_success"] == {"successes": 125, "denominator": 150, "rate": 125 / 150}
    assert arm["costs"]["executed_actions_including_padding_and_failures"] == 4800
    assert arm["costs"]["intrinsic_actions_known_total"] == 1500
    assert arm["costs"]["intrinsic_count_unknown_episodes"] == 25
    assert arm["costs"]["padding_actions_among_known_episodes"] == 2500
    assert arm["costs"]["summed_episode_wall_seconds"] == 75
    contrast = report["paired_contrasts"]["measurement-minus-masked_measurement"]["outcomes"]["task_success"]
    assert contrast["wins"] == 125 and contrast["losses"] == 0 and contrast["ties"] == 25
    assert contrast["mean_paired_difference"] == 125 / 150
    sensitivity = contrast["descriptive_group_sensitivity"]
    assert sensitivity["resamples"] == 46656
    assert sensitivity["interval"] == [0.5, 1.0]
    assert sensitivity["population_confidence_interval"] is False
    assert report["scheduler_costs"]["allocated_gpu_hours"] is None


@pytest.mark.parametrize("field,value", [("passes", False), ("record_checks_pass", False),
                                         ("replay_performed", False), ("replay_episodes_verified", 599),
                                         ("replay_census_rows_verified", 449), ("replay_error", "failure"),
                                         ("checks", {"runtime": False}), ("checks", {})])
def test_rejects_failed_missing_or_partial_replay_before_rates(files, field, value):
    rewrite(files[2], lambda obj: obj.update({field: value}))
    with pytest.raises(ValueError, match="passing independent"):
        analyze(*files)


@pytest.mark.parametrize("file_index", [0, 1])
def test_changed_result_or_protocol_rejected(files, file_index):
    with files[file_index].open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed after"):
        analyze(*files)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "split", "unequal_groups", "calls"])
def test_panel_contract_defended_even_against_rebound_fake_audit(files, mutation):
    def change(obj):
        if mutation == "missing":
            obj["episodes"].pop()
        elif mutation == "duplicate":
            obj["episodes"][-1] = obj["episodes"][0]
        elif mutation == "split":
            obj["episodes"][0]["split"] = "test"
        elif mutation == "unequal_groups":
            for row in obj["episodes"][:4]:
                row["target_group"] = "group1"
        else:
            obj["model_calls"] = 1
    rewrite(files[1], change)
    rebind(files)
    with pytest.raises(ValueError):
        analyze(*files)


def test_exact_six_cluster_sensitivity_is_not_iid_150_interval():
    positive = module.exact_group_bootstrap([0, 0, 0, 0, 0, 1])
    negative = module.exact_group_bootstrap([0, 0, 0, 0, 0, -1])
    assert positive["interval"] == [0, 0.5]
    assert negative["interval"] == [-0.5, 0]
    assert module.exact_group_bootstrap([0] * 6)["interval"] == [0, 0]
    with pytest.raises(ValueError, match="six"):
        module.exact_group_bootstrap([0] * 150)
