"""Finite-mechanism checks with explicit assumptions, not real Jev evidence."""
import importlib.util
from pathlib import Path
import random

import pytest


spec = importlib.util.spec_from_file_location(
    "theory_validation", Path(__file__).resolve().parents[1] / "scripts" / "validate_theory.py")
theory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theory)


def test_exact_sensor_values_and_cost_sensitive_gate():
    result = theory.sensor_example()
    assert result["cheap_policy_value"] == pytest.approx(0.725)
    assert result["always_sensor_value_after_cost"] == pytest.approx(0.765)
    assert result["selective_sensor_value_after_cost"] == pytest.approx(0.780)
    assert result["optimal_query_gate_by_stratum"] == [0, 1]
    costly = theory.sensor_example(query_cost=0.2)
    assert costly["optimal_query_gate_by_stratum"] == [0, 0]
    assert costly["selective_sensor_value_after_cost"] == pytest.approx(costly["cheap_policy_value"])


def test_uninformative_sensor_has_no_free_value():
    result = theory.sensor_example(sensor_accuracy=0.5)
    assert result["always_sensor_value_before_cost"] == pytest.approx(result["cheap_policy_value"])
    assert result["optimal_query_gate_by_stratum"] == [0, 0]


def test_dr_exact_expectation_with_wrong_outcome_model():
    result = theory.dr_exact_check()
    assert result["dr_expectation"] == pytest.approx(result["exact_target"], abs=1e-12)


def test_sequential_gate_and_independently_evaluated_identity():
    rng = random.Random(914)
    trials = [theory.mdp_trial(rng) for _ in range(100)]
    assert sum(t["switches"] for t in trials) > 0
    assert all(t["value_improvement"] >= -1e-12 for t in trials)
    assert all(t["identity_error"] < 1e-12 for t in trials)


def test_group_bootstrap_uses_original_problems():
    # Three seeds per problem are averaged first. Replicating identical seeds
    # must not manufacture additional independent problem observations.
    seeds = [[1, 1, 1], [-1, -1, -1], [0, 0, 0], [1, 0, 0]]
    means = [sum(row) / len(row) for row in seeds]
    replicated = [sum(row * 5) / len(row * 5) for row in seeds]
    assert theory.paired_problem_bootstrap(means) == theory.paired_problem_bootstrap(replicated)
    assert theory.paired_problem_bootstrap(means)["independent_problems"] == 4


def test_problem_bootstrap_rejects_empty_data():
    with pytest.raises(ValueError):
        theory.paired_problem_bootstrap([])


def test_validation_manifest_and_numeric_planning_constants():
    result = theory.validate(mdp_trials=20)
    assert result["all_checks_passed"]
    assert result["status"] == "synthetic_validation_only"
    assert result["replays_for_uniform_01_error_M48_A3_delta05"] == 433
    assert result["hoeffding_radius_n3000_k1_alpha05"] == pytest.approx(0.04469, abs=1e-5)


def test_adaptive_replay_count_can_bias_an_ordinary_arm_mean():
    result = theory.inference_counterexamples()["adaptive_replay_stopping"]
    assert result["true_bernoulli_mean"] == 0.5
    assert result["expected_stopped_sample_mean"] == 0.625
    assert result["bias"] == 0.125


def test_crossfit_exclusion_does_not_make_contributions_independent():
    result = theory.inference_counterexamples()["crossfit_dependence"]
    assert result["cross_problem_contribution_covariance"] == 0.1875
    assert result["actual_variance_of_mean"] == 2 * result["variance_if_incorrectly_independent"]


def test_fixed_contribution_bootstrap_has_no_generic_crossfit_coverage():
    result = theory.inference_counterexamples()["crossfit_dependence"]
    assert result["mean_procedure_value"] == 0.25
    assert result["fixed_contribution_percentile_bootstrap_coverage"] == 0.0


def test_empirical_best_arm_is_not_expected_value_oracle():
    result = theory.inference_counterexamples()["noisy_oracle_selection"]
    assert result["expected_maximum_of_two_single_rollout_outcomes"] == 0.75
    assert result["fresh_rollout_value_of_selected_arm"] == result["equal_arm_true_success"] == 0.5
