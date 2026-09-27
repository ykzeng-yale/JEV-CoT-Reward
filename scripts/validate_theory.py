#!/usr/bin/env python3
"""Exact finite examples and seeded proof-condition checks; no LLM/API use.

These checks can detect mistakes in formulas/implementation, not prove a theorem
for all models or predict Jev's real empirical value. Uses only the stdlib.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path


def weighted(values, probabilities):
    return sum(x * p for x, p in zip(values, probabilities, strict=True))


def sensor_example(sensor_accuracy=0.9, query_cost=0.03):
    """Known data-generating mechanism; costs measured in utility units.

    Half of problems are calm, for which continue is already best. The other
    half hide two equiprobable states with opposed repair/continue preferences.
    The sensor is informative only in the ambiguous stratum. No samples fit it.
    """
    calm_q = (0.85, 0.4, 0.5)
    ambiguous_q = ((0.8, 0.2, 0.6), (0.2, 0.8, 0.6))
    cheap_ambiguous = max(sum(row[a] for row in ambiguous_q) / 2 for a in range(3))
    # Integrate posterior utility after each possible binary sensor response.
    rich_ambiguous = 0.0
    for signal in range(2):
        joint = [0.5 * (sensor_accuracy if state == signal else 1 - sensor_accuracy)
                 for state in range(2)]
        rich_ambiguous += max(sum(joint[s] * ambiguous_q[s][a] for s in range(2))
                              for a in range(3))
    cheap = (max(calm_q) + cheap_ambiguous) / 2
    rich_free = (max(calm_q) + rich_ambiguous) / 2
    d = (-query_cost, rich_ambiguous - query_cost - cheap_ambiguous)
    gate = tuple(int(x > 0) for x in d)
    selective = cheap + sum(g * delta / 2 for g, delta in zip(gate, d))
    return {
        "synthetic_only": True,
        "sensor_accuracy_assumption": sensor_accuracy,
        "query_cost_in_utility_units": query_cost,
        "cheap_policy_value": cheap,
        "always_sensor_value_before_cost": rich_free,
        "always_sensor_value_after_cost": rich_free - query_cost,
        "selective_sensor_value_after_cost": selective,
        "acquisition_advantage_by_stratum": list(d),
        "optimal_query_gate_by_stratum": list(gate),
        "ambiguous_stratum_sensor_value_before_cost": rich_ambiguous,
    }


def dr_exact_check():
    """Evaluate DR expectation exactly over two contexts and three actions."""
    state_probs = (0.3, 0.7)
    q = ((0.8, 0.2, 0.6), (0.2, 0.8, 0.6))
    propensity = ((0.1, 0.3, 0.6), (0.5, 0.2, 0.3))
    # Deliberately misspecified outcome model; true propensities suffice.
    model = ((0.1, 0.9, 0.4), (0.6, 0.1, 0.3))
    policy = (0, 1)
    estimate = 0.0
    target = sum(state_probs[s] * q[s][policy[s]] for s in range(2))
    for s in range(2):
        for a in range(3):
            phi = model[s][policy[s]]
            if a == policy[s]:
                phi += (q[s][a] - model[s][a]) / propensity[s][a]
            estimate += state_probs[s] * propensity[s][a] * phi
    return {"exact_target": target, "dr_expectation": estimate,
            "absolute_error": abs(estimate - target)}


def inference_counterexamples():
    """Enumerate all binary outcomes to expose assumption failures exactly.

    These are counterexamples to overbroad inferences, not model experiments.
    The leave-one-out example has two independent Bernoulli outcome rows, but
    each policy decides whether to act from the OTHER row. Its held-out reward
    contributions are consequently identical products, hence dependent.
    """
    cases = list(itertools.product((0.0, 1.0), repeat=2))
    probability = 1.0 / len(cases)
    stopped_mean = sum(probability * (x if x else (x + y) / 2) for x, y in cases)
    loo_rows = [(x * y, y * x) for x, y in cases]
    marginal = sum(probability * pair[0] for pair in loo_rows)
    joint = sum(probability * pair[0] * pair[1] for pair in loo_rows)
    actual_variance = sum(probability * (((a + b) / 2) - marginal) ** 2
                          for a, b in loo_rows)
    # The problem-contribution bootstrap is point-valued because a == b.
    percentile_coverage = sum(probability * int(a <= marginal <= b)
                              for a, b in loo_rows)
    empirical_best = sum(probability * max(x, y) for x, y in cases)
    return {
        "adaptive_replay_stopping": {
            "true_bernoulli_mean": 0.5,
            "rule": "Stop after first success; otherwise take exactly one additional draw.",
            "expected_stopped_sample_mean": stopped_mean,
            "bias": stopped_mean - 0.5,
        },
        "crossfit_dependence": {
            "independent_original_problems": 2,
            "policy_rule": "Choose the positive-reward action iff the other problem's outcome was one.",
            "mean_procedure_value": marginal,
            "cross_problem_contribution_covariance": joint - marginal ** 2,
            "actual_variance_of_mean": actual_variance,
            "variance_if_incorrectly_independent": actual_variance / 2,
            "fixed_contribution_percentile_bootstrap_coverage": percentile_coverage,
            "scope": "Exact small-sample counterexample; no general nominal coverage follows from cross-fitting alone.",
        },
        "noisy_oracle_selection": {
            "equal_arm_true_success": 0.5,
            "expected_maximum_of_two_single_rollout_outcomes": empirical_best,
            "fresh_rollout_value_of_selected_arm": 0.5,
        },
    }


def mdp_trial(rng, horizon=4, states=3, actions=3, epsilon=0.025):
    """Finite budget-indexed MDP with action costs included in rewards."""
    terminal = [rng.random() for _ in range(states)]
    transitions, rewards = [], []
    for _ in range(horizon):
        layer, layer_rewards = [], []
        for _ in range(states):
            action_transitions, action_rewards = [], []
            for a in range(actions):
                raw = [rng.random() + 0.05 for _ in range(states)]
                norm = sum(raw)
                action_transitions.append([p / norm for p in raw])
                action_rewards.append(-0.015 * a)
            layer.append(action_transitions)
            layer_rewards.append(action_rewards)
        transitions.append(layer)
        rewards.append(layer_rewards)
    baseline = [[0.0] * states for _ in range(horizon + 1)]
    baseline[-1] = terminal[:]
    q_mu = [None] * horizon
    for t in reversed(range(horizon)):
        q_mu[t] = [[rewards[t][s][a] + weighted(baseline[t + 1], transitions[t][s][a])
                    for a in range(actions)] for s in range(states)]
        baseline[t] = [q_mu[t][s][0] for s in range(states)]
    policy = []
    max_error = 0.0
    min_advantage = 0.0
    switches = 0
    for t in range(horizon):
        layer = []
        for s in range(states):
            fitted = [q + rng.uniform(-epsilon, epsilon) for q in q_mu[t][s]]
            max_error = max(max_error, max(abs(x - q) for x, q in zip(fitted, q_mu[t][s])))
            best = max(range(actions), key=fitted.__getitem__)
            selected = best if fitted[best] - fitted[0] > 2 * epsilon else 0
            layer.append(selected)
            switches += int(selected != 0)
            min_advantage = min(min_advantage, q_mu[t][s][selected] - q_mu[t][s][0])
        policy.append(layer)
    deployed = terminal[:]
    for t in reversed(range(horizon)):
        deployed = [rewards[t][s][policy[t][s]]
                    + weighted(deployed, transitions[t][s][policy[t][s]])
                    for s in range(states)]
    # Independently compute the performance-difference RHS using occupancies.
    occupancy = [1 / states] * states
    rhs = 0.0
    for t in range(horizon):
        rhs += sum(occupancy[s] * (q_mu[t][s][policy[t][s]] - baseline[t][s])
                   for s in range(states))
        occupancy = [sum(occupancy[s] * transitions[t][s][policy[t][s]][j]
                         for s in range(states)) for j in range(states)]
    difference = sum(deployed[s] - baseline[0][s] for s in range(states)) / states
    return {"value_improvement": difference, "performance_difference_rhs": rhs,
            "identity_error": abs(difference - rhs), "min_true_advantage": min_advantage,
            "max_prediction_error": max_error, "switches": switches, "epsilon": epsilon}


def greedy_regret_trials(rng, count=500):
    max_excess, observed_regret = 0.0, 0.0
    for _ in range(count):
        q = [rng.random() for _ in range(3)]
        epsilon = rng.uniform(0.001, 0.2)
        fitted = [x + rng.uniform(-epsilon, epsilon) for x in q]
        selected = max(range(3), key=fitted.__getitem__)
        regret = max(q) - q[selected]
        max_excess = max(max_excess, regret - 2 * epsilon)
        observed_regret = max(observed_regret, regret)
    return {"trials": count, "largest_excess_over_bound": max_excess,
            "largest_observed_regret": observed_regret}


def paired_problem_bootstrap(differences, seed=2718, repeats=2000):
    """Resample original-problem aggregates, never their replay/seed rows."""
    if not differences:
        raise ValueError("At least one independent problem is required")
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(repeats))
    return {"independent_problems": n, "mean": sum(differences) / n,
            "percentile_interval": [means[int(0.025 * (repeats - 1))],
                                    means[int(0.975 * (repeats - 1))]],
            "repeats": repeats, "seed": seed}


def validate(seed=1729, mdp_trials=200):
    rng = random.Random(seed)
    mdps = [mdp_trial(rng) for _ in range(mdp_trials)]
    sensor = sensor_example()
    dr = dr_exact_check()
    regret = greedy_regret_trials(rng)
    inference = inference_counterexamples()
    # This counterexample detects the common error of ignoring acquisition.
    query_counterexample = {"baseline_value_without_query": 0.5,
                            "post_query_best_action_value": 0.54,
                            "acquisition_cost": 0.1,
                            "query_policy_net_value": 0.44}
    bootstrap = paired_problem_bootstrap([-0.5, 0, 0.5, 1.0])
    checks = {
        "dr_with_misspecified_outcome_model": dr["absolute_error"] < 1e-12,
        "sequential_nonnegative_advantage": all(x["min_true_advantage"] >= -1e-12 for x in mdps),
        "sequential_nonnegative_value_improvement": all(x["value_improvement"] >= -1e-12 for x in mdps),
        "performance_difference_identity": all(x["identity_error"] < 1e-12 for x in mdps),
        "uniform_error_condition_actually_holds": all(x["max_prediction_error"] <= x["epsilon"] + 1e-12 for x in mdps),
        "greedy_regret_bound": regret["largest_excess_over_bound"] < 1e-12,
        "free_information_monotonicity": sensor["always_sensor_value_before_cost"] >= sensor["cheap_policy_value"],
        "selective_gate_outperforms_both_fixed_choices_in_constructed_example": sensor["selective_sensor_value_after_cost"] > max(sensor["cheap_policy_value"], sensor["always_sensor_value_after_cost"]),
        "cost_can_reverse_apparent_improvement": query_counterexample["query_policy_net_value"] < query_counterexample["baseline_value_without_query"] < query_counterexample["post_query_best_action_value"],
        "bootstrap_uses_problem_count": bootstrap["independent_problems"] == 4,
        "adaptive_stopping_can_bias_arm_mean": inference["adaptive_replay_stopping"]["expected_stopped_sample_mean"] == 0.625,
        "crossfitting_does_not_imply_independent_contributions": inference["crossfit_dependence"]["cross_problem_contribution_covariance"] == 0.1875,
        "fixed_crossfit_contribution_bootstrap_can_fail": inference["crossfit_dependence"]["fixed_contribution_percentile_bootstrap_coverage"] == 0.0,
        "same_sample_oracle_is_optimistic": inference["noisy_oracle_selection"]["expected_maximum_of_two_single_rollout_outcomes"] == 0.75,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"status": "synthetic_validation_only", "seed": seed,
            "warning": "No LLM/Jev outcome data; exact constructed values are not empirical predictions. Checks do not establish deployment error bounds or OOD assumptions.",
            "checks": checks, "all_checks_passed": all(checks.values()),
            "finite_mdp": {"trials": mdp_trials, "horizon": 4, "states": 3, "actions": 3,
                           "smallest_value_improvement": min(x["value_improvement"] for x in mdps),
                           "largest_value_improvement": max(x["value_improvement"] for x in mdps),
                           "largest_identity_error": max(x["identity_error"] for x in mdps),
                           "total_switched_state_actions": sum(x["switches"] for x in mdps)},
            "sensor_example_exact": sensor, "dr_exact": dr, "greedy_regret": regret,
            "inference_counterexamples_exact": inference,
            "acquisition_counterexample": query_counterexample,
            "bootstrap_diagnostic": bootstrap,
            "hoeffding_radius_n3000_k1_alpha05": math.sqrt(2 * math.log(20) / 3000),
            "replays_for_uniform_01_error_M48_A3_delta05": math.ceil(math.log(2 * 48 * 3 / 0.05) / (2 * 0.1 ** 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/theory_validation.json"))
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    result = validate(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checks_passed": len(result["checks"]),
                      "finite_mdp_trials": result["finite_mdp"]["trials"],
                      "status": result["status"]}))


if __name__ == "__main__":
    main()
