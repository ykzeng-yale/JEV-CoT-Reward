"""Synthetic baseline contracts; no live data, models or hosted calls."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from jev_control.baselines import fit_action_forest, fit_sparse_pair_gate


def test_forest_learns_opposite_actions_and_handles_one_class_branch():
    x, actions, labels, groups = [], [], [], []
    for i in range(32):
        state = -1 if i % 2 else 1
        for action in range(3):
            for repeat in range(2):
                x.append([state]); actions.append(action); groups.append(f"p{i}")
                labels.append(int(action == (0 if state < 0 else 1)))
    model = fit_action_forest(np.asarray(x), actions, labels, groups)
    predicted = model.predict(np.asarray([[-1], [1]]))
    assert predicted["selection_score"].argmax(axis=1).tolist() == [0, 1]
    np.testing.assert_array_equal(predicted["success_probability"][:, 2], [0, 0])
    np.testing.assert_array_equal(predicted["tree_dispersion"][:, 2], [0, 0])


def test_sparse_gate_learns_direction_without_turning_ties_into_negative_labels():
    x = np.asarray([[-1], [1], [-1], [1]] * 40)
    base, alt = [1, 0, 1, 1] * 40, [0, 1, 1, 1] * 40
    model = fit_sparse_pair_gate(x, base, alt, [f"p{i}" for i in range(len(x))])
    p = model.predict(np.asarray([[-1], [1]]))
    assert p[0] < .5 < p[1]
    assert model.discordant_pairs == 80 and model.total_pairs == 160
    tied = fit_sparse_pair_gate(x, base, base, [f"p{i}" for i in range(len(x))])
    np.testing.assert_array_equal(tied.predict(x) > .5, np.zeros(len(x), dtype=bool))


def test_binary_gain_identity_for_all_feasible_couplings():
    for p_alt, p_base in ((.8, .7), (.4, .6), (.5, .5)):
        for joint_success in np.linspace(max(0, p_alt + p_base - 1), min(p_alt, p_base), 11):
            win, loss = p_alt - joint_success, p_base - joint_success
            assert win - loss == pytest.approx(p_alt - p_base)
            if win + loss:
                assert (win + loss) * (2 * win / (win + loss) - 1) == pytest.approx(p_alt - p_base)
    # Independent .8/.7 arms: a positive expected gain despite <.5 outright wins.
    win, loss = .8 * .3, .2 * .7
    assert win < .5 and win / (win + loss) > .5


def test_invalid_baseline_training_data_is_rejected():
    with pytest.raises(ValueError, match="three action"):
        fit_action_forest(np.zeros((2, 1)), [0, 1], [0, 1], ["a", "b"])
    with pytest.raises(ValueError, match="binary"):
        fit_action_forest(np.zeros((3, 1)), [0, 1, 2], [0, .5, 1], ["a"] * 3)
    with pytest.raises(ValueError, match="Binary"):
        fit_sparse_pair_gate(np.zeros((2, 1)), [0, 1], [0, .4], ["a", "b"])


def test_weighted_logistic_excess_bound_on_finite_mixtures():
    rng = np.random.default_rng(73)
    for _ in range(100):
        eta, q, discordance = rng.uniform(.001, .999, (3, 15))
        mass = rng.dirichlet(np.ones(15))
        regret = np.sum(mass * discordance * np.abs(2 * eta - 1) * ((eta > .5) != (q > .5)))
        kl = eta * np.log(eta / q) + (1 - eta) * np.log((1 - eta) / (1 - q))
        bound = np.sqrt(2 * np.sum(mass * discordance) * np.sum(mass * discordance * kl))
        assert regret <= bound + 1e-12


def test_complete_synthetic_comparison_uses_same_grouped_folds(tmp_path):
    from test_screen_audit import make_run, analysis
    script_dir = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("additional_baselines", script_dir / "analyze_baselines.py")
        extra = importlib.util.module_from_spec(spec); spec.loader.exec_module(extra)
    finally:
        sys.path.remove(str(script_dir))
    cps, rows, manifest = make_run(tmp_path / "synthetic")
    report = analysis.analyze(tmp_path / "synthetic", learned=True, draws=100)
    result = extra.compare(report, {cp["problem_id"]: cp for cp in cps}, rows, {}, manifest["budget"], draws=100)
    assert result["n_problems"] == 24
    assert result["policies"]["forest_mean:cheap_tfidf"]["success"]["mean"] == 1
    assert result["policies"]["sparse_repair_gate:cheap_tfidf"]["success"]["mean"] == 1
    assert set(result["family_policy_means"]) == {"left", "right"}
    assert all(sum(policy["action_counts"].values()) == 24 for policy in result["policies"].values())
    assert len(result["fold_diagnostics"]) == len(report["crossfit"]["folds"])
    report["crossfit"]["status"] = "not_run"
    with pytest.raises(ValueError, match="complete audited"):
        extra.compare(report, {}, [], {}, 100)


def test_semantic_heuristic_keeps_query_costs_and_rate_matched_controls(tmp_path):
    from test_screen_audit import make_run, analysis, lines
    script_dir = Path(__file__).parents[1] / 'scripts'
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location('semantic_baselines', script_dir / 'analyze_baselines.py')
        extra = importlib.util.module_from_spec(spec); spec.loader.exec_module(extra)
    finally:
        sys.path.remove(str(script_dir))
    run = tmp_path / 'synthetic_semantics'
    cps, rows, manifest = make_run(run)
    for cp in cps:
        validity = .9 if cp['task']['family'] == 'left' else .2
        cp['jev'] = {'response': {'answers': {key: {'noul': validity if key == 'locally_valid' else .5}
                                           for key in analysis.SEMANTIC_KEYS}},
                     'input_cost_usd': .003, 'elapsed_seconds': .7}
    lines(run / 'checkpoints.jsonl', cps)
    report = analysis.analyze(run, learned=True, draws=20)
    result = extra.compare(report, {cp['problem_id']: cp for cp in cps}, rows, {}, manifest['budget'], draws=20)
    name = 'validity_07_gate:cheap_tfidf_plus_jev'
    policy = result['policies'][name]
    assert policy['success']['mean'] == 1
    assert policy['action_counts'] == {'continue': 12, 'repair': 12}
    assert policy['costs']['jev_uncached_equivalent_usd_per_problem'] == pytest.approx(.003)
    assert policy['costs']['jev_recorded_service_seconds_per_problem'] == pytest.approx(.7)
    # Family alone explains this fixture's optimal action. A correct within-family
    # randomization therefore grants no spurious targeting advantage.
    control = result['within_fold_family_rate_matched_diagnostic'][name]
    assert control['expected_success']['mean'] == 1
    assert control['selected_minus_randomized']['mean'] == 0
    assert result['policies']['entropy_median_gate:cheap_tfidf']['action_counts'] == {'continue': 24}
