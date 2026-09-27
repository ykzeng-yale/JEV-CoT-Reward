import numpy as np
import pytest
from jev_control.controller import fit_controller


def fixture():
    ids = ['p0', 'p1', 'p2', 'p3']
    docs = ['first state', 'second state', 'first sample', 'second sample']
    numeric = [[0, 1], [1, 2], [0, 3], [1, np.nan]]
    rows = [{'problem_id': pid, 'action': action, 'outcome': {'success': bool((i % 2 == 0) == (action == 'continue'))}}
            for i, pid in enumerate(ids) for action in ('continue', 'repair', 'branch') for _ in range(i + 1)]
    return ids, docs, numeric, rows


def test_frozen_controller_matches_original_screen_predictor():
    from test_screen_audit import analysis
    ids, docs, numeric, rows = fixture()
    test_docs, test_num = ['first unseen', 'second unseen'], [[0, 2], [1, np.nan]]
    original, _ = analysis.fit_predict(docs, test_docs, numeric, test_num, rows, ids, ['t0', 't1'], 10.)
    controller = fit_controller(docs, numeric, rows, ids)
    np.testing.assert_allclose(controller.predict(test_docs, test_num), original, rtol=1e-12, atol=1e-12)
    assert controller.training_problem_ids == tuple(ids)
    assert controller.choose(test_docs, test_num) == [analysis.ACTIONS[i] for i in original.argmax(axis=1)]


def test_controller_rejects_missing_actions_and_unknown_problem():
    ids, docs, numeric, rows = fixture()
    with pytest.raises(ValueError, match='every action'):
        fit_controller(docs, numeric, [r for r in rows if r['action'] != 'branch'], ids)
    rows[0]['problem_id'] = 'heldout'
    with pytest.raises(ValueError, match='Unknown'):
        fit_controller(docs, numeric, rows, ids)
