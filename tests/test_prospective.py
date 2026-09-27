import json
import pytest
from jev_control.prospective import execute_selected, POLICIES


def test_all_decisions_durable_before_first_outcome_and_shared_cost_not_divided(tmp_path):
    decisions = dict(zip(POLICIES, ['continue', 'repair', 'repair', 'continue']))
    costs = {p: {'usd': .01 if p.endswith('jev') else 0} for p in POLICIES}
    starts = []
    decision_path, outcome_path = tmp_path/'decisions.jsonl', tmp_path/'outcomes.jsonl'
    def run(action, seed):
        logged = json.loads(decision_path.read_text())
        assert logged['decisions'] == decisions
        assert logged['seed'] == 123
        starts.append(action)
        return {'action':action, 'seed':seed, 'generated_tokens':10,
                'prompt_tokens_processed':20, 'elapsed_seconds':2., 'outcome':{'success':True}}
    initial = {'generated_tokens':5, 'prompt_tokens':8, 'elapsed_seconds':1.}
    outputs = execute_selected('p1', decisions, 123, initial, decision_path, outcome_path, run,
                               checkpoint_sha256='abc', acquisition_costs=costs)
    assert starts == ['continue', 'repair']
    assert len(outputs) == 2
    for output in outputs:
        assert len(output['policies']) == 2
        for measured in output['hypothetical_deployment_costs'].values():
            assert measured['generated_tokens'] == 15
            assert measured['model_service_seconds'] == 3
    with pytest.raises(ValueError, match='already recorded'):
        execute_selected('p1', decisions, 123, initial, decision_path, outcome_path, run,
                         checkpoint_sha256='abc', acquisition_costs=costs)
    assert starts == ['continue', 'repair']


def test_incomplete_decisions_refused_before_backend_or_files(tmp_path):
    def forbidden(*args):
        raise AssertionError('Must not generate')
    with pytest.raises(ValueError, match='Complete'):
        execute_selected('p', {'always_continue':'continue'}, 1, {}, tmp_path/'d', tmp_path/'o', forbidden,
                         checkpoint_sha256='a', acquisition_costs={})
    assert not list(tmp_path.iterdir())
