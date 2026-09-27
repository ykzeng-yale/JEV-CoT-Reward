import copy
import json
import pytest
from jev_control.prospective import execute_selected, POLICIES
from jev_control.prospective_audit import assignment_audit


def fixture(tmp_path):
    initial={'generated_tokens':5,'prompt_tokens':8,'elapsed_seconds':1.}
    decisions=dict(zip(POLICIES,['continue','repair','repair','continue']))
    costs={p:{'usd':0} for p in POLICIES}
    def action(a,s):return {'action':a,'seed':s,'generated_tokens':10,'prompt_tokens_processed':20,'elapsed_seconds':2.}
    rows=execute_selected('p',decisions,12,initial,tmp_path/'d',tmp_path/'o',action,
                          checkpoint_sha256='h',acquisition_costs=costs)
    data=(tmp_path/'d').read_bytes()
    timestamp=json.loads(data)['recorded_unix']
    return [[{'task':{'id':'p'},'continuation_seed':12}],
            [{'problem_id':'p','initial':initial,'sha256':'h'}],data,rows,
            [{'problem_id':'p','phase':'continuation','action':'continue','started_unix':timestamp+1}]]


def test_assignment_audit_detects_ordering_mapping_and_cost_corruption(tmp_path):
    args=fixture(tmp_path)
    assert assignment_audit(*args)['policy_episodes']==4
    corrupted=copy.deepcopy(args);corrupted[4][0]['started_unix']=0
    with pytest.raises(ValueError,match='before'):assignment_audit(*corrupted)
    corrupted=copy.deepcopy(args);corrupted[3][0]['policies'].pop()
    with pytest.raises(ValueError,match='mapping'):assignment_audit(*corrupted)
    corrupted=copy.deepcopy(args);next(iter(corrupted[3][0]['hypothetical_deployment_costs'].values()))['generated_tokens']=7.5
    with pytest.raises(ValueError,match='cost mismatch'):assignment_audit(*corrupted)
    corrupted=copy.deepcopy(args);corrupted[3].pop()
    with pytest.raises(ValueError,match='Missing selected'):assignment_audit(*corrupted)
