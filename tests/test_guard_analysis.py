import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from analyze_guard_qualification import analyze


def fixture(tmp_path):
    def write(name,obj):
        (tmp_path/name).write_text(json.dumps(obj)+'\n')
    write('manifest.json',{'config':{'problems':1,'policies':['continue','segmented_sham']}})
    write('checkpoints.jsonl',{'problem_id':'p','task':{'family':'f'}})
    rows=[{'problem_id':'p','policy':p,'outcome':{'success':True},'episode_generated_tokens':7} for p in ['continue','segmented_sham']]
    (tmp_path/'outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    g={'prompt_tokens':3,'generated_tokens':2,'elapsed_seconds':1.,'finish_reason':'length'}
    ee=[{'problem_id':'p','phase':'initial','generation':g}]+[{'problem_id':'p','phase':'qualification','policy':p,'generation':{**g,'generated_tokens':5}} for p in ['continue','segmented_sham']]
    (tmp_path/'generation_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in ee))
    for f in ['decisions.jsonl','skipped.jsonl']:(tmp_path/f).write_text('')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    write('audit.json',{'status':'passed_guard_runtime_audit','ready_for_analysis':True,'input_sha256':hashes})
    return tmp_path/'audit.json'


def test_shared_cost_is_not_double_counted_in_collection(tmp_path):
    a=fixture(tmp_path);r=analyze(tmp_path,a)
    assert r['collection_generated_tokens']==12
    assert r['policies']['continue']['episode_generated_tokens']==7
    assert r['policies']['continue']['service_seconds']==2
    assert r['paired_contrasts']['segmented_sham_minus_continue']['discordant_problems']==0


def test_stale_outcomes_rejected(tmp_path):
    a=fixture(tmp_path)
    with (tmp_path/'outcomes.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Stale audit'):analyze(tmp_path,a)


def test_timeout_sensitivity_allows_unknown_completion(tmp_path):
    a=fixture(tmp_path)
    path=tmp_path/'generation_events.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    rows[1]['generation']['finish_reason']='timeout'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    audit=json.loads(a.read_text());audit['input_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest();a.write_text(json.dumps(audit))
    r=analyze(tmp_path,a)
    assert r['policies']['continue']['episodes_with_timeout']==1
    assert r['paired_contrasts']['segmented_sham_minus_continue']['timeout_completion_sensitivity']==[0.,1.]


def test_report_identifies_executed_analysis_dependencies(tmp_path):
    a=fixture(tmp_path);r=analyze(tmp_path,a)
    assert set(r['analysis_dependency_sha256'])=={'analyze_guard_qualification.py','analyze_screen.py'}
    for name,digest in r['analysis_dependency_sha256'].items():
        assert digest==hashlib.sha256((Path(__file__).parents[1]/'scripts'/name).read_bytes()).hexdigest()


def test_zero_discordance_is_not_certified_equivalence(tmp_path):
    a=fixture(tmp_path);r=analyze(tmp_path,a)
    c=r['paired_contrasts']['segmented_sham_minus_continue']
    assert c['bootstrap_percentile_95']==[0.,0.]
    assert c['simultaneous_hoeffding_95']==[-1.,1.]
