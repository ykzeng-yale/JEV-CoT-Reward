import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_cuda_qualification import audit


def fixture(path):
    (path/'outputs').mkdir()
    records={'submission.json':{'job_id':'123','model':'q','revision':'rev','source_sha256':{'qualify_cuda.py':'hash'}},
    'outputs/manifest.json':{'job_id':'123','model':'q','revision':'rev','script_sha256':'hash','dtype':'bfloat16','quantization':None,
        'versions':{'torch':'2.9.1','transformers':'4.55.2'},'cuda':'12.8','gpu':'NVIDIA B200','snapshot_bytes':8000000000},
    'outputs/summary.json':{'gpu':'NVIDIA B200','status':'passed','batch_tokens_per_second':4.,'peak_allocated_gib':10.},
    'outputs/resume_check.json':{'greedy_exact_prefix_resume':True,'whole_ids':[1]*64,'resumed_ids':[1]*64,'generated_tokens':64},
    'outputs/long_context.json':{'input_tokens':8192,'output_tokens':64,'seconds':2.},
    'outputs/batch.json':{'outputs':[{'token_ids':[1,2],'text':'ok'}]*4,'elapsed_seconds':2.,'emitted_tokens':8}}
    for name,value in records.items():(path/name).write_text(json.dumps(value))
    accounting=path/'accounting.txt';accounting.write_text('123|COMPLETED|0:0|00:01:00|gpu=1\n')
    return records,accounting

def test_valid_records(tmp_path):
    _,accounting=fixture(tmp_path)
    assert audit(tmp_path,accounting)['batch_tokens_per_second']==4.

@pytest.mark.parametrize('file,key,value',[
    ('outputs/manifest.json','revision','wrong'),('outputs/summary.json','batch_tokens_per_second',400.),
    ('outputs/resume_check.json','resumed_ids',[2]*64),('outputs/long_context.json','input_tokens',100),
    ('outputs/batch.json','emitted_tokens',800),('outputs/summary.json','peak_allocated_gib',float('nan'))])
def test_inconsistent_evidence_fails(tmp_path,file,key,value):
    records,accounting=fixture(tmp_path);records[file][key]=value;(tmp_path/file).write_text(json.dumps(records[file]))
    with pytest.raises(ValueError):audit(tmp_path,accounting)

def test_failed_job_cannot_pass(tmp_path):
    _,accounting=fixture(tmp_path);accounting.write_text('123|FAILED|132:0|00:00:23|gpu=1\n')
    with pytest.raises(ValueError):audit(tmp_path,accounting)
