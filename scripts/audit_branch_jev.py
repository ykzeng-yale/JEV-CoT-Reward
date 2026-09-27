"""Independent hosted-selector record checks; never calls the service."""
import hashlib
import json
import math

MODEL='jev-1.13.0'
PRICE_PER_MILLION=.042
RESERVATION=.01


def require(ok,message):
    if not ok: raise ValueError(message)


def number(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def equal(left,right,label):
    require(number(left) and number(right) and math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-12),label)


def audit_record(row,expected_request,local_time,decision_time):
    require(row['request']==expected_request,'Hosted information mismatch')
    require(all(number(t) for t in (local_time,row['recorded_unix'],decision_time)) and
            local_time<=row['recorded_unix']<=decision_time,'Hosted chronology mismatch')
    require(number(row['acquisition_seconds']),'Invalid acquisition time')
    if row['error'] is not None:
        require(isinstance(row['error'],str) and row['error'] and row['choice'] is None,'Invalid failed choice')
        require(row['input_tokens'] is None and 'result' not in row,'Failure contains successful result')
        equal(row['accounted_usd'],RESERVATION,'Failure reservation mismatch')
        return None
    result=row['result'];response=result['response']
    require(response['model']==MODEL and set(response['answers'])=={'candidate'},'Hosted response mismatch')
    answer=response['answers']['candidate'];probs=answer['probabilities']
    keys=set(expected_request['questions']['candidate']['criteria'])
    require(answer['type']=='choice' and isinstance(probs,dict) and set(probs)==keys,'Hosted option mismatch')
    require(all(number(v) and v<=1 for v in probs.values()) and abs(sum(probs.values())-1.)<=1e-5,'Invalid distribution')
    choice=answer['choice']
    require(isinstance(choice,str) and choice in keys and probs[choice]>=max(probs.values())-1e-8,'Invalid selection')
    require(type(row['choice']) is int and row['choice']==int(choice),'Choice parsing mismatch')
    require(number(answer['confidence']) and answer['confidence']<=1,'Invalid confidence')
    usage=response['usage']['input_tokens']
    require(type(usage) is int and usage>=0 and type(row['input_tokens']) is int and row['input_tokens']==usage,'Invalid usage')
    equal(row['accounted_usd'],usage*PRICE_PER_MILLION/1e6,'Hosted accounting mismatch')
    equal(result['input_cost_usd'],row['accounted_usd'],'Cached accounting mismatch')
    require(type(result['cache_hit']) is bool and number(result['elapsed_seconds']),'Invalid cache metadata')
    payload={'model':MODEL,**expected_request}
    canonical=json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    require(result['request_sha256']==hashlib.sha256(canonical.encode()).hexdigest(),'Hosted request hash mismatch')
    return row['choice']
