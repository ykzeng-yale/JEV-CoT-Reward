"""Independent token-chain/control audit of a recorded segmented episode."""
import hashlib
import json
import math


def audit_episode(result,prompt,retained,initial_tokens,encode,*,budget=2048,reserve=128,
                  segment_tokens=64,branch_tokens=100,seed=1,branching=True,random_rate=None):
    calls=result['calls'];decisions=result['decisions'];cursor=0;di=0;spent=initial_tokens;history=list(retained);entropies=[]
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    def take(context,suffix,cap,temp):
        nonlocal cursor,spent
        require(cursor<len(calls),'Missing call')
        row=calls[cursor];prefix=list(prompt)+history+suffix;g=row['generation']
        require(row['context']==context and row['prefix_ids']==prefix,'Call context/prefix mismatch')
        require(row['requested_tokens']==cap and row['temperature']==temp and row['seed']==seed+cursor,'Sampling mismatch')
        require(g['seed']==seed+cursor and g['prompt_tokens']==len(prefix),'Generation identity mismatch')
        require(g['prefix_sha256']==hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),'Prefix digest mismatch')
        n=g['generated_tokens'];require(type(n) is int and n==len(g['token_ids']) and 0<=n<=cap,'Generation length mismatch')
        require(len(g['token_entropies'])==n and all(math.isfinite(x) and x>=0 for x in g['token_entropies']),'Entropy record mismatch')
        spent+=n;require(row['spent_generated_tokens']==spent and spent<=budget,'Budget accounting mismatch')
        cursor+=1;return g
    terminal=False
    while spent<budget-reserve and not terminal:
        g=take('segment',[],min(segment_tokens,budget-reserve-spent),.7);history+=g['token_ids']
        terminal=g['finish_reason'] in ('stop','timeout')
        if terminal or not g['token_ids']:break
        entropies.append(g['token_entropies'][-1]);remaining=budget-spent
        capacity=remaining>=3*branch_tokens+reserve+1
        trigger=branching and len(entropies)>=5 and remaining>200 and entropies[-1]>sorted(entropies)[int(len(entropies)*.9)] and capacity
        if random_rate is not None:
            u=int.from_bytes(hashlib.sha256(f'{seed}:{len(entropies)}'.encode()).digest()[:8],'big')/2**64
            trigger=branching and len(entropies)>=11 and capacity and u<random_rate
        require(di<len(decisions),'Missing decision');d=decisions[di];di+=1
        expected={'boundary':len(entropies),'trigger':trigger,'pool_capacity':capacity,
                  'spent_before_pool':spent,'remaining_before_pool':remaining,'selected':None}
        if trigger:
            pool=[]
            for i,(suffix,temp) in enumerate(zip(('', 'Wait', 'Let me reconsider: '),(0.,.6,1.5))):
                inserted=encode(suffix) if suffix else [];c=take('candidate_'+str(i),inserted,branch_tokens,temp)
                mean=sum(c['token_entropies'])/len(c['token_entropies']) if c['token_entropies'] else None
                require(c['mean_entropy']==mean or (mean is not None and c['mean_entropy'] is not None and math.isclose(c['mean_entropy'],mean,rel_tol=1e-6)),'Mean entropy mismatch')
                # Runtime selects using its stored reduction; validate it above, then
                # preserve that value to avoid changing near ties through summation order.
                pool.append((c,inserted,c['mean_entropy']))
            winner=min(range(3),key=lambda i:pool[i][2] if pool[i][2] is not None else math.inf)
            chosen,inserted,_=pool[winner];history+=inserted+chosen['token_ids']
            terminal=chosen['finish_reason'] in ('stop','timeout')
            expected.update(selected=winner,scores=[-x[2] if x[2] is not None else None for x in pool],spent_after_pool=spent)
        require(d==expected,'Decision mismatch')
    if not terminal and spent<budget:
        suffix=encode('\nGive the final answer now using the required FINAL: format.\n')
        g=take('final',suffix,budget-spent,.7);history+=suffix+g['token_ids']
    require(cursor==len(calls) and di==len(decisions),'Extra calls/decisions')
    require(result['retained_ids']==history and result['generated_tokens']==spent,'Terminal history/accounting mismatch')
    return {'calls':cursor,'decisions':di,'generated_tokens':spent}
