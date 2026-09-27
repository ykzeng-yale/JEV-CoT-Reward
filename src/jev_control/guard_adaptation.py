"""Budgeted segment-boundary GUARD adaptation, pending local runtime qualification.

Unlike the original pre-sampling tokenizer-boundary trigger, this adapter uses
last emitted token entropy at fixed 64-token boundaries. It preserves the
current-inclusive trigger and heterogeneous branch temperatures, but does not
force EOS or reward a partial boxed marker. Never label as original GUARD.
"""
from .guard_contract import entropy_trigger,BRANCH_TEMPERATURES


def run(task_prefix,retained,initial_tokens,generate,encode,record,*,budget=2048,
        reserve=128,segment_tokens=64,branch_tokens=100,seed=1):
    """generate(prefix, cap, seed, temperature) -> Generation.

    record(kind, row) must durably save calls/decisions. No task answer or verifier
    is accessible here. All candidate generation, including losers, consumes cap.
    initial_tokens is all already spent generation, not merely retained length.
    """
    if min(budget,reserve,segment_tokens,branch_tokens)<=0 or initial_tokens<0 or initial_tokens>budget-reserve:
        raise ValueError('Invalid allowance')
    prefix=list(task_prefix);history=list(retained);spent=initial_tokens;calls=[];entropies=[];decisions=[]
    terminal=False
    def call(context,suffix,cap,temperature):
        nonlocal spent
        cap=min(cap,budget-spent)
        if cap<=0:raise ValueError('No generation allowance')
        result=generate(prefix+history+suffix,cap,seed+len(calls),temperature)
        if result.generated_tokens!=len(result.token_ids) or not 0<=result.generated_tokens<=cap:
            raise ValueError('Backend violated token allowance')
        spent+=result.generated_tokens
        row={'context':context,'generation':result.to_dict(),'spent_generated_tokens':spent}
        record('calls',row);calls.append(row)
        return result
    while spent<budget-reserve and not terminal:
        g=call('segment',[],min(segment_tokens,budget-reserve-spent),.7)
        history+=g.token_ids
        terminal=g.finish_reason in ('stop','timeout') or 'FINAL:' in g.text
        if terminal or not g.token_ids:break
        if not g.token_entropies:raise ValueError('Predictive entropy statistics required')
        entropies.append(g.token_entropies[-1])
        remaining=budget-spent
        # Require full pool and at least one continuation token beyond reserve.
        capacity=remaining>=3*branch_tokens+reserve+1
        trigger=entropy_trigger(entropies,remaining) and capacity
        decision={'boundary':len(entropies),'trigger':trigger,'pool_capacity':capacity,
                  'spent_before_pool':spent,'remaining_before_pool':remaining,'selected':None}
        if trigger:
            candidates=[]
            for i,(suffix,temp) in enumerate(zip(('', 'Wait', 'Let me reconsider: '),BRANCH_TEMPERATURES)):
                inserted=encode(suffix) if suffix else []
                c=call('candidate_'+str(i),inserted,branch_tokens,temp)
                # A completed marker earns no unverified correctness bonus.
                score=-(c.mean_entropy if c.mean_entropy is not None else float('inf'))
                candidates.append((c,inserted,score))
            winner=max(range(3),key=lambda i:candidates[i][2])
            selected,inserted,_=candidates[winner]
            decision.update(selected=winner,scores=[c[2] if c[2]!=-float('inf') else None for c in candidates],
                            spent_after_pool=spent)
            record('decisions',decision);decisions.append(decision)
            history+=inserted+selected.token_ids
            terminal=selected.finish_reason in ('stop','timeout') or 'FINAL:' in selected.text
        else:
            record('decisions',decision);decisions.append(decision)
    if not terminal and spent<budget:
        suffix=encode('\nGive the final answer now using the required FINAL: format.\n')
        g=call('final',suffix,budget-spent,.7)
        history+=suffix+g.token_ids
    return {'retained_ids':history,'generated_tokens':spent,'calls':calls,'decisions':decisions,
            'scope':'Segment-boundary GUARD adaptation; total generator work charged; not original GUARD.'}
