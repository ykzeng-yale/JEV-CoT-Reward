"""Common candidate pool with decisions recorded before shadow continuations.

Generator-token allowances are matched; local selector inference is separately
charged and is not a matched-total-compute experiment.
"""
from .branch_selectors import CandidateView, choose, local_prompt, parse_local_choice
from .tasks import verify


def local_choice(backend, task_prompt, retained_text, views, seed, max_tokens):
    from mlx_lm.sample_utils import make_sampler
    raw=backend.backend;old=raw._sampler
    prompt=backend.encode_chat([{'role':'user','content':local_prompt(task_prompt,retained_text,views)}])
    try:
        raw._sampler=make_sampler(temp=0,top_p=.9)
        generation=backend.generate(prompt,max_tokens,seed,timeout_s=120)
    finally:raw._sampler=old
    error=None;choice=None
    try:
        if generation.finish_reason=='timeout':raise ValueError('Judge timeout')
        choice=parse_local_choice(generation.text,len(views))
    except ValueError:error='invalid_local_choice'
    return {'choice':choice,'error':error,'generation':generation.to_dict(),'prompt_ids':prompt}


def collect_pool(backend,task,cp,config,index,record,rollout,judge=local_choice):
    """record(kind, row) must durably persist each row before returning."""
    pid=task['id'];prompt=cp['prompt_ids'];retained=cp['retained_ids'];initial=cp['initial']
    budget=config['budget'];reserve=config['final_reserve'];base=config['task_seed']+index*10000
    if initial['generated_tokens']+config['candidate_count']*config['candidate_tokens']+reserve>=budget:
        raise ValueError('No continuation capacity after candidate generation')
    candidates=[]
    for j in range(config['candidate_count']):
        backend.context={'phase':'candidate','problem_id':pid,'candidate':j}
        g=backend.generate(prompt+retained,config['candidate_tokens'],base+100+j,timeout_s=180)
        candidates.append(g)
        record('candidates',{'problem_id':pid,'candidate':j,'generation':g.to_dict()})
    views=[CandidateView(g.text,g.mean_logprob,g.mean_entropy) for g in candidates]
    choices={};failures={}
    for method in ('uniform','likelihood','entropy_reduction'):
        try:choices[method]=choose(views,method,base+500)
        except ValueError:
            choices[method]=0;failures[method]='no_usable_score_fallback_first'
    backend.context={'phase':'local_selector','problem_id':pid,'temperature':0}
    local=judge(backend,task['prompt'],backend.decode(retained),views,base+600,config['judge_tokens'])
    record('local_judge',{'problem_id':pid,**local})
    choices['local_semantic']=local['choice'] if local['choice'] is not None else choices['likelihood']
    if local['choice'] is None:failures['local_semantic']='invalid_choice_fallback_likelihood'
    pool_tokens=sum(g.generated_tokens for g in candidates)
    pool_prompts=sum(g.prompt_tokens for g in candidates)
    pool_seconds=sum(g.elapsed_seconds for g in candidates)
    decision={'problem_id':pid,'choices':choices,'failures':failures,'pool_generated_tokens':pool_tokens,
              'pool_prompt_tokens':pool_prompts,'pool_service_seconds':pool_seconds}
    record('decisions',decision)  # Must precede every downstream outcome.
    outcomes=[]
    for repeat in range(config['repeats']):
        seed=base+1000+repeat*100
        backend.context={'phase':'baseline','problem_id':pid,'repeat':repeat}
        row=rollout(backend,task,prompt,retained,'continue',seed,budget-initial['generated_tokens'],reserve)
        row.update(problem_id=pid,repeat=repeat,candidate=None,selectors=['continue'],
                   episode_generated_tokens=initial['generated_tokens']+row['generated_tokens'],
                   episode_prompt_tokens=initial['prompt_tokens']+row['prompt_tokens_processed'],
                   episode_service_seconds=initial['elapsed_seconds']+row['elapsed_seconds'],
                   selector_generated_tokens=0,selector_prompt_tokens=0,selector_service_seconds=0.)
        record('outcomes',row);outcomes.append(row)
        for j,g in enumerate(candidates):
            backend.context={'phase':'shadow_continuation','problem_id':pid,'candidate':j,'repeat':repeat}
            prefix=retained+g.token_ids
            if g.finish_reason in ('stop','timeout'):
                text=backend.decode(prefix)
                row={'action':'continue','seed':seed,'text':text,'outcome':verify(task,text),
                     'generated_tokens':0,'prompt_tokens_processed':0,'elapsed_seconds':0.,'calls':[],'overhead':[]}
            else:
                row=rollout(backend,task,prompt,prefix,'continue',seed,budget-initial['generated_tokens']-pool_tokens,reserve)
            row.update(problem_id=pid,repeat=repeat,candidate=j,
                selectors=[name for name,choice in choices.items() if choice==j],
                episode_generated_tokens=initial['generated_tokens']+pool_tokens+row['generated_tokens'],
                episode_prompt_tokens=initial['prompt_tokens']+pool_prompts+row['prompt_tokens_processed'],
                episode_service_seconds=initial['elapsed_seconds']+pool_seconds+row['elapsed_seconds'],
                selector_generated_tokens=local['generation']['generated_tokens'],
                selector_prompt_tokens=local['generation']['prompt_tokens'],
                selector_service_seconds=local['generation']['elapsed_seconds'],
                acquisition_note='Selector costs apply to local_semantic only; cheap selectors use zero additional generation.',
                pool_generated_tokens=pool_tokens,selected_candidate_generated_tokens=g.generated_tokens)
            if row['episode_generated_tokens']>budget:raise ValueError('Generator budget exceeded')
            record('outcomes',row);outcomes.append(row)
    return decision,outcomes
