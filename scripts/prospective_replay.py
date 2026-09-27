"""Reuse inspected token-chain contracts with the prospective call context.

This is one component of the prospective audit, not a completed-run certificate.
It never runs generation or changes stored outcomes.
"""
import ast
from collections import Counter
from fractions import Fraction
import heapq
import random
import re

import audit_mechanism as mechanism


def checked_task_contracts(run, manifest):
    """Approve frozen task/rollout function ASTs; execute only approved task nodes."""
    trees = {}
    for name, digest in manifest['source_sha256'].items():
        mechanism.safe_relative(name)
        path = run/'source'/name
        mechanism.require(mechanism.digest(path.read_bytes()) == digest, 'Frozen source digest mismatch')
        if name in mechanism.CONTRACTS:
            trees[name] = ast.parse(path.read_text())
    nodes = []
    for name, functions in mechanism.CONTRACTS.items():
        mechanism.require(name in trees, 'Missing required frozen contract source')
        for function_name, expected in functions.items():
            node = mechanism.function(trees[name], function_name)
            canonical = ast.dump(node, include_attributes=False).replace(', type_params=[]', '')
            mechanism.require(mechanism.digest(canonical.encode()) == expected, 'Unsupported frozen function contract')
            if name == 'src/jev_control/tasks.py':nodes.append(node)
    namespace={'ast':ast,'Counter':Counter,'Fraction':Fraction,'heapq':heapq,'random':random,'re':re}
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(module,'<approved task contracts>','exec'),namespace)
    constants={name:mechanism.literal(trees['scripts/run_screen.py'],name) for name in ('FINAL','REPAIR')}
    return constants, namespace


class ProspectiveLedger:
    def __init__(self, ledger):
        self.ledger=ledger

    def take(self, context, *args, **kwargs):
        # The prospective launch records problem/action, without mechanism-only
        # repeated-arm fields. All other call/prefix/seed checks remain exact.
        context={k:v for k,v in context.items() if k not in ('index','repeat')}
        return self.ledger.take(context,*args,**kwargs)


def audit_selected_rollout(row, item, checkpoint, config, tokenizer, constants, ledger):
    initial=checkpoint['initial']
    if initial['finish_reason'] in ('stop','timeout'):
        mechanism.require(row['action']=='continue', 'Finished/ineligible episode was intervened on')
        mechanism.require(row['calls']==[] and row['overhead']==[], 'Completed prefix has unexpected continuation')
        mechanism.require(row['text']==initial['text'], 'Completed prefix text changed')
        for key in ('generated_tokens','prompt_tokens_processed','elapsed_seconds'):
            mechanism.equal_number(row[key],0,key)
        return 0
    # Explicit adaptation of record fields, never of outcome labels or call data.
    enriched={**row,'index':item['index'],'repeat':0,'family':item['task']['family'],
              'shared_prefix_generated_tokens':initial['generated_tokens']}
    scheduled={**item,'scheduled':{'action':row['action'],'seed':item['continuation_seed'],'repeat':0}}
    return mechanism.audit_rollout(enriched,scheduled,checkpoint,config,tokenizer,constants,ProspectiveLedger(ledger))
