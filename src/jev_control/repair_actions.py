"""Development-only intervention definitions, independent of terminal gold labels."""
ACTIONS = ('continue', 'sham', 'suffix_repair', 'recheck', 'segment_repair')
RECHECK = '\n\nLet me re-check the last reasoning step and correct any specific error before proceeding.\n'
SHAM = '\n\nLet me continue the reasoning from this point and then provide the requested final answer.\n'
SEGMENT = '\n\nLet me reconstruct the next reasoning segment, checking each calculation and constraint before proceeding.\n'


def segment_start(backend, retained):
    """Last nonempty paragraph start, exactly on a token boundary; no re-encoding."""
    text=backend.decode(retained)
    nonempty_end=len(text.rstrip())
    # Find the latest double newline preceding content, not the stop delimiter.
    for end in range(len(retained)-1, 0, -1):
        prefix=backend.decode(retained[:end])
        if len(prefix)<=nonempty_end and prefix.endswith('\n\n'):
            return end
    return 0


def prepare(backend, retained, action):
    if action not in ACTIONS: raise ValueError('Unknown action')
    kept=list(retained);instruction='';removed=0
    if action=='suffix_repair':
        cutoff=0
        for i in range(len(kept)-1,max(0,len(kept)-128),-1):
            if backend.decode(kept[:i]).endswith('\n\n'):
                cutoff=i;break
        cutoff=max(cutoff,len(kept)-128)
        removed=len(kept)-cutoff;kept=kept[:cutoff];instruction=RECHECK
    elif action=='segment_repair':
        cutoff=segment_start(backend,kept)
        removed=len(kept)-cutoff;kept=kept[:cutoff];instruction=SEGMENT
    elif action=='recheck':instruction=RECHECK
    elif action=='sham':instruction=SHAM
    injected=backend.encode_text(instruction) if instruction else []
    return kept+injected, {'action':action,'removed_tokens':removed,
        'instruction':instruction,'inserted_tokens':len(injected),'kept_tokens':len(kept),
        'note':'Removed generation remains charged; inserted tokens incur prefill cost.'}
