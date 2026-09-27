import math
import pytest
from jev_control.branch_selectors import CandidateView,choose,local_prompt,parse_local_choice


def test_uniform_selection_is_reproducible_and_scores_disagree_without_gold():
    candidates=[CandidateView('confident',-.1,2.),CandidateView('careful',-.5,.2)]
    assert choose(candidates,'likelihood')==0
    assert choose(candidates,'entropy_reduction')==1
    assert choose(candidates,'uniform',17)==choose(candidates,'uniform',17)
    assert {choose(candidates,'uniform',s) for s in range(20)}=={0,1}


def test_missing_scores_do_not_accidentally_win_or_silently_fallback():
    candidates=[CandidateView('missing',None,float('nan')),CandidateView('valid',-2.,1.)]
    assert choose(candidates,'likelihood')==choose(candidates,'entropy_reduction')==1
    with pytest.raises(ValueError,match='fallback'):choose(candidates[:1],'entropy_reduction')
    with pytest.raises(ValueError,match='Empty'):choose([],'uniform')


@pytest.mark.parametrize('text',['{"choice":true}','{"choice":-1}','{"choice":2}',
    '{"choice":0,"reason":"x"}','not JSON','```json\n{"choice":0}\n```'])
def test_local_choice_rejects_invalid_outputs(text):
    with pytest.raises(ValueError):parse_local_choice(text,2)


def test_local_choice_and_prompt_use_only_observable_text():
    c=CandidateView('next',-1.,.5)
    assert parse_local_choice(' {"choice": 1} ',2)==1
    prompt=local_prompt('question','prefix',[c])
    assert 'question' in prompt and 'prefix' in prompt and 'next' in prompt
    assert 'mean_logprob' not in prompt and 'mean_entropy' not in prompt
