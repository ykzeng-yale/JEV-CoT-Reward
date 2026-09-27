import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_prospective import chat_ids, restored_judge_prompt
from local_judge import judge_prompt


def test_chat_requests_token_ids_not_default_batch_encoding():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role':'user','content':'hello'}]
            return [1, 2] if kwargs.get('return_dict') is False else {'input_ids':[1,2]}
    assert chat_ids(Tokenizer(), 'hello') == [1,2]


def test_sorted_durable_state_restores_original_prompt_order():
    state={'task':'goal','history':'earlier','latest_segment':'now'}
    recovered=json.loads(json.dumps(state,sort_keys=True))
    assert list(recovered)!=list(state)
    assert restored_judge_prompt(recovered, {}) == judge_prompt(state,{})
