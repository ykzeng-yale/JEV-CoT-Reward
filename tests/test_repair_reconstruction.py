from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_action_qualification import reconstruct
from jev_control.repair_actions import ACTIONS,RECHECK,SHAM,SEGMENT,prepare

class Tokens:
    def decode(self,ids):return ''.join(map(chr,ids))
    def encode(self,text):return list(map(ord,text))
    encode_text=encode


def test_independent_treatment_reconstruction_across_boundary_cases():
    tokenizer=Tokens();constants={'RECHECK':RECHECK,'SHAM':SHAM,'SEGMENT':SEGMENT}
    for text in ('','single\n','previous\n\nlatest\n\n','x'*260+'\n','x'*140+'\n\n'+'y'*140+'\n'):
        retained=tokenizer.encode(text)
        for action in ACTIONS:
            assert reconstruct(tokenizer,retained,action,constants)==prepare(tokenizer,retained,action)
