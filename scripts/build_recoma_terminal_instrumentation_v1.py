#!/usr/bin/env python3
"""Prepare a distinct future runtime: record stop branches without changing returns."""
import argparse,hashlib,json
from pathlib import Path

def patch_controller(text):
    old='return last_message.action_json["arg1"] + last_message.action_json.get("thought", "")'
    new='state.data["recoma_terminal_trigger"] = "submit"\n                    '+old
    other='if task_completed():\n            return last_child.output'
    changed='if task_completed():\n            state.data["recoma_terminal_trigger"] = "official_completion"\n            return last_child.output'
    if text.count(old)!=1 or text.count(other)!=1:raise ValueError('source anchors differ')
    return text.replace(old,new).replace(other,changed)

def patch_llm_cap(text):
    old='if num_calls >= self.max_llm_calls:\n            logger.warning'
    if text.count(old)!=1:raise ValueError('LLM cap source anchor differs')
    return text.replace(old,'if num_calls >= self.max_llm_calls:\n            current_state.data["recoma_terminal_trigger"] = "llm_cap"\n            logger.warning')

def patch_action_cap(text):
    old='if env_calls >= self.max_env_calls:\n            logger.warning'
    if text.count(old)!=1:raise ValueError('action cap source anchor differs')
    return text.replace(old,'if env_calls >= self.max_env_calls:\n            current_state.data["recoma_terminal_trigger"] = "action_cap"\n            logger.warning')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('refuse overwrite or mutate frozen source')
    changes={'discoveryworld/agents/recoma/react_controller.py':patch_controller,'recoma/recoma/search/early_stopping.py':patch_llm_cap,'discoveryworld/agents/recoma/discoveryworld_env_models.py':patch_action_cap}
    a.output.mkdir(parents=True);receipt={}
    for rel,patch in changes.items():
        src=a.source/rel;out=a.output/rel;out.parent.mkdir(parents=True,exist_ok=True);new=patch(src.read_text());compile(new,str(out),'exec');out.write_text(new)
        receipt[rel]={'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'prepared_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    (a.output/'preparation-receipt.json').write_text(json.dumps({'status':'PARTIAL_RUNTIME_PREPARED_NOT_RELEASED','source':receipt,'remaining':['format/infrastructure hook','durable receipt integration','full end-to-end runtime audit','bounded post-phase compute envelope'],'behavior_boundary':'Only observed executed stop branches are tagged; official endpoints and controller returns unchanged. Not a complete release.'},indent=2)+'\n')
if __name__=='__main__':main()
