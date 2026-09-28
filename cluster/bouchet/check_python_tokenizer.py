import json
from pathlib import Path
from transformers import AutoTokenizer
p=Path('/home/yz2324/project_pi_fl426/yz2324/JEV-CoT-Reward/cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554')
t=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False,use_fast=False)
fixtures=json.loads(Path('tokenizer_fixture.json').read_text())
for r in fixtures:
    assert t.encode(r['text'],add_special_tokens=False)==r['ids']
    assert t.apply_chat_template([{'role':'user','content':r['text']}],tokenize=True,add_generation_prompt=True)==r['chat_ids']
print(json.dumps({'status':'passed','class':type(t).__name__,'is_fast':t.is_fast,'fixture_count':len(fixtures)}))
