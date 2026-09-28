"""Bounded CUDA infrastructure qualification; never a research efficacy result."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import snapshot_download,constants
assert constants.HF_HUB_DISABLE_XET, "Use standard HTTP: Xet crashed on this cluster"

MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
OUT=Path('outputs');OUT.mkdir(exist_ok=False)
metadata={'model':MODEL,'revision':REVISION,'dtype':'bfloat16','quantization':None,
          'purpose':'CUDA/runtime qualification only; not interchangeable with MLX 4-bit results',
          'job_id':os.environ.get('SLURM_JOB_ID'),'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','huggingface-hub')},
          'download_backend':'HTTP (HF_HUB_DISABLE_XET=1)','cuda':torch.version.cuda,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
save('manifest.json',metadata)
assert torch.cuda.is_available(),'Allocated CUDA GPU unavailable'
metadata['gpu']=torch.cuda.get_device_name();metadata['capability']=torch.cuda.get_device_capability()
metadata['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
save('manifest.json',metadata)
start=time.monotonic()
snapshot=snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer*','vocab*','merges*'],max_workers=2)
footprint=sum(p.stat().st_size for p in Path(snapshot).iterdir() if p.is_file())
assert footprint<16*1024**3,'Model exceeds approved 16 GiB footprint'
metadata['snapshot_bytes']=footprint
metadata['download_seconds']=time.monotonic()-start
metadata['file_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(snapshot).iterdir() if p.is_file() and p.suffix!='.safetensors'}
save('manifest.json',metadata)
tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,padding_side='left',use_fast=False)
fixtures=json.loads(Path('tokenizer_fixture.json').read_text())
for r in fixtures:
    assert tokenizer.encode(r['text'],add_special_tokens=False)==r['ids']
    assert tokenizer.apply_chat_template([{'role':'user','content':r['text']}],tokenize=True,add_generation_prompt=True,return_dict=False)==r['chat_ids']
metadata['tokenizer_class']=type(tokenizer).__name__
metadata['tokenizer_fixture_count']=len(fixtures)
metadata['tokenizer_fixture_sha256']=hashlib.sha256(Path('tokenizer_fixture.json').read_bytes()).hexdigest()
save('manifest.json',metadata)
model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id

def greedy(ids,n):
    x=torch.tensor([ids],device='cuda');mask=torch.ones_like(x)
    with torch.inference_mode():
        out=model.generate(input_ids=x,attention_mask=mask,max_new_tokens=n,do_sample=False,
                           pad_token_id=tokenizer.pad_token_id,eos_token_id=None,
                           min_new_tokens=n)
    return out[0,x.shape[1]:].tolist()

prompt=tokenizer.apply_chat_template([{'role':'user','content':'Explain how to find a shortest path in a graph with nonnegative edge weights.'}],tokenize=True,add_generation_prompt=True,return_dict=False)
torch.cuda.synchronize();start=time.monotonic()
whole=greedy(prompt,64);first=greedy(prompt,32);rest=greedy(prompt+first,32)
torch.cuda.synchronize()
checks={'greedy_exact_prefix_resume':whole==first+rest,'generated_tokens':len(whole),
        'elapsed_seconds':time.monotonic()-start,'whole_ids':whole,'resumed_ids':first+rest}
save('resume_check.json',checks)
# Longer resident context and batched inference are distinct infrastructure probes.
long_ids=(prompt+tokenizer.encode(' graph edge weight '*3000,add_special_tokens=False))[:8192]
start=time.monotonic();long_output=greedy(long_ids,64);torch.cuda.synchronize()
save('long_context.json',{'input_tokens':len(long_ids),'output_tokens':len(long_output),'seconds':time.monotonic()-start})
prompts=[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in
         ['Compute 17*23.','Explain binary search.','Give a small counterexample to a false universal claim.','Explain Dijkstra algorithm.']]
batch=tokenizer(prompts,return_tensors='pt',padding=True).to('cuda')
start=time.monotonic()
with torch.inference_mode():out=model.generate(**batch,max_new_tokens=256,do_sample=False,pad_token_id=tokenizer.pad_token_id)
torch.cuda.synchronize();elapsed=time.monotonic()-start
length=batch['input_ids'].shape[1];outputs=[]
for tokens in out[:,length:].tolist():
    if tokenizer.eos_token_id in tokens:tokens=tokens[:tokens.index(tokenizer.eos_token_id)+1]
    outputs.append({'token_ids':tokens,'text':tokenizer.decode(tokens,skip_special_tokens=True)})
save('batch.json',{'outputs':outputs,'elapsed_seconds':elapsed,'emitted_tokens':sum(len(x['token_ids']) for x in outputs)})
save('summary.json',{'status':'passed' if checks['greedy_exact_prefix_resume'] else 'resume_mismatch',
     'gpu':metadata['gpu'],'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
     'batch_tokens_per_second':sum(len(x['token_ids']) for x in outputs)/elapsed,
     'interpretation':'Qualification only. Backend/precision changes require fresh frozen scientific evaluations.'})
assert checks['greedy_exact_prefix_resume'],'Exact greedy resume mismatch; do not scale this backend yet'
