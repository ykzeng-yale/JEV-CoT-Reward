"""Pinned-local BF16 CUDA adapter; separate protocol from quantized MLX."""
import hashlib
import json
from pathlib import Path
import time
from .mlx_backend import Generation


class CUDABackend:
    def __init__(self,model_path,temperature=.7,top_p=.9,max_context_tokens=16384):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        path=Path(model_path)
        if not path.is_dir():raise ValueError('Existing local snapshot required')
        if not torch.cuda.is_available():raise RuntimeError('CUDA allocation required')
        self.torch=torch;self.temperature=temperature;self.top_p=top_p;self.max_context_tokens=max_context_tokens
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_fast=False)
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        self.quantization_config={'bits':16,'mode':'unquantized_bfloat16'}
        self.model_path=str(path);self._sampler=temperature
    def encode_chat(self,messages):
        return list(self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False))
    def encode_text(self,text):return self.tokenizer.encode(text,add_special_tokens=False)
    def decode(self,ids):return self.tokenizer.decode(list(ids),skip_special_tokens=True)
    def generate(self,prefix_ids,max_tokens,seed,timeout_s=180,stop_when=None):
        torch=self.torch;prefix=list(prefix_ids)
        if not prefix or any(type(t) is not int or t<0 for t in prefix):raise ValueError('Invalid prefix')
        if type(max_tokens) is not int or max_tokens<0 or len(prefix)+max_tokens>self.max_context_tokens:raise ValueError('Invalid context allowance')
        rng=torch.Generator(device='cuda').manual_seed(seed)
        ids=[];logprobs=[];entropies=[];finish='length';cache=None
        x=torch.tensor([prefix],device='cuda');start=time.monotonic()
        eos=self.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        with torch.inference_mode():
            for _ in range(max_tokens):
                out=self.model(input_ids=x,past_key_values=cache,use_cache=True)
                cache=out.past_key_values;logits=out.logits[0,-1].float()
                base=torch.log_softmax(logits,dim=-1);p=base.exp()
                entropy=float(-(torch.where(p>0,p*base,0.)).sum().item())
                temperature=float(self._sampler)
                if temperature==0:token=int(torch.argmax(logits).item())
                else:
                    probs=torch.softmax(logits/temperature,dim=-1)
                    ordered,indices=torch.sort(probs,descending=True)
                    mask=ordered.cumsum(0)-ordered>=self.top_p
                    ordered=ordered.masked_fill(mask,0.)
                    token=int(indices[torch.multinomial(ordered,1,generator=rng)].item())
                ids.append(token);logprobs.append(float(base[token].item()));entropies.append(entropy)
                if token in eos:finish='stop';break
                if stop_when and stop_when(ids):finish='checkpoint';break
                if time.monotonic()-start>=timeout_s:finish='timeout';break
                x=torch.tensor([[token]],device='cuda')
        torch.cuda.synchronize()
        return Generation(ids,self.decode(ids),len(prefix),len(ids),sum(logprobs)/len(ids) if ids else None,
                          sum(entropies)/len(ids) if ids else None,time.monotonic()-start,finish,
                          torch.cuda.max_memory_allocated()/1e9,hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),
                          seed,logprobs,entropies)
