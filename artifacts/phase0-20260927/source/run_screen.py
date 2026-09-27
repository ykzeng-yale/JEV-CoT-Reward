#!/usr/bin/env python3
"""Collect all-arm public-task replays. Phase 0 only; no learned policy claim."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_control.tasks import make_task, verify
from jev_control.features import QUESTIONS, SCHEMA_VERSION, judge_state
from jev_control.mlx_backend import MLXBackend

ACTIONS = ("continue", "repair", "branch")
REPAIR = "\n\nLet me re-check the last reasoning step and correct any specific error before proceeding.\n"
FINAL = "\n\nI must now provide the required final answer without further explanation.\nFINAL: "


def boundary(backend, tokens, minimum=32):
    """Choose a completed paragraph without re-encoding the prefix.

    Initial generation after this boundary is discarded and STILL charged. No
    future text is supplied as a feature. No boundary => explicitly ineligible.
    """
    for end in range(len(tokens), minimum - 1, -1):
        text = backend.decode(tokens[:end])
        if text.endswith("\n\n") and "FINAL:" not in text:
            return end
    return None


def rollout(backend, task, prompt, retained, action, seed, remaining, final_reserve):
    calls=[]; overhead=[]; assistant=list(retained); spent=0
    if action == "repair":
        # Remove last complete paragraph (at most 128 retained tokens).
        cutoff=0
        for i in range(len(retained)-1, max(0,len(retained)-128), -1):
            if backend.decode(retained[:i]).endswith("\n\n"):
                cutoff=i; break
        cutoff=max(cutoff,len(retained)-128)
        assistant=retained[:cutoff]+backend.encode_text(REPAIR)
        overhead.append({"kind":"repair_instruction","tokens":len(backend.encode_text(REPAIR)),"removed_tokens":len(retained)-cutoff})
    if action == "branch":
        cap=min(64,max(1,(remaining-final_reserve)//3))
        candidates=[backend.generate(prompt+assistant,cap,seed+100+j,timeout_s=180) for j in range(2)]
        calls.extend(candidates); spent+=sum(x.generated_tokens for x in candidates)
        winner=backend.choose_candidate(candidates)
        assistant+=candidates[winner].token_ids
        overhead.append({"kind":"branch_selection","winner":winner,"selector":"mean_base_logprob"})
        done=candidates[winner].finish_reason=="stop"
    else:
        done=False
    allowance=remaining-spent-final_reserve
    if not done and allowance>0:
        g=backend.generate(prompt+assistant,allowance,seed+200,timeout_s=180)
        calls.append(g); spent+=g.generated_tokens; assistant+=g.token_ids
        done=g.finish_reason=="stop"
    if not done and "FINAL:" not in backend.decode(assistant) and remaining-spent>0:
        injected=backend.encode_text(FINAL); assistant+=injected
        overhead.append({"kind":"finalization_instruction","tokens":len(injected)})
        g=backend.generate(prompt+assistant,remaining-spent,seed+300,timeout_s=180)
        calls.append(g); spent+=g.generated_tokens; assistant+=g.token_ids
    assert spent<=remaining
    text=backend.decode(assistant)
    return {"action":action,"seed":seed,"outcome":verify(task,text),"text":text,
            "generated_tokens":spent,"prompt_tokens_processed":sum(x.prompt_tokens for x in calls),
            "elapsed_seconds":sum(x.elapsed_seconds for x in calls),
            "calls":[x.to_dict() for x in calls],"overhead":overhead}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--problems",type=int,default=6)
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--budget",type=int,default=512)
    p.add_argument("--prefix-tokens",type=int,default=128)
    p.add_argument("--final-reserve",type=int,default=96)
    p.add_argument("--quantize-bits",type=int,default=4)
    p.add_argument("--seed",type=int,default=20260927)
    p.add_argument("--jev",action="store_true")
    p.add_argument("--output",type=Path,default=Path("runs/phase0"))
    args=p.parse_args()
    if not 1<=args.problems<=48 or not 1<=args.repeats<=16:
        p.error("Phase 0 bounds are 1–48 problems and 1–16 repeats")
    if args.prefix_tokens+args.final_reserve+128>args.budget:
        p.error("Insufficient continuation budget")
    if args.output.exists(): p.error("Output already exists; use a new directory to avoid mixing runs")
    args.output.mkdir(parents=True)
    manifest={**vars(args),"output":str(args.output),"kind":"phase0_instrumentation",
              "schema":SCHEMA_VERSION,"started_unix":time.time(),"selector":"mean_base_logprob",
              "note":"Explicit observable rationales; non-thinking instruct generator. Not a confirmatory study."}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    backend=MLXBackend(args.model,quantization_bits=args.quantize_bits)
    client=None
    if args.jev:
        from jev_control.jev import JevClient
        client=JevClient(stage_cap_usd=.25)
    rows=[]; checkpoints=[]; skipped=[]
    for index in range(args.problems):
        task=make_task(index,args.seed)
        prompt=backend.encode_chat([{"role":"user","content":task["prompt"]}])
        initial=backend.generate(prompt,args.prefix_tokens,args.seed+index,timeout_s=180)
        end=boundary(backend,initial.token_ids)
        if initial.finish_reason=="stop" or "FINAL:" in initial.text or end is None:
            record={"task":task,"initial":initial.to_dict(),"reason":"completed_before_checkpoint" if initial.finish_reason=="stop" or "FINAL:" in initial.text else "no_paragraph_boundary"}
            skipped.append(record)
            (args.output/"skipped.json").write_text(json.dumps(skipped,indent=2)+"\n")
            print(json.dumps({"problem":index,"status":"ineligible","reason":record["reason"]}),flush=True)
            continue
        retained=initial.token_ids[:end]
        text=backend.decode(retained)
        parts=text.rstrip().rsplit("\n\n",1)
        history=parts[0] if len(parts)>1 else ""
        segment=parts[-1]
        checkpoint={"problem_id":task["id"],"task":task,"prompt_ids":prompt,"retained_ids":retained,
                    "initial":initial.to_dict(),"discarded_prefix_tail_tokens":len(initial.token_ids)-end,
                    "state":judge_state(task["prompt"],history,segment)}
        checkpoint["sha256"]=hashlib.sha256(json.dumps({"prompt":prompt,"retained":retained},sort_keys=True).encode()).hexdigest()
        if client: checkpoint["jev"]=client.evaluate(checkpoint["state"],QUESTIONS)
        checkpoints.append(checkpoint)
        with (args.output/"checkpoints.jsonl").open("a") as f: f.write(json.dumps(checkpoint)+"\n")
        schedule=[(action,r) for action in ACTIONS for r in range(args.repeats)]
        random.Random(args.seed+index).shuffle(schedule)
        for action,repeat in schedule:
            # Separate reproducible arm/replicate seeds. No false common-RNG claim.
            seed=args.seed+index*10000+ACTIONS.index(action)*1000+repeat*10
            row=rollout(backend,task,prompt,retained,action,seed,args.budget-initial.generated_tokens,args.final_reserve)
            row.update({"problem_id":task["id"],"family":task["family"],"checkpoint_sha256":checkpoint["sha256"],"repeat":repeat,"shared_prefix_generated_tokens":initial.generated_tokens})
            assert row["generated_tokens"]+initial.generated_tokens<=args.budget
            rows.append(row)
            with (args.output/"outcomes.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps({"problem":index,"action":action,"repeat":repeat,"success":row["outcome"]["success"],"generated_tokens":row["generated_tokens"],"seconds":round(row["elapsed_seconds"],2)}),flush=True)
    summary={"kind":"real_model_instrumentation_not_confirmatory","attempted_problems":args.problems,
             "eligible_problems":len(checkpoints),"skipped_problems":len(skipped),"episodes":len(rows),
             "actions":{a:{"successes":sum(r["outcome"]["success"] for r in rows if r["action"]==a),"n":sum(r["action"]==a for r in rows)} for a in ACTIONS},
             "continuation_generated_tokens":sum(r["generated_tokens"] for r in rows),
             "continuation_seconds":sum(r["elapsed_seconds"] for r in rows),
             "actual_shared_prefix_tokens":sum(c["initial"]["generated_tokens"] for c in checkpoints)+sum(s["initial"]["generated_tokens"] for s in skipped),
             "jev_budget":client.status() if client else None,
             "interpretation":"All-arm data collection, not evaluation of a Jev-guided policy. Repeats are clustered within problems."}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
