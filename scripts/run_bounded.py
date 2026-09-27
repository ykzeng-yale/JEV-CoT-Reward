#!/usr/bin/env python3
"""Supervise one authorized project command and retain its immutable receipt.

RSS is observed once per second, not a hard limit on Metal/unified memory or
descendants. Wall/output guards stop only the process group created here.
Keep control receipts private when commands contain local filesystem paths.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run(command, control, *, wall_seconds, rss_mib, output_mib, poll_seconds=1):
    if not command or min(wall_seconds, rss_mib, output_mib, poll_seconds) <= 0:
        raise ValueError("Positive bounds and a command are required")
    control=Path(control)
    control.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    env=dict(os.environ)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    with (control/'stdout.log').open('wb') as out, (control/'stderr.log').open('wb') as err:
        worker=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
        identity=subprocess.run(['ps','-p',str(worker.pid),'-o','lstart='],text=True,capture_output=True,check=False).stdout.strip()
        receipt={"pid":worker.pid,"process_group":worker.pid,"os_process_start":identity,
                 "started_utc":datetime.now(timezone.utc).isoformat(),"command":command,
                 "wall_seconds_cap":wall_seconds,"observed_rss_mib_cap":rss_mib,
                 "combined_log_mib_cap":output_mib,"max_observed_rss_mib":0,
                 "memory_caveat":"RSS sampling excludes some GPU/shared/child memory and can overshoot between samples; not a hard physical-memory limit.",
                 "status":"running"}
        record=control/'process.json'
        record.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({"pid":worker.pid,"status":"running","control":str(control)}),flush=True)
        stop_reason=None
        try:
            while worker.poll() is None:
                rss=subprocess.run(['ps','-p',str(worker.pid),'-o','rss='],text=True,capture_output=True,check=False).stdout.strip()
                measured=int(rss)/1024 if rss.isdigit() else 0
                receipt['max_observed_rss_mib']=max(receipt['max_observed_rss_mib'],measured)
                log_bytes=sum((control/name).stat().st_size for name in ['stdout.log','stderr.log'])
                if time.monotonic()-started>=wall_seconds: stop_reason='wall_limit'
                elif measured>rss_mib: stop_reason='observed_rss_limit'
                elif log_bytes>output_mib*1024**2: stop_reason='log_limit'
                if stop_reason: break
                time.sleep(poll_seconds)
        except BaseException:
            stop_reason='supervisor_interrupted'
            raise
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid,signal.SIGTERM)
                try: worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGKILL);worker.wait()
            receipt.update(status='completed' if worker.returncode==0 and stop_reason is None else 'failed',
                           exit_code=worker.returncode,stop_reason=stop_reason,
                           finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
            record.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,required=True)
    p.add_argument('--rss-mib',type=float,default=8192)
    p.add_argument('--output-mib',type=float,default=64)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    receipt=run(command,args.control,wall_seconds=args.wall_seconds,rss_mib=args.rss_mib,output_mib=args.output_mib)
    print(json.dumps({k:receipt[k] for k in ['pid','status','exit_code','stop_reason','elapsed_seconds','max_observed_rss_mib']}),flush=True)
    if receipt['status']!='completed': raise SystemExit(1)


if __name__=='__main__':main()
