#!/usr/bin/env python3
"""Run frozen stdlib mathematical checks on an enrolled Mac; no model/API use.

Uses existing SSH aliases and their strict pins. The private task directory and
remote account names are not included in the public result. Leaves its tiny
immutable input and logs on the remote host for audit; never touches peer jobs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import uuid


SUPERVISOR = '''import hashlib,json,os,pathlib,resource,runpy,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; supervisor_expected=sys.argv[3]
source=root/'inputs'/'validate_theory.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
assert hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()==supervisor_expected
if len(sys.argv)>4 and sys.argv[4]=='--worker':
 resource.setrlimit(resource.RLIMIT_CPU,(60,60))
 resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024*1024,16*1024*1024))
 identity={'pid':os.getpid(),'process_group':os.getpgrp(),'os_process_start':subprocess.check_output(['/bin/ps','-p',str(os.getpid()),'-o','lstart='],text=True).strip(),'started_unix':time.time(),'python_version':sys.version.split()[0],'source_sha256':expected,'supervisor_sha256':supervisor_expected}
 (root/'outputs'/'process.json').write_text(json.dumps(identity))
 sys.argv=[str(source),'--output',str(root/'outputs'/'validation.json')]
 runpy.run_path(str(source),run_name='__main__')
 sys.exit(0)
start=time.time()
with (root/'outputs'/'stdout.log').open('w') as out, (root/'outputs'/'stderr.log').open('w') as err:
 child=subprocess.Popen([sys.executable,'-I','-B',__file__,str(root),expected,supervisor_expected,'--worker'],stdout=out,stderr=err,start_new_session=True)
 timed_out=False
 try: code=child.wait(timeout=120)
 except subprocess.TimeoutExpired:
  timed_out=True;child.kill();code=child.wait()
identity_path=root/'outputs'/'process.json'
identity=json.loads(identity_path.read_text()) if identity_path.exists() else {'pid':child.pid,'identity_capture_failed':True,'source_sha256':expected,'supervisor_sha256':supervisor_expected}
receipt={**identity,'supervisor_started_unix':start,'finished_unix':time.time(),'exit_code':code,'timed_out':timed_out,'worker_cpu_seconds_limit':60,'worker_wall_seconds_limit':120,'per_file_bytes_limit':16777216,'memory_limit_note':'No enforced RSS bound; trusted stdlib finite-state validation only, one worker.'}
(root/'outputs'/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
sys.exit(code if 0<=code<=255 else 1)
'''


def call(command, **kwargs):
    return subprocess.run(command, check=kwargs.pop('check', True), text=True, capture_output=True, timeout=kwargs.pop('timeout', 30), **kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['mac-mini','mac-aux'],default='mac-mini')
    p.add_argument('--remote-python',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('Refusing to overwrite a previous validation result')
    source=Path(__file__).resolve().parent/'validate_theory.py'
    frozen=source.read_bytes(); digest=hashlib.sha256(frozen).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    supervisor_digest=hashlib.sha256(SUPERVISOR.encode()).hexdigest()
    task_name=f'{stamp}-theory-{digest[:12]}-{uuid.uuid4().hex[:8]}'
    options=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','ForwardAgent=no']
    ssh=['ssh','-T',*options,args.host]
    scp=['scp','-q',*options]
    setup="import pathlib,shutil,sys; base=pathlib.Path.home()/'.local/share/jev-cot-reward/tasks'; assert shutil.disk_usage(pathlib.Path.home()).free >= 10*1024**3, 'Less than 10 GiB disk reserve'; base.mkdir(parents=True,exist_ok=True); root=base/sys.argv[1]; root.mkdir(mode=0o700); (root/'inputs').mkdir(); (root/'outputs').mkdir(); print(root)"
    remote=call(ssh+[f'{shlex.quote(args.remote_python)} -I -B -c {shlex.quote(setup)} {shlex.quote(task_name)}']).stdout.strip()
    # The enrolled hosts use simple POSIX home paths; fail closed before scp.
    if not remote.startswith('/Users/') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-' for c in remote):
        raise ValueError('Unexpected remote task path')
    with tempfile.TemporaryDirectory(prefix='jev-remote-validation-') as temporary:
        tmp=Path(temporary)
        (tmp/'validate_theory.py').write_bytes(frozen)
        (tmp/'supervisor.py').write_text(SUPERVISOR)
        call(scp+[str(tmp/'validate_theory.py'),str(tmp/'supervisor.py'),f'{args.host}:{remote}/inputs/'])
        seal="import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1])/'inputs'; names=['validate_theory.py','supervisor.py']; expected=sys.argv[2:]; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in zip(names,expected,strict=True)); [(p/n).chmod(0o444) for n in names]; p.chmod(0o555)"
        call(ssh+[f'{shlex.quote(args.remote_python)} -I -B -c {shlex.quote(seal)} {shlex.quote(remote)} {digest} {supervisor_digest}'])
        run=call(ssh+[f'{shlex.quote(args.remote_python)} -I -B {shlex.quote(remote+"/inputs/supervisor.py")} {shlex.quote(remote)} {digest} {supervisor_digest}'],timeout=150,check=False)
        call(scp+[f'{args.host}:{remote}/outputs/receipt.json',str(tmp/'receipt.json')])
        receipt=json.loads((tmp/'receipt.json').read_text())
        if run.returncode == 0:
            call(scp+[f'{args.host}:{remote}/outputs/validation.json',str(tmp/'validation.json')])
            result=json.loads((tmp/'validation.json').read_text())
        else:
            result={'all_checks_passed':False,'checks':{},'status':'remote_worker_failed'}
        assert receipt['source_sha256']==digest and receipt['supervisor_sha256']==supervisor_digest
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as output:
            output.write(json.dumps({'host_alias':args.host,'task_name':task_name,'receipt':receipt,'validation':result},indent=2)+'\n')
    print(json.dumps({'host_alias':args.host,'output':str(args.output),'all_checks_passed':result['all_checks_passed'],'checks':len(result['checks']),'source_sha256':digest}))
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__=='__main__':main()
