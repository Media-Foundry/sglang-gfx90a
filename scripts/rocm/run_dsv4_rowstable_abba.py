#!/usr/bin/env python3
"""Controlled A(candidate),B(baseline),B,A service-level acceptance run."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-flag', choices=[
        'SGLANG_DSV4_GFX90A_ROW_STABLE_PREFILL',
        'SGLANG_DSV4_GFX90A_TP8_C1_SHARED_GATE_ROUND',
    ], default='SGLANG_DSV4_GFX90A_ROW_STABLE_PREFILL')
    a=p.parse_args()
    service=psutil.Process(a.pid)
    cmd,env,cwd=service.cmdline(),service.environ(),service.cwd()
    assert cmd[1:3]==['-m','sglang.launch_server']
    assert '--tp-size' in cmd and cmd[cmd.index('--tp-size')+1]=='8'
    assert cmd[cmd.index('--max-total-tokens')+1]=='1048576'
    assert cmd[cmd.index('--port')+1]=='30011'
    assert env.get('SGLANG_DSV4_GFX90A_ROW_STABLE_PREFILL')=='1'
    assert not any(k.startswith('SGLANG_DSV4_DEBUG_') for k in env)
    state={'status':'running','protocol':'A(candidate),B(baseline),B,A',
           'candidate_flag':a.candidate_flag,'service_pid':service.pid,'blocks':[]}
    def save():
        a.output.write_text(json.dumps(state,indent=2)+'\n')
    def resource_check():
        owned={service.pid,*[c.pid for c in service.children(recursive=True)]}
        seen=set()
        def walk(x):
            if isinstance(x,dict):
                if 'pid' in x:seen.add(int(x['pid']))
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(json.loads(subprocess.check_output(['amd-smi','process','--json'])))
        assert not seen-owned,('external GPU processes',seen-owned)
    def ready():
        deadline=time.monotonic()+300
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic()<deadline:
            assert service.is_running() and service.status()!=psutil.STATUS_ZOMBIE
            try:
                with opener.open('http://127.0.0.1:30011/health',timeout=3) as r:
                    if r.status==200:return
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError('service readiness; do not restart automatically')
    def bench(arguments,path):
        with path.with_suffix('.log').open('w') as log:
            subprocess.run([cmd[0],*arguments,'--output',str(path)],cwd=cwd,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        active=env.get(a.candidate_flag, '0') == '1'
        for index,wanted in enumerate([True,False,False,True]):
            resource_check()
            if active!=wanted:
                children=service.children(recursive=True)
                service.terminate()
                _,alive=psutil.wait_procs([service,*children],timeout=20)
                for child in alive:child.terminate()
                _,alive=psutil.wait_procs(alive,timeout=10)
                assert not alive,[(x.pid,x.status()) for x in alive]
                env[a.candidate_flag]=str(int(wanted))
                with a.output.with_name(a.output.stem+f'_service_{index}.log').open('w') as log:
                    child=subprocess.Popen(['numactl','--interleave=all',*cmd],cwd=cwd,
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                service=psutil.Process(child.pid)
                state['service_pid']=service.pid;active=wanted;save()
            ready()
            block={'index':index,'candidate':wanted,'status':'c1','service_pid':service.pid}
            state['blocks'].append(block);save()
            root=a.output.with_name(a.output.stem+f'_{index}')
            c1=Path(str(root)+'_c1.json');c32=Path(str(root)+'_c32.json')
            bench(['scripts/rocm/bench_dsv4_c1_mhc_recovery.py','--arm',f'ABBA{index}',
                   '--rounds','2','--skip-freeze-gc','--reference',
                   '/tmp/dsv4_runtime_m_c1_B_20260908.json'],c1)
            if a.candidate_flag == 'SGLANG_DSV4_GFX90A_TP8_C1_SHARED_GATE_ROUND':
                result=json.loads(c1.read_text())
                reference=json.loads(Path('/tmp/dsv4_runtime_m_c1_B_20260908.json').read_text())
                expected={row['case']:row['output_ids'] for row in reference['measurements']
                          if row['rep']==0}
                assert result['france_exact']
                measured=[row for row in result['measurements'] if row['rep']>=0]
                assert len(measured)==6
                assert all(row['output_ids']==expected[row['case']] for row in measured)
                block['c1_reference_exact']=len(measured)
            block.update(status='c32',c1=str(c1));save()
            resource_check()
            bench(['scripts/rocm/bench_dsv4_tp4_diverse_concurrent.py','--base-url',
                   'http://127.0.0.1:30011','--inputs',
                   '/tmp/dsv4_tp8_c32_ar_code_workload_20260908.json','--request-count',
                   '32','--tokens','256','--rounds','6','--save-output-ids'],c32)
            block.update(status='complete',c32=str(c32));save()
            print('completed block',index,'candidate',wanted,flush=True)
        state['status']='complete';save()
    except Exception as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
