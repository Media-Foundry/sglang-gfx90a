"""Small native TP8 C32 service ABBA; unchanged best-AR settings except consumer.

Two natural-EOS waves/leg, same 32 real public-source cases in every wave.
No concurrent GPU benchmarks. Only this script's owned processes may be stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/pc/Code/sglang')
OLD=REPO/'.agents/experiments/dsv4_tp8_revalidation_20260913'
URL='http://127.0.0.1:30021'
FLAG='SGLANG_DSV4_GFX90A_TP8_M32_DOWN_CONSUMER'
sys.path.insert(0,str(REPO/'scripts/rocm'))
from bench_dsv4_tp8_mhc_fusion_drift_trial import get, post
from bench_dsv4_tp4_diverse_concurrent import completion_ids


def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')


def resources(name,proc=None):
    gpu=json.loads(subprocess.check_output(['amd-smi','process','--json']))
    active={p['process_info']['pid'] for g in gpu for p in g.get('process_list',[])
            if isinstance(p.get('process_info'),dict)}
    owned=set() if proc is None else {proc.pid,*[p.pid for p in proc.children(recursive=True)]}
    assert active<=owned, f'Unexpected GPU owners: {active-owned}'
    save(name+'.gpu.json',gpu)


def start(label,enabled):
    resources(label+'-before-start')
    log=ROOT/f'{label}.service.log'
    assert not log.exists() and not (ROOT/f'{label}.state.json').exists()
    command=['env', f'{FLAG}={enabled}',
             'SGLANG_DSV4_GFX90A_M32_DOWN_CONSUMER=0',
             'SGLANG_DSV4_GFX90A_M64_DOWN_CONSUMER=0',
             'SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP=1',
             'SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV=1',
             'bash',str(OLD/'start-ar-matrix.sh')]
    session=f'dsv4-ar-down-{label}-20260914'
    subprocess.run(['tmux','new-session','-d','-s',session,
                    shlex.join(command)+' >'+shlex.quote(str(log))+' 2>&1'],check=True)
    pane=int(subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_pid}']))
    state=dict(label=label,enabled=enabled,log=str(log),session=session,pane=pane,
               launch=command,created=time.time(),
               head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip())
    source_paths=[
        'python/sglang/srt/environ.py',
        'python/sglang/srt/distributed/device_communicators/dsv4_ar_experiment.py',
        'python/sglang/srt/layers/moe/moe_runner/aiter.py',
        'python/sglang/kernels/ops/moe/gfx90a_fp4_down_consumer_quant_oracle.py',
        'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_down_consumer_quant_oracle.cuh',
    ]
    state['source_sha256']={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in source_paths}
    diff=subprocess.check_output(['git','diff','--',*source_paths],cwd=REPO).decode()
    (ROOT/f'{label}.source.patch').write_text(diff)
    save(label+'.state.json',state)
    for _ in range(120):
        parent=psutil.Process(pane)
        for proc in [parent,*parent.children(recursive=True)]:
            try:
                cmd=proc.cmdline()
                if 'sglang.launch_server' in cmd and '/home/pc/models/modelscope' in cmd:
                    state.update(pid=proc.pid,birth=proc.create_time(),command=cmd)
                    save(label+'.state.json',state)
                    print('STARTED',label,proc.pid,flush=True)
                    return state
            except psutil.NoSuchProcess:
                pass
        time.sleep(.5)
    raise RuntimeError('No server process; inspect '+str(log))


def owned(state):
    proc=psutil.Process(state['pid'])
    assert proc.create_time()==state['birth'] and proc.cmdline()==state['command']
    return proc


def ready(state):
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        proc=owned(state)
        text=Path(state['log']).read_text(errors='replace')
        if 'The server is fired up and ready to roll!' in text:
            break
        if 'Scheduler hit an exception' in text:
            raise RuntimeError('Startup exception; inspect '+state['log'])
        time.sleep(10)
    else:
        raise TimeoutError('Service readiness exceeded 20 min')
    env=proc.environ()
    assert env[FLAG]==str(state['enabled'])
    assert env['SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV']=='1'
    info=get(URL+'/server_info')
    assert info['tp_size']==8 and info['speculative_algorithm'] is None
    assert info['model_path']=='/home/pc/models/modelscope'
    assert info['max_total_tokens']==1048576
    save(state['label']+'.server-info.json',info)
    if state['enabled']:
        assert 'DSV4 native TP8 M32 down-consumer CTA16 selected' in text, 'Flag did not select candidate'
    resources(state['label']+'-ready',proc)
    print('READY',state['label'],flush=True)


def measure(state,names,concurrency=32):
    proc=owned(state)
    france=json.loads((REPO/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=post(URL+'/generate',dict(input_ids=france['input_ids'],
                cache_salt=f'ar-down-{state["label"]}-{time.time_ns()}',
                sampling_params=dict(temperature=0,max_new_tokens=32,ignore_eos=False)),600)
    save(state['label']+'.France.json',answer)
    assert 'paris' in answer['text'].lower()
    assert len(completion_ids(answer))==answer['meta_info']['completion_tokens']
    manifest=ROOT/'inputs.json'
    if not manifest.exists():
        data=json.loads((OLD/'manifests64/decode.json').read_text())
        data['requests']=data['requests'][:32]
        save('inputs.json',data)
    for name,tokens,rounds in [(state['label']+'-warm',256,1),*[(n,2048,2) for n in names]]:
        resources(name,proc)
        output=ROOT/f'{name}.json'
        assert not output.exists()
        command=[sys.executable,str(REPO/'scripts/rocm/bench_dsv4_open_code_decode.py'),
                 '--base-url',URL,'--inputs',str(manifest),'--request-count',str(concurrency),
                 '--rounds',str(rounds),'--seconds','0','--tokens',str(tokens),
                 '--output',str(output)]
        print('MEASURING',name,flush=True)
        with (ROOT/f'{name}.bench.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
        data=json.loads(output.read_text())
        assert data['status']=='complete'
        assert all(q['spec_accept_length'] is None for r in data['rounds'] for w in r['waves'] for q in w['requests'])
        print('RESULT',name,data['median_decode_tok_s'],flush=True)
    text=Path(state['log']).read_text(errors='replace')
    witness=f'raw_bs={concurrency} executed_rows={concurrency} input_rows={concurrency}'
    assert witness in text, f'No exact M{concurrency} execution witness'


def stop(state):
    proc=owned(state)
    children=proc.children(recursive=True)
    proc.send_signal(signal.SIGINT)
    _,alive=psutil.wait_procs([proc,*children],timeout=25)
    remaining=[p.pid for p in alive if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE]
    save(state['label']+'.stop.json',dict(pid=proc.pid,remaining=remaining,time=time.time()))
    assert not remaining, remaining
    resources(state['label']+'-after-stop')
    print('STOPPED',state['label'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('A1','B','A2'),required=True)
    parser.add_argument('--c1-regression',action='store_true',
                        help='After positive C32 ABBA only: C1 scope-isolation point, not a new C1 kernel')
    parser.add_argument('--follow-c1-if-beneficial',action='store_true',
                        help='After C32 A2, analyze then reuse its control service for C1 A1')
    args=parser.parse_args()
    assert not args.follow_c1_if_beneficial or (args.arm=='A2' and not args.c1_regression)
    assert (ROOT/'component.json').exists(), 'Run exact component checks first'
    if args.c1_regression:
        summary=json.loads((ROOT/'summary.json').read_text())
        assert summary['gain_pct']>0, 'User requested C1 only after C32 benefit'
    label=('C1-' if args.c1_regression else '')+args.arm
    state=start(label,int(args.arm=='B'))
    try:
        ready(state)
        names=('B1','B2') if args.arm=='B' else (args.arm,)
        if args.c1_regression:
            names=tuple('C1-'+n for n in names)
        measure(state,names,1 if args.c1_regression else 32)
        if args.follow_c1_if_beneficial:
            subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True,cwd=REPO)
            summary=json.loads((ROOT/'summary.json').read_text())
            if summary['gain_pct']>0:
                c1state=dict(state,label='C1-A1',reuses_control_service=state['label'])
                save('C1-A1.state.json',c1state)
                measure(c1state,('C1-A1',),1)
    finally:
        stop(state)


if __name__=='__main__': main()
