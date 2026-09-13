"""Wait for owned original V4 AR service, correctness sentinel, few-size pilot."""
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path('/home/pc/Code/sglang')
OUT = ROOT / '.agents/experiments/dsv4_tp8_revalidation_20260913'
sys.path.insert(0, str(ROOT / 'scripts/rocm'))
from bench_dsv4_tp8_mhc_fusion_drift_trial import post, get

service = psutil.Process(int(sys.argv[1]))
birth = service.create_time()
state = {'status': 'waiting_ready', 'pid': service.pid, 'command': service.cmdline()}
def save():
    (OUT / 'ar-pilot-state.json').write_text(json.dumps(state, indent=2)+'\n')
def owned():
    if not service.is_running() or service.create_time() != birth:
        raise RuntimeError('Owned service exited or PID reused')
def resources(name):
    owned()
    gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    owners = {service.pid, *(p.pid for p in service.children(recursive=True))}
    active = {p['process_info']['pid'] for g in gpu for p in g.get('process_list', [])
              if isinstance(p.get('process_info'), dict)}
    if not active <= owners:
        raise RuntimeError(f'External GPU users: {active - owners}')
    (OUT / (name+'.gpu-before.json')).write_text(json.dumps(gpu,indent=2)+'\n')
try:
    save()
    deadline=time.monotonic()+1800
    while 'The server is fired up and ready to roll!' not in (OUT / 'ar-matrix.service.log').read_text():
        owned()
        if time.monotonic()>deadline:
            raise TimeoutError('Startup exceeded30min; inspect before restarting')
        time.sleep(5)
    info=get('http://127.0.0.1:30021/server_info')
    state['server_info']=info
    assert info['tp_size'] == 8 and info['speculative_algorithm'] is None
    assert info['max_total_tokens'] == 1048576
    env=service.environ()
    state['profile']={k:v for k,v in env.items() if k.startswith(('SGLANG_DSV4_', 'CUDA_GRAPH_', 'AITER_GFX90A_'))}
    resources('ar-france')
    item=json.loads((ROOT/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    response=post('http://127.0.0.1:30021/generate', {
        'input_ids': item['input_ids'], 'cache_salt':'v4-native-france-20260913',
        'sampling_params': {'temperature':0,'max_new_tokens':32,'ignore_eos':False}},600)
    (OUT/'ar-france.json').write_text(json.dumps(response,indent=2)+'\n')
    assert 'paris' in response.get('text','').lower() and response.get('output_ids'), response
    assert response['meta_info'].get('spec_accept_length') is None
    print('France:',response['text'],flush=True)
    for c in (1,8,32):
        for kind, rounds, tokens in (('warmup',1,128),('measured',2,512)):
            name=f'ar-pilot-c{c}-{kind}'
            resources(name)
            state.update(status='running',current=name)
            save()
            cmd=[sys.executable,str(ROOT/'scripts/rocm/bench_dsv4_open_code_decode.py'),
                 '--base-url','http://127.0.0.1:30021','--inputs',str(OUT/'legacy-manifests/decode.json'),
                 '--request-count',str(c),'--rounds',str(rounds),'--seconds','0','--tokens',str(tokens),
                 '--output',str(OUT/(name+'.json'))]
            print('START',name,flush=True)
            with (OUT/(name+'.client.log')).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((OUT/(name+'.json')).read_text())
            assert all(r['spec_accept_length'] is None for rnd in result['rounds'] for w in rnd['waves'] for r in w['requests'])
            print('DONE',name,result['median_decode_tok_s'],flush=True)
    state.update(status='complete',current=None)
except BaseException as exc:
    state.update(status='failed',error=repr(exc))
    raise
finally:
    save()
