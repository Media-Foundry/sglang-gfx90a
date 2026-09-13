"""C1-only wo_a A/B: require complete main matrix and owned native service."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--woa-gemv', type=int, choices=(0, 1), required=True)
    parser.add_argument('--outputs', nargs='+', required=True)
    args = parser.parse_args()
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in [args.label, *args.outputs])
    root = Path(__file__).resolve().parent
    repo = Path('/home/pc/Code/sglang')
    assert json.loads((root/'ar-final-decode/state.json').read_text())['status'] == 'complete'
    state = json.loads((root/f'empty-tiles-{args.label}-service.json').read_text())
    proc = psutil.Process(state['pid'])
    assert proc.create_time() == state['birth'] and proc.cmdline() == state['command']
    env = proc.environ()
    assert env.get('SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP') == '1'
    assert env.get('SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV', '0') == str(args.woa_gemv)
    assert env.get('SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR') == '1'
    assert env.get('SGLANG_DSV4_GFX90A_WAVE64_GROUPED_GEMV') == '1'
    assert 'The server is fired up and ready to roll!' in Path(state['log']).read_text()
    sys.path.insert(0, str(repo/'scripts/rocm'))
    from bench_dsv4_tp8_mhc_fusion_drift_trial import get, post
    from bench_dsv4_tp4_diverse_concurrent import completion_ids

    def resources(name):
        assert proc.is_running() and proc.create_time() == state['birth']
        owned = {proc.pid, *[p.pid for p in proc.children(recursive=True)]}
        gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
        active = {p['process_info']['pid'] for g in gpu for p in g.get('process_list', [])
                  if isinstance(p.get('process_info'), dict)}
        assert active <= owned, active-owned
        (root/f'woa-{name}.gpu-before.json').write_text(json.dumps(gpu, indent=2)+'\n')

    resources(args.label+'-ready')
    url = 'http://127.0.0.1:30021'
    info = get(url+'/server_info')
    assert info['tp_size'] == 8 and info['speculative_algorithm'] is None
    assert info['model_path'] == '/home/pc/models/modelscope'
    assert info['max_total_tokens'] == 1048576
    france = json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer = post(url+'/generate', dict(input_ids=france['input_ids'],
                  cache_salt=f'woa-{args.label}-France-{time.time_ns()}',
                  sampling_params=dict(temperature=0,max_new_tokens=32,ignore_eos=False)), 600)
    assert 'paris' in answer['text'].lower()
    assert len(completion_ids(answer)) == answer['meta_info']['completion_tokens']
    assert answer['meta_info'].get('spec_accept_length') is None
    output = root/f'woa-{args.label}-France.json'
    assert not output.exists()
    output.write_text(json.dumps(answer,indent=2)+'\n')
    print('France', repr(answer['text']), flush=True)
    for name, tokens, seconds in [(args.label+'-warm',256,0), *[(s,2048,30) for s in args.outputs]]:
        resources(name)
        output = root/f'woa-{name}.json'
        assert not output.exists()
        command = [sys.executable, str(repo/'scripts/rocm/bench_dsv4_open_code_decode.py'),
                   '--base-url',url,'--inputs',str(root/'manifests64/decode.json'),
                   '--request-count','1','--rounds','1','--seconds',str(seconds),
                   '--tokens',str(tokens),'--output',str(output)]
        print('START', name, flush=True)
        with (root/f'woa-{name}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=json.loads(output.read_text())
        assert data['status']=='complete' and not data['ignore_eos']
        assert all(q['spec_accept_length'] is None for r in data['rounds'] for w in r['waves'] for q in w['requests'])
        print('DONE',name,data['median_decode_tok_s'],flush=True)


if __name__=='__main__':
    main()
