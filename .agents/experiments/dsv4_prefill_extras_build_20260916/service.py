"""Original validated binaries versus independently rebuilt packaged binaries; arithmetic unchanged."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
repo = root.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm', choices=('check', 'A1', 'B', 'A2'), required=True)
args = parser.parse_args()
diagnostic = args.arm == 'check'
candidate = args.arm in ('check', 'B')
smoke = json.loads((root/'peer-smoke.json').read_text())
assert smoke['status'] == 'complete'
assert len(smoke['per_rank']) == 8
assert all(len(checks) == 10 and all(c['byte_exact'] for c in checks) for checks in smoke['per_rank'])
if not diagnostic:
    assert json.loads((root/'check/complete.json').read_text())['all_layers_exact']
out = root/args.arm
out.mkdir(exist_ok=False)
helper = repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec = importlib.util.spec_from_file_location('h16_life', helper)
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)
sys.path.insert(0, str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life)
life.ROOT = life.OLD = out
prior = root.parent/'dsv4_ck_unique_service_20260916/A2'
manifest = json.loads((prior/'inputs.json').read_text())
assert len(manifest['requests']) == 16
assert sum(len(r['input_ids']) for r in manifest['requests']) == 131069
life.save('inputs.json', manifest)
launcher = (prior/'start-ar-matrix.sh').read_text()
needle = 'exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle) == 1
unique_manifest = (root/'build-v1/unique/manifest.json' if candidate else
    root.parent/'dsv4_ck_unique_store_20260916/build-5fe5f1d3ff25/manifest.json')
ipc_manifest = (root/'build-v1/ipc/manifest.json' if candidate else
    root.parent/'dsv4_h16_service_20260916/ipc-manifest.json')
flags = (f'export SGLANG_DSV4_DEBUG_CK_UNIQUE_MANIFEST={unique_manifest}\n'
         'export SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT=1\n'
         'unset SGLANG_DSV4_DEBUG_ATTN_PEER_CAPTURE_DIR\n'
         f'export SGLANG_DSV4_DEBUG_H16_CHECK={int(diagnostic)}\n')
flags += f'export SGLANG_DSV4_DEBUG_H16_IPC_MANIFEST={ipc_manifest}\n'
execute = needle
if diagnostic:
    script = (repo/'scripts/rocm_dsv4_flash.sh').read_text()
    serve = 'exec "${server_prefix[@]}" "${PYTHON_BIN}" -m sglang.launch_server "${server_args[@]}"'
    assert script.count(serve) == 1
    (out/'diagnostic-launcher.sh').write_text(script.replace(serve, serve+' --watchdog-timeout 1800'))
    execute = f'exec bash {out}/diagnostic-launcher.sh serve'
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle, flags+execute))
paths = set(json.loads((prior/'plan.json').read_text())['sources']) | {
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/debug/dsv4_h16_peer.py',
    'python/sglang/kernels/ops/debug/dsv4_h16_peer_kernel.py',
    'python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py',
    str((root.parent/'dsv4_h16_service_20260916/ipc.cu').relative_to(repo)),
    str((root.parent/'dsv4_h16_service_20260916/ipc-manifest.json').relative_to(repo)),
    'python/sglang/srt/models/deepseek_v4.py',
    'scripts/rocm/build_dsv4_prefill_extras.py',
    'scripts/rocm/dsv4_prefill_extras/build-contract.json',
    'scripts/rocm/dsv4_prefill_extras/hip_ipc.cu',
    'scripts/rocm/dsv4_prefill_extras/unique_stage2.cu',
}
sources = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json', dict(sources=sources, candidate=candidate, diagnostic=diagnostic,
    kv_tokens=1048576, prefill_budget=32768, original_weights=True,
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
label = 'P16-rebuilt-extras-'+args.arm
state = life.start(label, 0)
try:
    life.ready(state)
    info = json.loads((out/(label+'.server-info.json')).read_text())
    assert info['chunked_prefill_size'] == info['max_prefill_tokens'] == 32768
    assert info['max_total_tokens'] == 1048576 and info['ep_size'] == 1
    env = life.owned(state).environ()
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER'] == '0'
    assert env['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT'] == '1'
    assert env['SGLANG_DSV4_DEBUG_H16_IPC_MANIFEST'] == str(ipc_manifest)
    assert env['SGLANG_DSV4_DEBUG_H16_CHECK'] == str(int(diagnostic))
    france = json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer = life.post(life.URL+'/generate', dict(input_ids=france['input_ids'],
        cache_salt=f'h16-france-{time.time_ns()}', sampling_params=dict(temperature=0,max_new_tokens=32)), 600)
    life.save('France.json', answer)
    assert 'paris' in answer['text'].lower()
    assert 'H16 peer selected:' not in Path(state['log']).read_text()
    client = root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
    legs = [('check', 1)] if diagnostic else [('warmup',1), *[(n,3) for n in (('B1','B2') if candidate else (args.arm,))]]
    progress = []
    for name, rounds in legs:
        assert sources == {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
        life.resources(name+'-before', life.owned(state))
        command = [sys.executable,str(client),'--base-url',life.URL,'--inputs',str(out/'inputs.json'),
            '--request-count','16','--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        data = json.loads((out/(name+'.json')).read_text())
        assert all(r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16 for r in data['rounds'])
        progress.append(dict(leg=name,median=data['median_input_tok_s'],
            rates=[r['aggregate_input_tok_s'] for r in data['rounds']]))
        life.save('progress.json',progress)
        print('RESULT',progress[-1],flush=True)

    answers = []
    if not diagnostic:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('/home/pc/models/modelscope', local_files_only=True)
        for rep in range(4):
            rids = [f'corrected-{args.arm}-{rep}-{i}' for i in range(16)]
            payload = dict(input_ids=[r['input_ids'] for r in manifest['requests']], rid=rids,
                cache_salt=[f'corrected-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
                sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),
                return_prompt_token_ids=True)
            responses = life.post(life.URL+'/generate',payload,1800)
            life.save(f'quality-{rep}.json',responses)
            byid = {r['meta_info']['id']:r for r in responses}
            assert set(byid) == set(rids)
            ordered = [byid[rid] for rid in rids]
            for request,response in zip(manifest['requests'],ordered,strict=True):
                ids = life.completion_ids(response)
                assert response['prompt_token_ids'] == request['input_ids']
                assert response['meta_info']['cached_tokens'] == 0 and len(ids) == 128
                assert tokenizer.decode(ids,skip_special_tokens=False) == response['text']
            answers.append(ordered)
            print('QUALITY COMPLETE',args.arm,rep,flush=True)

    logs = Path(state['log']).read_text()
    for rank in range(8): assert f'unique-slot CK selected: rank={rank} ' in logs
    exact = re.findall(r'\[TP(\d+)\] H16 peer output exact: layer=(\d+) rows=(\d+)',logs)
    if diagnostic:
        config = json.loads(Path('/home/pc/models/modelscope/config.json').read_text())
        layers = [l for l,ratio in enumerate(config['compress_ratios'][:43]) if ratio in (4,128)]
        assert {(int(rank),int(layer)) for rank,layer,_ in exact} == {(r,l) for r in range(8) for l in layers}
        assert all(int(m)>=16384 for _,_,m in exact)
    for rank in range(8): assert f'[TP{rank}] H16 peer selected:' in logs
    assert sources == {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(progress=progress,all_layers_exact=diagnostic,
        exact_comparisons=len(exact),diagnostic=diagnostic,france_passed=True,
        unique_ck=True,quality_repeat_exact=[sum(life.completion_ids(a)==life.completion_ids(b)
            for a,b in zip(answers[0],wave,strict=True)) for wave in answers[1:]]))
    print('ARM COMPLETE',args.arm,flush=True)
finally:
    life.stop(state)
