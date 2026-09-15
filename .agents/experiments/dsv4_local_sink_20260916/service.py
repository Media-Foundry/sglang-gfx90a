"""Corrected-sink native TP8 baseline: real capture, C16 prefill and continuations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'check';out.mkdir(exist_ok=False)
capture=out/'capture';capture.mkdir()
spec=importlib.util.spec_from_file_location('sink_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life);life.ROOT=life.OLD=out
prior=root.parent/'dsv4_h16_service_20260916/A1'
launcher=(prior/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
flags=f'export SGLANG_DSV4_DEBUG_ATTN_PEER_CAPTURE_DIR={capture}\n'
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
manifest=json.loads((prior/'inputs.json').read_text());life.save('inputs.json',manifest)
paths=set(json.loads((prior/'plan.json').read_text())['sources'])|{
    'python/sglang/srt/models/deepseek_v4.py',str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/debug/dsv4_attention_peer_capture.py'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
life.save('plan.json',dict(sources=sources,h16_enabled=False,original_weights=True,kv_tokens=1048576))
tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
label='P16-local-sink';state=life.start(label,0)
try:
    life.ready(state)
    env=life.owned(state).environ();assert not env.get('SGLANG_DSV4_DEBUG_H16_IPC_MANIFEST')
    assert env['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT']=='0'
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],cache_salt=f'sink-france-{time.time_ns()}',
        sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    client=root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
    progress=[]
    for name,rounds in (('capture-warmup',1),('corrected-baseline',3)):
        with (out/(name+'.client.log')).open('w') as log:
            subprocess.run([sys.executable,str(client),'--base-url',life.URL,'--inputs',str(out/'inputs.json'),
                '--request-count','16','--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        data=json.loads((out/(name+'.json')).read_text())
        assert all(r['input_echo_exact'] and r['cached_tokens']==[0]*16 for r in data['rounds'])
        progress.append(dict(leg=name,median=data['median_input_tok_s']))
        life.save('progress.json',progress);print('RESULT',progress[-1],flush=True)
    import torch
    from safetensors import safe_open
    model=Path('/home/pc/models/modelscope');key='layers.20.attn.attn_sink'
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(model/index[key],framework='pt',device='cpu') as f:full=f.get_tensor(key)
    sink_checks=[]
    for rank in range(8):
        meta=json.loads((capture/f'rank-{rank}.json').read_text())
        data=torch.load(capture/f'rank-{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
        actual=data['attn_sink'];expected=full[rank*8:(rank+1)*8]
        assert torch.equal(actual,expected),rank
        sink_checks.append(dict(rank=rank,checkpoint_slice_exact=True,rows=meta['rows'],
            sink_sha256=meta['hashes']['attn_sink'],canonical_output_byte_exact=meta['canonical_output_byte_exact']))
    life.save('sink-checks.json',sink_checks)
    answers=[]
    for rep in range(2):
        rids=[f'sink-{rep}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'sink-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),return_prompt_token_ids=True)
        responses=life.post(life.URL+'/generate',payload,1800);life.save(f'answers-{rep}.json',responses)
        byid={r['meta_info']['id']:r for r in responses};assert set(byid)==set(rids)
        ordered=[byid[rid] for rid in rids]
        for request,response in zip(manifest['requests'],ordered,strict=True):
            ids=life.completion_ids(response)
            assert response['prompt_token_ids']==request['input_ids'] and response['meta_info']['cached_tokens']==0
            assert len(ids)==128 and tokenizer.decode(ids,skip_special_tokens=False)==response['text']
        answers.append(ordered);print('QUALITY COMPLETE',rep,flush=True)
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(progress=progress,france_passed=True,sink_checks=sink_checks,
        quality_repeat_exact=sum(life.completion_ids(a)==life.completion_ids(b) for a,b in zip(*answers,strict=True)),
        input_echo_exact=32,original_weights=True,kv_tokens=1048576))
finally:
    life.stop(state)
