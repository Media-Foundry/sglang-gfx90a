"""Two fresh default-production services; all-layer first-prefill frontier.

No free-running continuation is used for this frontier: all captured rows are
fixed prompt tokens. This diagnoses the first forward, not all cached decode.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import psutil

root=Path(__file__).resolve().parent;repo=root.parents[2]
prior=root.parent/'dsv4_c16_attention_stage1_service_20260915/B'
manifest=json.loads((prior/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
stages=('entry_prev_residual','attn_residual','attn_norm','attn_core','attn_out',
        'ffn_input','ffn_topk_ids','ffn_topk_weights','ffn_routed','ffn_out')
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('all_layer_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
for arm in ('A2','B2'):
    out=root/arm;out.mkdir(exist_ok=False);data=out/'data';data.mkdir()
    life.ROOT=life.OLD=out
    launcher=(prior/'start-ar-matrix.sh').read_text()
    needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
    flags=(f'export SGLANG_DSV4_DEBUG_FIRST_DIV_DIR={data}\n'
           f'export SGLANG_DSV4_DEBUG_FIRST_DIV_LAYERS={",".join(map(str,range(43)))}\n'
           f'export SGLANG_DSV4_DEBUG_FIRST_DIV_STAGES={",".join(stages)}\n'
           'export SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER=0\n'
           'export SGLANG_DSV4_PREFILL_ATTN_STAGE1=1\n'
           'export SGLANG_DSV4_DEBUG_PREFILL_ATTN_STAGE1_CHECK=0\n'
           'unset SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR\n'
           'unset SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR\n'
           'unset SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
           'export SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK=0\n')
    # Isolated copy changes only serve's timeout; production launcher untouched.
    script=(repo/'scripts/rocm_dsv4_flash.sh').read_text()
    serve_line='exec "${server_prefix[@]}" "${PYTHON_BIN}" -m sglang.launch_server "${server_args[@]}"'
    assert script.count(serve_line)==1
    diagnostic_script=out/'diagnostic-launcher.sh'
    diagnostic_script.write_text(script.replace(serve_line,serve_line+' --watchdog-timeout 1800'))
    (out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+f'exec bash {diagnostic_script} serve'))
    life.save('inputs.json',manifest)
    paths=set(json.loads((prior/'plan.json').read_text())['sources'])|{
        str(Path(__file__).relative_to(repo)),
        'python/sglang/kernels/ops/debug/dsv4_first_divergence.py',
        'python/sglang/srt/models/deepseek_v4.py','python/sglang/srt/models/deepseek_v2.py'}
    hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
    life.save('plan.json',dict(scope=__doc__,diagnostic_only=True,layers=list(range(43)),stages=stages,
        sources=hashes,input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
    label='P16-all-layer-'+arm;state=life.start(label,0)
    try:
        life.ready(state)
        env=life.owned(state).environ()
        assert env['SGLANG_DSV4_C4_PREFILL_QUERY_OWNER']=='1'
        assert env['SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER']=='0'
        assert env['SGLANG_DSV4_DEBUG_FIRST_DIV_DIR']==str(data)
        info=json.loads((out/(label+'.server-info.json')).read_text())
        assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==info['max_prefill_tokens']==32768
        assert info['ep_size']==1 and info['speculative_algorithm'] is None
        assert '--watchdog-timeout' in life.owned(state).cmdline()
        command=life.owned(state).cmdline()
        assert command[command.index('--watchdog-timeout')+1]=='1800'
        rids=[f'all-layer-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'all-layer-{arm}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
        life.save('sent.json',payload)
        responses=life.post(life.URL+'/generate',payload,2400);life.save('responses.json',responses)
        byid={r['meta_info']['id']:r for r in responses};assert set(byid)==set(rids)
        for request,rid in zip(manifest['requests'],rids,strict=True):
            r=byid[rid];assert r['prompt_token_ids']==request['input_ids'] and r['meta_info']['cached_tokens']==0
            assert len(life.completion_ids(r))==1
        for rank in range(8):
            for layer in range(43):
                for name in stages:
                    assert (data/f'layer-{layer}-rank-{rank}-{name}.json').exists(),(rank,layer,name)
        assert hashes=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in hashes}
        france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
        answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
            cache_salt=f'all-layer-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
        life.save('France.json',answer);assert 'paris' in answer['text'].lower()
        life.save('complete.json',dict(input_echo_exact=16,diagnostic_only=True,
            records=len(list(data.glob('*.json'))),stages=stages))
        print('CAPTURE COMPLETE',arm,flush=True)
    finally:
        try:life.stop(state)
        except psutil.NoSuchProcess:
            life.save(label+'.stop-race.json',dict(parent_already_gone=True,pid=state['pid']))
            raise
