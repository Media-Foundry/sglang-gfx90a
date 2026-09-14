"""Echo exact input IDs and sample layer0 across controlled batch composition changes."""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]


def digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--short',action='store_true')
    parser.add_argument('--fp32-attn-ar',action='store_true')
    parser.add_argument('--stable-woa',action='store_true')
    parser.add_argument('--fp32-ffn-ar',action='store_true')
    parser.add_argument('--prepare-dump',action='store_true')
    parser.add_argument('--stable-qkv',action='store_true')
    parser.add_argument('--stable-wqb',action='store_true')
    parser.add_argument('--stable-wob',action='store_true')
    parser.add_argument('--stable-shared',action='store_true')
    parser.add_argument('--stage-layer',type=int,default=0)
    parser.add_argument('--skip-weight-dumps',action='store_true')
    parser.add_argument('--sample-positions',default='0,511,2047,4095,8191')
    parser.add_argument('--baseline-dir',type=Path)
    args=parser.parse_args()
    assert all(int(v)>=0 for v in args.sample_positions.split(','))
    if args.run_name:
        assert '/' not in args.run_name and args.run_name not in ('.','..')
        ROOT=ROOT/args.run_name
        ROOT.mkdir(exist_ok=False)
    helper=REPO/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
    spec=importlib.util.spec_from_file_location('life',helper)
    life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
    life.ROOT=life.OLD=ROOT
    old=args.baseline_dir or REPO/'.agents/experiments/dsv4_tp8_c16_prefill_20260914/B'
    manifest=json.loads((old/'inputs.json').read_text())
    cases=manifest['requests']
    assert len(cases)==16 and len({digest(r['input_ids']) for r in cases})==16
    life.save('inputs.json',manifest)
    assert not (ROOT/'start-ar-matrix.sh').exists()
    trace=ROOT/'trace-current'
    flags=(f'export SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR={trace}\n'
           f'export SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR={trace}\n'
           f'export SGLANG_DSV4_DEBUG_STAGE_LAYER={args.stage_layer} SGLANG_DSV4_DEBUG_STAGE_RANK={args.rank}\n'
           'export SGLANG_DSV4_DEBUG_STAGE_ROWS=-1 SGLANG_DSV4_DEBUG_STAGE_POSITION=-1\n'
           'export SGLANG_DSV4_DEBUG_STAGE_PREFILL_ONLY=1\n'
           f'export SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS={args.sample_positions}\n'
           'export SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT=0\n')
    flags += f'export SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32={int(args.fp32_attn_ar)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_PREFILL_WOA_STABLE={int(args.stable_woa)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_PREFILL_FFN_AR_FP32={int(args.fp32_ffn_ar)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_PREPARE_DUMP={int(args.prepare_dump)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_PREFILL_QKV_STABLE={int(args.stable_qkv)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_PREFILL_WQB_STABLE={int(args.stable_wqb)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_PREFILL_WOB_STABLE={int(args.stable_wob)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_PREFILL_SHARED_STABLE={int(args.stable_shared)}\n'
    flags += f'export SGLANG_DSV4_DEBUG_STAGE_SKIP_WEIGHTS={int(args.skip_weight_dumps)}\n'
    launch=(old/'start-ar-matrix.sh').read_text().replace(
        'exec bash scripts/rocm_dsv4_flash.sh serve',flags+'exec bash scripts/rocm_dsv4_flash.sh serve')
    (ROOT/'start-ar-matrix.sh').write_text(launch)
    state=life.start('INPUT-IDENTITY',0)
    try:
        life.ready(state)
        if args.stable_shared:
            env=life.owned(state).environ()
            assert env.get('SGLANG_DSV4_DEBUG_PREFILL_SHARED_STABLE')=='1'
            paths=('python/sglang/srt/models/deepseek_v2.py',
                   'python/sglang/kernels/ops/debug/dsv4_prefill_shared.py')
            life.save('shared-runtime-contract.json',dict(
                flag=env['SGLANG_DSV4_DEBUG_PREFILL_SHARED_STABLE'],
                pythonpath=env.get('PYTHONPATH'),
                sources={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths}))
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
        canonical=list(range(16));changed=canonical.copy()
        changed[2],changed[14]=changed[14],changed[2]
        jobs=[('warmup',canonical,1),('A1',canonical,128),('B1',changed,128),
              ('B2',changed,128),('A2',canonical,128)]
        if args.short: jobs=[j for j in jobs if j[0]!='B2']
        evidence=[]
        for name,order,tokens in jobs:
            life.resources(name+'-before',life.owned(state))
            rids={i:f'identity-{name}-{i}-{digest(cases[i]["input_ids"])[:12]}' for i in order}
            salt=f'identity-{name}-{time.time_ns()}'
            payload=dict(input_ids=[cases[i]['input_ids'] for i in order],
                rid=[rids[i] for i in order],cache_salt=[salt+f'-{i}' for i in order],
                sampling_params=dict(temperature=0,max_new_tokens=tokens,ignore_eos=True),
                return_prompt_token_ids=True,return_logprob=True,
                logprob_start_len=-1,top_logprobs_num=5)
            life.save(name+'-sent.json',payload)
            response=life.post(life.URL+'/generate',payload,1800)
            life.save(name+'-response.json',response)
            assert len(response)==16
            by_rid={r['meta_info']['id']:r for r in response}
            assert set(by_rid)==set(rids.values())
            normalized=[]
            for i in canonical:
                row=by_rid[rids[i]]
                assert row['prompt_token_ids']==cases[i]['input_ids'], (name,i,'input changed')
                assert row['meta_info']['prompt_tokens']==len(cases[i]['input_ids'])
                assert row['meta_info']['cached_tokens']==0
                ids=life.completion_ids(row)
                assert len(ids)==tokens and tokenizer.decode(ids,skip_special_tokens=False)==row['text']
                normalized.append(row)
            life.save(name+'-by-case.json',normalized)
            assert trace.is_dir()
            trace.rename(ROOT/('trace-'+name))
            evidence.append(dict(wave=name,order=order,echo_exact=16,
                hashes=[digest(r['prompt_token_ids']) for r in normalized]))
            life.save('identity.json',evidence)
            print('INPUT ECHO OK',name,16,flush=True)
            if name=='warmup' and args.stable_shared:
                assert 'DSV4 diagnostic stable shared selected' in Path(state['log']).read_text(), 'Shared selector did not execute; inspect scope log before further requests'
        life.save('complete.json',dict(waves=[j[0] for j in jobs],diagnostic_only=True))
    finally:
        life.stop(state)


if __name__=='__main__':main()
