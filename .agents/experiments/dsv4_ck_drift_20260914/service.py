"""Owned TP8 atomic/fixed-slot service: controlled admission and normal HTTP waves."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('atomic','fixed'),required=True)
    parser.add_argument('--matched-only',action='store_true')
    args=parser.parse_args()
    run_name=args.mode+('-matched' if args.matched_only else '')
    out=ROOT/run_name
    out.mkdir(exist_ok=True)
    helper=REPO/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
    spec=importlib.util.spec_from_file_location('owned_lifecycle',helper)
    life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
    life.ROOT=life.OLD=out
    old=REPO/'.agents/experiments/dsv4_tp8_c16_prefill_20260914/B'
    launch=(old/'start-ar-matrix.sh').read_text()
    value='1' if args.mode=='fixed' else '0'
    launch=launch.replace('exec bash scripts/rocm_dsv4_flash.sh serve',
        f'export SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT={value}\nexec bash scripts/rocm_dsv4_flash.sh serve')
    assert not (out/'start-ar-matrix.sh').exists()
    (out/'start-ar-matrix.sh').write_text(launch)
    inputs=json.loads((old/'inputs.json').read_text())
    life.save('inputs.json',inputs)
    label='CK-DRIFT-'+run_name
    state=life.start(label,0)
    try:
        life.ready(state)
        assert life.owned(state).environ()['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT']==value
        france=json.loads((REPO/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
        res=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
            cache_salt=f'ck-drift-france-{time.time_ns()}',
            sampling_params=dict(temperature=0,max_new_tokens=32)),600)
        life.save('France.json',res)
        assert 'paris' in res['text'].lower()
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
        records=[]
        # One batch-shaped request removes independent HTTP arrival permutation;
        # logs still determine the actual scheduler batches, not this intent.
        for wave,tokens in enumerate((1,128,128,128)):
            life.resources(f'batch-{wave}-before',life.owned(state))
            start=time.monotonic()
            res=life.post(life.URL+'/generate',dict(
                input_ids=[r['input_ids'] for r in inputs['requests']],
                cache_salt=f'ck-drift-{args.mode}-{wave}-{time.time_ns()}',
                sampling_params=dict(temperature=0,max_new_tokens=tokens,ignore_eos=True),
                return_logprob=True,logprob_start_len=-1,top_logprobs_num=5),1200)
            life.save(f'batch-{wave}.json',res)
            assert isinstance(res,list) and len(res)==16
            for row in res:
                ids=life.completion_ids(row)
                assert len(ids)==tokens and row['meta_info']['cached_tokens']==0
                assert tokenizer.decode(ids,skip_special_tokens=False)==row['text']
            records.append(dict(wave=wave,tokens=tokens,wall_s=time.monotonic()-start))
            life.save('batch-progress.json',records)
            print('BATCH',args.mode,records[-1],flush=True)
        if args.matched_only:
            for wave in range(2):
                life.resources(f'split-lp-{wave}-before',life.owned(state))
                barrier=Barrier(16)
                salt=f'ck-matched-{time.time_ns()}'
                def request(index):
                    barrier.wait(timeout=30)
                    return life.post(life.URL+'/generate',dict(
                        input_ids=inputs['requests'][index]['input_ids'],
                        cache_salt=salt+f'-{index}',
                        sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),
                        return_logprob=True,logprob_start_len=-1,top_logprobs_num=5),1200)
                with ThreadPoolExecutor(max_workers=16) as pool:
                    res=list(pool.map(request,range(16)))
                life.save(f'split-lp-{wave}.json',res)
                for row in res:
                    ids=life.completion_ids(row)
                    assert len(ids)==128 and row['meta_info']['cached_tokens']==0
                    assert tokenizer.decode(ids,skip_special_tokens=False)==row['text']
                print('MATCHED SPLIT',run_name,wave,flush=True)
            life.save('complete.json',dict(mode=args.mode,matched_only=True,batch_waves=records))
            return
        for name,tokens,rounds in [('prefill',1,3),('concurrent-quality',128,2)]:
            life.resources(name+'-before',life.owned(state))
            command=[sys.executable,str(REPO/'scripts/rocm/bench_dsv4_prefill_diverse_concurrent.py'),
                '--base-url',life.URL,'--inputs',str(out/'inputs.json'),
                '--request-count','16','--rounds',str(rounds),'--tokens',str(tokens),
                '--output',str(out/f'{name}.json')]
            with (out/f'{name}.client.log').open('w') as log:
                subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            result=json.loads((out/f'{name}.json').read_text())
            for row in result['rounds']:
                assert row['cached_tokens']==[0]*16 and row['completion_lengths']==[tokens]*16
                for ids,text in zip(row['completion_ids'],row['texts'],strict=True):
                    assert tokenizer.decode(ids,skip_special_tokens=False)==text
            print('RESULT',args.mode,name,result['median_input_tok_s'],flush=True)
        if value=='1':
            assert 'CK fixed-slot FP32 reduction selected' in Path(state['log']).read_text()
        life.save('complete.json',dict(mode=args.mode,batch_waves=records))
    finally:
        life.stop(state)


if __name__=='__main__':main()
