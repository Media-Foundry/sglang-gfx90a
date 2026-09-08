#!/usr/bin/env python3
"""Concurrent fixed-prefix probe, NOT a cached-decode or throughput benchmark.

Six established prefixes repeated across32 clients. Request concurrency does
not guarantee a single GPU batch. Compare next ID and supplied-prefix logprobs
against sequential reference and across repeated concurrent waves.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:30011')
    p.add_argument('--rounds',type=int,default=3)
    args=p.parse_args()
    refs=json.loads(args.reference.read_text())['teacher_forced']
    assert len(refs)==6
    result={'status':'running','kind':'concurrent-prefill-fixed-prefix','rounds':[]}
    def save():
        tmp=args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    for rep in range(args.rounds):
        barrier=threading.Barrier(32)
        def request(i):
            source=refs[i%6]
            payload={'input_ids':source['input_ids'],
                     'sampling_params':{'temperature':0,'max_new_tokens':1},
                     'cache_salt':'fixed-concurrent-'+uuid.uuid4().hex,
                     'return_logprob':True,'top_logprobs_num':20,
                     'logprob_start_len':len(source['input_ids'])-source['continuation_length']}
            req=urllib.request.Request(args.base_url+'/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            barrier.wait(timeout=30)
            with opener.open(req,timeout=180) as response:r=json.load(response)
            meta=r['meta_info']
            assert meta['cached_tokens']==0 and meta['completion_tokens']==len(r['output_ids'])==1
            assert all(v is None or math.isfinite(v) for v,*_ in meta['input_token_logprobs'])
            row={'index':i,'probe':i%6,'output_ids':r['output_ids'],
                 'input_token_logprobs':meta['input_token_logprobs'],
                 'output_top_logprobs':meta['output_top_logprobs']}
            row['reference_exact']={k:row[k]==source[k] for k in ('output_ids','input_token_logprobs','output_top_logprobs')}
            return row
        save()
        with ThreadPoolExecutor(max_workers=32) as pool:rows=list(pool.map(request,range(32)))
        result['rounds'].append(rows);save()
        print(rep,{k:sum(r['reference_exact'][k] for r in rows) for k in rows[0]['reference_exact']},flush=True)
    result['cross_round_exact']={k:sum(all(rows[i][k]==result['rounds'][0][i][k] for rows in result['rounds'][1:]) for i in range(32)) for k in ('output_ids','input_token_logprobs','output_top_logprobs')}
    result['status']='complete';save()
    print('cross_round_exact',result['cross_round_exact'],flush=True)


if __name__=='__main__':main()
