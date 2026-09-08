#!/usr/bin/env python3
"""Repeated real-prompt decode diagnostic, never a diverse-throughput benchmark."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

from bench_dsv4_tp4_diverse_concurrent import completion_ids, post_stream


def digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--index',type=int,default=17)
    p.add_argument('--base-url',default='http://127.0.0.1:30011')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--clients',type=int,default=32)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix-result',type=Path,
                   help='Teacher-force a prefix from a previous result of this script')
    p.add_argument('--prefix-length',type=int,default=0)
    a=p.parse_args()
    assert a.rounds>0 and a.tokens>0 and a.clients>0
    source=json.loads(a.inputs.read_text())['requests'][a.index]
    if a.prefix_result:
        previous=json.loads(a.prefix_result.read_text())
        assert previous['source']['input_ids']==source['input_ids']
        continuation=previous['rounds'][0]['output_ids'][0]
        assert 0<=a.prefix_length<=len(continuation)
        source=dict(source,input_ids=source['input_ids']+continuation[:a.prefix_length])
    else:
        assert a.prefix_length==0
    out={'kind':'homogeneous-real-prompt-decode-correctness-only',
         'source':source,'tokens':a.tokens,'clients':a.clients,'rounds':[]}
    for rep in range(a.rounds):
        barrier=threading.Barrier(a.clients)
        nonce=time.time_ns()
        def run(i):
            payload={'input_ids':source['input_ids'],
                'sampling_params':{'temperature':0,'max_new_tokens':a.tokens,
                                   'ignore_eos':True,'stream_interval':1},
                'cache_salt':f'same-code-{nonce}-{i}','stream':True}
            barrier.wait()
            response,_=post_stream(a.base_url.rstrip('/')+'/generate',payload,1200,False)
            ids=completion_ids(response)
            assert len(ids)==a.tokens,(i,len(ids))
            assert response['meta_info']['finish_reason']['type']=='length'
            return ids
        with ThreadPoolExecutor(max_workers=a.clients) as pool:
            rows=list(pool.map(run,range(a.clients)))
        hashes=[digest(x) for x in rows]
        first=[next((i for i,(x,y) in enumerate(zip(rows[0],r)) if x!=y),None)
               for r in rows]
        record={'output_ids':rows,'hashes':hashes,'multiplicities':dict(Counter(hashes)),
                'first_divergence_from_client0':first}
        out['rounds'].append(record)
        a.output.write_text(json.dumps(out,indent=2)+'\n')
        print(rep,'unique',len(set(hashes)),'first_divergence',first,flush=True)
    out['multiset_exact']=all(Counter(r['hashes'])==Counter(out['rounds'][0]['hashes'])
                              for r in out['rounds'])
    out['cross_round_exact_clients']=sum(
        all(r['output_ids'][i]==out['rounds'][0]['output_ids'][i] for r in out['rounds'])
        for i in range(a.clients))
    out['status']='complete'
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print('multiset_exact',out['multiset_exact'],
          'cross_round_exact_clients',out['cross_round_exact_clients'],flush=True)


if __name__=='__main__':
    main()
