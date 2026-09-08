#!/usr/bin/env python3
"""Repeated France sentinel for C32 correctness only, never a throughput workload.

Force length to exercise the resident graph; only the nine-token answer ending
at EOS is a semantic oracle. Post-EOS tokens are not judged as natural text.
"""
import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

from bench_dsv4_tp4_diverse_concurrent import FRANCE_EXPECTED, decode_moments, get_url, post_stream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:30011')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    item=json.loads((root/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    assert item['prompt'].strip()=='What is the capital of France?'
    before=decode_moments(get_url(a.base_url+'/v1/loads?include=core',60))
    barrier=threading.Barrier(33);nonce=time.time_ns()
    def run(i):
        payload={'input_ids':item['input_ids'],'sampling_params':{
            'temperature':0,'max_new_tokens':256,'ignore_eos':True,'stream_interval':1},
            'cache_salt':f'france-check-{nonce}-{i}','stream':True}
        barrier.wait()
        result,samples=post_stream(a.base_url+'/generate',payload,180)
        return {'result':result,'samples':samples}
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures=[pool.submit(run,i) for i in range(32)]
        barrier.wait()
        rows=[f.result() for f in futures]
    after=decode_moments(get_url(a.base_url+'/v1/loads?include=core',60))
    delta=[b-a for a,b in zip(before,after)] if before and after else None
    exact=[r['result']['output_ids'][:9]==FRANCE_EXPECTED for r in rows]
    # Intersection after first token and before final token proves co-residency.
    start=max(next(t for t,n in r['samples'] if n>=1) for r in rows)
    end=min(r['samples'][-1][0] for r in rows)
    report={'exact_count':sum(exact),'request_count':32,'resident_window_s':end-start,
            'decode_moments_delta':delta,'rows':rows,'purpose':'correctness-only repeated sentinel'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='rows'})
    assert all(exact) and end>start
    assert all(len(r['result']['output_ids'])==256 for r in rows)


if __name__=='__main__':main()
