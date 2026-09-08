#!/usr/bin/env python3
"""EOS-respecting code decode: accumulate full-concurrency decode-only windows.

Each synchronized wave excludes all prefill (start at latest first token) and
all batch drain (end at earliest last token). Multiple waves supply at least
30 seconds of measured common decode windows per round. Not GPU-only timing.
"""
import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

from bench_dsv4_tp4_diverse_concurrent import post_stream, completion_ids


def count_at(samples, timestamp):
    return next((n for t, n in reversed(samples) if t <= timestamp), 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--request-count', type=int, choices=(1,2,4,8,16,32), required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--tokens', type=int, default=2048)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    manifest = json.loads(args.inputs.read_text())
    requests = manifest['requests']
    assert len(requests) >= args.request_count
    result = dict(format='dsv4-open-code-natural-decode-v1', status='running',
                  concurrency=args.request_count, ignore_eos=False, rounds=[],
                  manifest_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                  requested_decode_seconds_per_round=args.seconds)
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    for rep in range(args.rounds):
        record = dict(round=rep, waves=[], decode_seconds=0., decode_tokens=0)
        result['rounds'].append(record)
        while not record['waves'] or record['decode_seconds'] < args.seconds:
            wave = len(record['waves'])
            assert wave < 64, 'Insufficient common resident decode; inspect wave data'
            chosen = [requests[(wave*args.request_count+i)%len(requests)] for i in range(args.request_count)]
            barrier = threading.Barrier(args.request_count+1)
            nonce = time.time_ns()
            def run(i, item):
                payload = dict(input_ids=item['input_ids'], stream=True,
                               cache_salt=f'open-code-{rep}-{wave}-{i}-{nonce}',
                               sampling_params=dict(temperature=0, max_new_tokens=args.tokens,
                                                    ignore_eos=False, stream_interval=1))
                barrier.wait()
                begin = time.perf_counter()
                response, samples = post_stream(args.base_url+'/generate', payload, 1200)
                return begin, response, samples
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.request_count) as pool:
                fs = [pool.submit(run,i,item) for i,item in enumerate(chosen)]
                barrier.wait()
                outputs = [f.result() for f in fs]
            start = max(s[0][0] for _,_,s in outputs)
            end = min(s[-1][0] for _,_,s in outputs)
            tokens = sum(count_at(s,end)-count_at(s,start) for _,_,s in outputs) if end>start else 0
            wall = max(0.,end-start)
            witnesses=[]
            for item,(begin,response,samples) in zip(chosen,outputs):
                ids=completion_ids(response)
                meta=response['meta_info']
                assert ids and meta['finish_reason']['type'] in ('stop','length')
                assert len(ids)==meta['completion_tokens']
                witnesses.append(dict(index=item['index'], output_ids=ids, text=response.get('text'),
                                      finish_reason=meta['finish_reason'], ttft=samples[0][0]-begin,
                                      spec_accept_length=meta.get('spec_accept_length'),
                                      sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
            record['waves'].append(dict(wave=wave, resident_seconds=wall, resident_tokens=tokens,
                                         resident_tok_s=tokens/wall if wall else None,
                                         common_start=start, common_end=end, requests=witnesses))
            record['decode_seconds'] += wall
            record['decode_tokens'] += tokens
            save()
            print(rep,wave,'resident',round(wall,3),'seconds',round(tokens/wall,2) if wall else None,flush=True)
        record['decode_tok_s']=record['decode_tokens']/record['decode_seconds']
    result.update(status='complete',median_decode_tok_s=statistics.median(x['decode_tok_s'] for x in result['rounds']))
    save()
    print('COMPLETE',result['median_decode_tok_s'],flush=True)


if __name__=='__main__':
    main()
