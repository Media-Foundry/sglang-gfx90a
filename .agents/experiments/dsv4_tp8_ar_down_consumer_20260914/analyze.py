"""Recompute the small ABBA from saved outputs; no GPU/service activity."""
import hashlib
import argparse
import json
from pathlib import Path
import statistics

from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--concurrency',type=int,choices=(1,32),default=32)
    args=parser.parse_args()
    prefix='C1-' if args.concurrency==1 else ''
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    records={}
    cases={}
    for arm in ('A1','B1','B2','A2'):
        path=ROOT/f'{prefix}{arm}.json'
        data=json.loads(path.read_text())
        assert data['status']=='complete' and data['concurrency']==args.concurrency and not data['ignore_eos']
        rates=[]; windows=[]; outputs=[]; wall_tokens=wall_seconds=0
        for rep in data['rounds']:
            assert len(rep['waves'])==1, 'This experiment fixes two waves/leg'
            wave=rep['waves'][0]
            assert wave['resident_seconds']>0
            windows.append(wave['resident_seconds'])
            rates.append(wave['resident_tokens']/wave['resident_seconds'])
            wall_seconds+=wave['http_wall_seconds']
            for q in wave['requests']:
                assert q['spec_accept_length'] is None
                assert tokenizer.decode(q['output_ids'],skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)==q['text']
                wall_tokens+=len(q['output_ids'])
                outputs.append(q)
            assert len({q['index'] for q in wave['requests']})==args.concurrency
        med=statistics.median(rates)
        assert abs(med-data['median_decode_tok_s'])<1e-8
        first={q['index']:q['output_ids'] for q in data['rounds'][0]['waves'][0]['requests']}
        second={q['index']:q['output_ids'] for q in data['rounds'][1]['waves'][0]['requests']}
        tail_repetition=[]
        for q in outputs:
            ids=q['output_ids'][-256:]
            grams=[tuple(ids[i:i+8]) for i in range(len(ids)-7)]
            tail_repetition.append(1-len(set(grams))/len(grams) if grams else 0.)
        records[arm]=dict(resident_tok_s=med,waves_tok_s=rates,resident_seconds=windows,
                          http_tok_s=wall_tokens/wall_seconds,complete_responses=len(outputs),
                          exact_ids_to_text=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          manifest_sha256=data['manifest_sha256'],
                          within_leg_full_ids_exact=sum(first[i]==second[i] for i in first),
                          max_tail8_repeat_ratio=max(tail_repetition),
                          tail8_repeat_above_0_5=sum(x>.5 for x in tail_repetition))
        cases[arm]=outputs
    assert len({r['manifest_sha256'] for r in records.values()})==1
    a=statistics.mean(records[s]['resident_tok_s'] for s in ('A1','A2'))
    b=statistics.mean(records[s]['resident_tok_s'] for s in ('B1','B2'))
    report=dict(scope=f'Small TP8 C{args.concurrency} original-weight native-AR ABBA, fixed public-source cases, two natural-EOS waves/leg',
                candidate_hits_requested_tier=args.concurrency==32,
                interpretation='C32 candidate trial' if args.concurrency==32 else 'C1 isolation/regression only: existing M1 kernel unchanged',
                control_mean=a,candidate_mean=b,gain_pct=100*(b/a-1),arms=records,
                control_return_pct=100*(records['A2']['resident_tok_s']/records['A1']['resident_tok_s']-1),
                candidate_leg_difference_pct=100*(records['B2']['resident_tok_s']/records['B1']['resident_tok_s']-1),
                caveat='Short screening trial, not the earlier >=30s/round formal matrix; ID/text integrity and component equality are not whole-model bitwise/factual correctness.')
    if args.concurrency==1:
        report['unique_completion_sequences_across_all_8_waves']=len({tuple(q['output_ids']) for qs in cases.values() for q in qs})
    (ROOT/f'{prefix}summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    # Compact, readable quality-review excerpts; all full outputs stay in JSON.
    for arm in ('A1','B1','B2','A2'):
        print('\nQUALITY',arm)
        for q in cases[arm][:3]:
            print(q['index'],len(q['output_ids']),repr(q['text'][:1000]))


if __name__=='__main__': main()
