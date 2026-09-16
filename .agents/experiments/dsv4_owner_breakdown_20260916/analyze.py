"""Close all owner substages within the same selected-rank forward envelopes."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
args=p.parse_args();root=args.root
target=root/'owner-analysis.json';assert not target.exists()
coarse=json.loads((root/'analysis.json').read_text())
done=json.loads((root/'complete.json').read_text())
n=done['forwards_per_wave'];tokens=done['input_tokens_per_wave']
frames={(f['rank'],f['sequence']):f for path in (root/'markers').glob('rank-*-frame-*.json')
        for f in [json.loads(path.read_text())]}
names=('planning','pack','logits','topk','gather','reconstruct')

def spans(frame):
    totals={name:0. for name in (*names,'outside_detail','owner_total')}
    count=0
    for layer in frame['layers']:
        t=layer['ticks'];l=layer['layer']
        if frame['detail_paths'].get(f'{l}:53')!='indexer_owner_chain_done':
            assert not any(t[55:62]),(frame['rank'],frame['sequence'],l)
            continue
        assert 0<t[52]<=t[55] and t[61]<=t[53]
        assert all(t[i]<=t[i+1] for i in range(55,61))
        for i,name in enumerate(names):
            totals[name]+=(t[56+i]-t[55+i])*frame['us_per_tick']/1000
        totals['outside_detail']+=((t[55]-t[52])+(t[53]-t[61]))*frame['us_per_tick']/1000
        totals['owner_total']+=(t[53]-t[52])*frame['us_per_tick']/1000
        count+=1
    assert count==21,count
    assert abs(sum(totals[k] for k in (*names,'outside_detail'))-totals['owner_total'])<1e-7
    return totals

# Check every rank/frame, including warmup; only report the measured waves.
all_spans={key:spans(frame) for key,frame in frames.items()}
selected=coarse['selected_frames'];assert len(selected)==n*3
waves=[]
for wave in range(3):
    choices=selected[wave*n:(wave+1)*n]
    totals={name:sum(all_spans[f['rank'],f['sequence']][name] for f in choices)
            for name in (*names,'outside_detail','owner_total')}
    waves.append(dict(totals_ms=totals,ranks=[f['rank'] for f in choices]))
mean={name:statistics.mean(w['totals_ms'][name] for w in waves) for name in waves[0]['totals_ms']}
per_rank={str(rank):{name:sum(v[name] for (r,seq),v in all_spans.items() if r==rank and seq>n)/3
                    for name in mean} for rank in range(8)}
result=dict(diagnostic_only=True,scope='Same longest outer-envelope rank per forward; nested owner stages incl waits, not standalone kernel or independent rank-max sums.',
    input_tokens=tokens,warm_mean_owner_ms=mean,us_per_input_token={k:v*1000/tokens for k,v in mean.items()},
    owner_share={k:v/mean['owner_total'] for k,v in mean.items() if k!='owner_total'},
    all_rank_frames_checked=len(frames),waves=waves,per_rank_mean_wave_ms=per_rank)
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('waves','per_rank_mean_wave_ms')},indent=2))
