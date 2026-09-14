"""Close each measured forward envelope without mixing rank-wise stage maxima."""
import json
from pathlib import Path
from statistics import mean
import argparse
import re

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--label', default='B')
args=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9-]+',args.label)
ROOT = Path(__file__).resolve().parent / 'capture'
NAMES = ['attn_mhc_norm', 'attn_entry_gap', 'attn_prepare', 'sparse_attention',
         'attn_output_projection_collective', 'ffn_mhc_norm', 'moe_collective']
frames = [json.loads(p.read_text()) for p in sorted((ROOT/'markers').glob('rank-*-frame-*.json'))]
assert len(frames) == 128
assert {(f['rank'], f['sequence']) for f in frames} == {(r,s) for r in range(8) for s in range(1,17)}
assert all(f['wall_clock_khz'] == 25000 and len(f['layers']) == 43 for f in frames)
assert all(l['coarse_valid'] for f in frames for l in f['layers'])
assert json.loads((ROOT/'complete.json').read_text())['input_echo_exact'] == 64
assert not json.loads((ROOT/'P16-markers-B.stop.json').read_text())['remaining']

def summarize(f):
    layers = f['layers']
    spans = {n:sum(l['coarse_us'][i] for l in layers)/1000 for i,n in enumerate(NAMES)}
    # Preparation sub-boundaries are serial in the ordinary native prefill path.
    # 10->15 includes Q/KV preparation AND indexer, not an isolated indexer timer.
    sub = {'qkv_projection':(9,10), 'qkv_prepare_plus_indexer':(10,15),
           'core_compressor':(15,14), 'routed_stage':(19,20),
           'moe_output_collective':(23,24)}
    subdivisions = {}
    for name,(a,b) in sub.items():
        assert all(0 < l['ticks'][a] <= l['ticks'][b] for l in layers)
        subdivisions[name] = sum((l['ticks'][b]-l['ticks'][a])*f['us_per_tick']/1000 for l in layers)
    gaps = sum(layers[i+1]['start_us']-layers[i]['end_us'] for i in range(42))/1000
    assert gaps >= 0
    outer = (layers[0]['start_us'] + f['realtime_frame_ms']*1000-layers[-1]['end_us'])/1000
    assert outer >= 0
    assert abs(sum(spans.values())+gaps+outer-f['realtime_frame_ms']) < 1e-6
    return dict(rank=f['rank'], sequence=f['sequence'], rows=f['rows'],
                requests=f['requests'], prefix_lens=f['prefix_lens'],
                frame_ms=f['realtime_frame_ms'], stages_ms=spans,
                interlayer_ms=gaps, outer_ms=outer, subdivisions_ms=subdivisions)

selected=[]
spread=[]
for seq in range(5,17):  # Entire first wave excluded, not merely first forward.
    group=[f for f in frames if f['sequence']==seq]
    assert len({(f['rows'],tuple(f['extend_lens']),tuple(f['prefix_lens'])) for f in group})==1
    chosen=max(group,key=lambda f:f['realtime_frame_ms'])
    selected.append(summarize(chosen))
    times=[f['realtime_frame_ms'] for f in group]
    spread.append(max(times)-min(times))
waves=[]
for i in range(3):
    fs=selected[i*4:(i+1)*4]
    wave=dict(sequence_range=[fs[0]['sequence'],fs[-1]['sequence']],
              selected_ranks=[f['rank'] for f in fs],
              frame_ms=sum(f['frame_ms'] for f in fs),
              stages_ms={n:sum(f['stages_ms'][n] for f in fs) for n in NAMES},
              subdivisions_ms={n:sum(f['subdivisions_ms'][n] for f in fs) for n in fs[0]['subdivisions_ms']},
              interlayer_ms=sum(f['interlayer_ms'] for f in fs),
              outer_ms=sum(f['outer_ms'] for f in fs))
    wave['http_s']=json.loads((ROOT/f'trace{i+1}-summary.json').read_text())['request_wall_s']
    waves.append(wave)
result=dict(scope='Diagnostic spans include synchronization/waits and instrumentation. Select longest rank envelope per forward, then retain all its stages; not independent rank maxima or a synchronized multi-rank critical-path reconstruction.',
            snapshots=len(frames), warmup_frames=32,
            event_envelope_ratio_range=[min(f['event_envelope_ratio'] for f in frames if f['sequence']>4),max(f['event_envelope_ratio'] for f in frames if f['sequence']>4)],
            max_rank_envelope_spread_ms=max(spread),
            warm_mean_wave_ms=mean(w['frame_ms'] for w in waves),
            warm_mean_stages_ms={n:mean(w['stages_ms'][n] for w in waves) for n in NAMES},
            warm_mean_subdivisions_ms={n:mean(w['subdivisions_ms'][n] for w in waves) for n in waves[0]['subdivisions_ms']},
            waves=waves,selected_frames=selected)
(ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('waves','selected_frames')},indent=2))

if len(frames[0]['layers'][0]['ticks'])==64:
    detail_waves=[]
    for wave in range(3):
        totals={};counts={};paths={}
        for selected_frame in selected[wave*4:(wave+1)*4]:
            f=next(f for f in frames if (f['rank'],f['sequence']) ==
                   (selected_frame['rank'],selected_frame['sequence']))
            paths.update(f.get('detail_paths',{}))
            for layer in f['layers']:
                t=layer['ticks']
                for side,base in [('attn',32),('ffn',40)]:
                    if not any(t[base:base+5]):
                        # First-layer pre has no preceding post boundary.
                        assert side=='attn' and layer['layer']==0
                        continue
                    assert all(0<t[base+i]<=t[base+i+1] for i in range(4))
                    for i,name in enumerate(('post','mix','sinkhorn','weighted_norm')):
                        key=f'{side}_{name}'
                        totals[key]=totals.get(key,0)+(t[base+i+1]-t[base+i])*.04/1000
                        counts[key]=counts.get(key,0)+1
                if t[48]:
                    assert all(0<t[i]<=t[i+1] for i in range(48,54))
                    for a,b,name in [(49,50,'index_weights'),(50,51,'index_query'),
                                     (51,52,'index_compressor'),(52,53,'index_logits_plus_metadata'),
                                     (53,54,'index_topk_plus_metadata')]:
                        totals[name]=totals.get(name,0)+(t[b]-t[a])*.04/1000
                        counts[name]=counts.get(name,0)+1
        detail_waves.append(dict(totals_ms=totals,counts=counts,paths=paths))
    assert all(w['counts']==detail_waves[0]['counts'] for w in detail_waves)
    details=dict(scope='Nested subdivisions, already included in coarse spans. Same selected rank/forward. Metadata/launch/wait costs remain inside call boundaries.',
                 mean_wave_ms={n:mean(w['totals_ms'][n] for w in detail_waves) for n in detail_waves[0]['totals_ms']},
                 waves=detail_waves)
    (ROOT/'details-analysis.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps(details['mean_wave_ms'],indent=2))
