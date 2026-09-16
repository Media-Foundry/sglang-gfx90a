"""Synthetic CPU marker subdivision tests, not hardware timing evidence."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parent
prior=root.parent/'dsv4_prefill_length_profile_20260916/8k'
analysis=json.loads((prior/'analysis.json').read_text())
details=json.loads((prior/'details-analysis.json').read_text())
frames=[json.loads(p.read_text()) for p in sorted((prior/'markers').glob('rank-*-frame-*.json'))]
assert len(frames)==128
with tempfile.TemporaryDirectory(prefix='dsv4-owner-markers-') as temporary:
    out=Path(temporary);(out/'markers').mkdir()
    (out/'analysis.json').write_text(json.dumps(analysis))
    (out/'complete.json').write_text((prior/'complete.json').read_text())
    for frame in frames:
        f=copy.deepcopy(frame)
        for layer in f['layers']:
            if f['detail_paths'].get(str(layer['layer'])+':53')=='indexer_owner_chain_done':
                t=layer['ticks'];a,b=t[52],t[53]
                for i in range(7):t[55+i]=a+(b-a)*(i+1)//8
        (out/'markers'/f'rank-{f["rank"]}-frame-{f["sequence"]:04d}.json').write_text(json.dumps(f))
    command=[sys.executable,str(root/'analyze.py'),'--root',str(out)]
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
    result=json.loads((out/'owner-analysis.json').read_text())
    assert result['all_rank_frames_checked']==128
    assert math.isclose(result['warm_mean_owner_ms']['owner_total'],details['mean_wave_ms']['index_owner_chain'],rel_tol=1e-12)
    assert math.isclose(sum(result['owner_share'].values()),1,rel_tol=1e-12)
    print('PASS synthetic owner phases exactly close archived outer-owner envelopes')
    (out/'owner-analysis.json').unlink()
    bad=out/'markers/rank-0-frame-0001.json'
    f=json.loads(bad.read_text())
    layer=next(l for l in f['layers'] if f['detail_paths'].get(str(l['layer'])+':53')=='indexer_owner_chain_done')
    layer['ticks'][58]=0
    bad.write_text(json.dumps(f))
    failed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert failed.returncode!=0 and b'AssertionError' in failed.stderr
    print('PASS missing internal marker rejected even in unscored warmup')
