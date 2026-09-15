"""Read-only recomputation of current evidence; never overwrite prior results."""
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
repo=root.parents[2]
for name,var in [('dsv4_c16_wide_prewarm_service_20260915','report'),
                 ('dsv4_c16_long_attention_check_20260915','result')]:
    p=root.parent/name/'analyze.py'
    source=p.read_text()
    marker="target=root/'analysis.json';assert not target.exists();"
    assert source.count(marker)==1
    # Run all original assertions, omitting only the immutable output writer.
    scope={'__file__':str(p),'__name__':'readonly_recheck'}
    exec(compile(source.split(marker)[0],str(p),'exec'),scope)
    assert scope[var]==json.loads(p.with_name('analysis.json').read_text())
    print(name,'all assertions and saved analysis match')

r=json.loads((root/'result.json').read_text())
assert r['status']=='complete' and len(r['cases'])==3
for p,h in r['sources'].items():
    assert hashlib.sha256((repo/p).read_bytes()).hexdigest()==h,p
for case in r['cases']:
    assert len(case['inputs_sha256'])==8 and len(set(case['inputs_sha256']))==1
    assert len(case['checks'])==3
    assert all(c['scores_byte_exact'] and c['logical_physical_exact'] for c in case['checks'])
    ranks=case['per_rank_ms'];assert len(ranks)==8
    for arm in ('A','B'):
        assert all(len(rank[arm])==6 for rank in ranks)
        mx=[max(rank[arm][i] for rank in ranks) for i in range(6)]
        assert mx==case['rankmax_ms'][arm]
        assert statistics.median(mx)==case['median_ms'][arm]
    a,b=case['median_ms']['A'],case['median_ms']['B']
    assert a/b==case['speedup']
    print(case['name'],f'{a:.6f} -> {b:.6f} ms, {a/b:.4f}x; NOT service throughput')
