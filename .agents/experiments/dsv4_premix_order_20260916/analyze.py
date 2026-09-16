"""Aggregate recorded instruction/traffic counters without calling them utilization."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parent;target=root/'counter-summary.json';assert not target.exists()
report=dict(scope='Three isolated original pre-mix calls/pass; profiler durations, not E2E throughput',passes={})
for name,pid in [('memory',2302841),('issue',2303008)]:
    assert json.loads((root/f'{name}-result.json').read_text())['status']=='complete'
    directory=root/name/'DiamondHill'
    trace=directory/f'{pid}_kernel_trace.csv';count=directory/f'{pid}_counter_collection.csv'
    with trace.open() as f:rows=list(csv.DictReader(f))
    with count.open() as f:counters=list(csv.DictReader(f))
    assert len(rows)==3 and len(counters)==6
    assert all(r['Kernel_Name']=='premix8_pair' and r['Agent_Id']=='Agent 7' for r in rows)
    values={}
    for c in counters:
        assert c['Kernel_Name']=='premix8_pair' and c['Agent_Id']=='Agent 7'
        values.setdefault(c['Counter_Name'],[]).append(float(c['Counter_Value']))
    assert all(len(v)==3 for v in values.values())
    report['passes'][name]=dict(profiled_ms=[(int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e6 for r in rows],
        raw_counters=values,mean_counters={k:mean(v) for k,v in values.items()},
        profiler_vgpr=int(rows[0]['VGPR_Count']),scratch_bytes=int(rows[0]['Scratch_Size']),
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),counter_sha256=hashlib.sha256(count.read_bytes()).hexdigest())
target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
