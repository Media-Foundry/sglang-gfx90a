"""Summarize actual isolated counters; no inference of peak utilization or E2E gain."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent;target=root/'summary.json';assert not target.exists()
result={}
for family,check in (('issue-v3','issue-v3-result.json'),('memory','memory-result.json')):
    proof=json.loads((root/check).read_text());assert proof['status']=='complete' and proof['partial_byte_exact']
    directory=root/family/'DiamondHill'
    trace=next(directory.glob('*kernel_trace.csv'));counter=next(directory.glob('*counter_collection.csv'))
    kernels=list(csv.DictReader(trace.open()));counts=list(csv.DictReader(counter.open()))
    assert len(kernels)==6 and len(counts)==12
    assert {k['Agent_Id'] for k in kernels}=={'Agent 8'}
    grouped={}
    for k in kernels:
        name=k['Kernel_Name']
        label='reducer' if 'ck_slot_reduce_kernel' in name else 'unique_ck'
        assert label=='reducer' or 'kernel_moe_gemm_2lds' in name
        row=dict(ms=(int(k['End_Timestamp'])-int(k['Start_Timestamp']))/1e6,
            vgpr=int(k['VGPR_Count']),acc_vgpr=int(k['Accum_VGPR_Count']),
            lds_bytes=int(k['LDS_Block_Size']),scratch_bytes=int(k['Scratch_Size']),
            grid=[int(k['Grid_Size_'+d]) for d in ('X','Y','Z')],
            workgroup=[int(k['Workgroup_Size_'+d]) for d in ('X','Y','Z')],
            counters={c['Counter_Name']:float(c['Counter_Value']) for c in counts if c['Dispatch_Id']==k['Dispatch_Id']})
        assert len(row['counters'])==2
        grouped.setdefault(label,[]).append(row)
    assert all(len(rows)==3 for rows in grouped.values())
    result[family]=dict(samples=grouped,means={label:dict(
        profiled_ms=statistics.mean(r['ms'] for r in rows),
        counters={c:statistics.mean(r['counters'][c] for r in rows) for c in rows[0]['counters']})
        for label,rows in grouped.items()},
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        counter_sha256=hashlib.sha256(counter.read_bytes()).hexdigest())
summary=dict(scope='Isolated historical real M32767 W2 input on physical GCD6; not fresh corrected-model routing or service timing',
    measurements=result,interpretation='Unique W2 writes about3GiB partial; reducer reads about3GiB. Weight-fetch savings alone cannot remove these writes/reads. Instruction counts are not utilization percentages.',
    profiler_caveat='SDK substitutes a system-memory intercept queue and does not preserve original priority/CU mask. Durations are diagnostic only.',
    earlier_failures=['Initial mixed SDK plus profiler-injected hipconfig stdout prevented AIter import.',
        'Isolating version query fixed that; injected rocminfo stdout then broke architecture parsing.',
        'Explicit GPU_ARCHS=gfx90a plus runtime hardware assertion allowed collection. No fake version/arch query output was used.'])
target.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v['means'] for k,v in result.items()},indent=2))
