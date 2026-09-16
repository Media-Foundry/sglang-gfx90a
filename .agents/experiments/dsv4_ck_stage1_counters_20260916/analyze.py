"""Separate useful-flop estimate, hardware counters, and service critical spans."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
target=root/'summary.json';assert not target.exists()
collection=json.loads((root/'collection.json').read_text());assert collection['status']=='complete'
timing=json.loads((root/'timing.json').read_text());assert timing['captured_intermediate_byte_exact']
measurements={}
for family in collection['passes']:
    name=family['name'];proof=json.loads((root/(name+'-result.json')).read_text())
    assert proof['profiled_exact_replays']==3 and proof['module']==timing['module']
    assert proof['pci']==timing['pci']
    directory=root/name/'DiamondHill'
    trace=next(directory.glob('*kernel_trace.csv'));counter=next(directory.glob('*counter_collection.csv'))
    kernels=list(csv.DictReader(trace.open()));counts=list(csv.DictReader(counter.open()))
    assert len(kernels)==3 and len(counts)==3*len(family['counters'])
    assert len({k['Agent_Id'] for k in kernels})==1
    samples=[]
    for row in kernels:
        assert 'kernel_moe_gemm' in row['Kernel_Name']
        values={c['Counter_Name']:float(c['Counter_Value']) for c in counts if c['Dispatch_Id']==row['Dispatch_Id']}
        assert set(values)==set(family['counters'])
        samples.append(dict(profiled_ms=(int(row['End_Timestamp'])-int(row['Start_Timestamp']))/1e6,
            vgpr=int(row['VGPR_Count']),acc_vgpr=int(row['Accum_VGPR_Count']),lds_bytes=int(row['LDS_Block_Size']),
            scratch_bytes=int(row['Scratch_Size']),grid=[int(row['Grid_Size_'+d]) for d in ('X','Y','Z')],
            counters=values,kernel=row['Kernel_Name']))
    measurements[name]=dict(samples=samples,mean_profiled_ms=statistics.mean(s['profiled_ms'] for s in samples),
        mean_counters={key:statistics.mean(s['counters'][key] for s in samples) for key in family['counters']},
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),counter_sha256=hashlib.sha256(counter.read_bytes()).hexdigest())
useful_flops=2*32767*6*4096*256*2
result=dict(scope='Historical real M32767 stage1 only, one GCD, output exactly matches captured stage1; not E2E',
    timing=timing,measurements=measurements,useful_flops=useful_flops,
    useful_tflops=useful_flops/(timing['median_ms']*1e9),
    interpretation='Useful FLOPs exclude padding/nonlinear/scalar work. Instruction counts are not utilization. Profiler intercept changes queue properties; profiled durations are diagnostic only.')
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(uninstrumented_ms=timing['median_ms'],useful_tflops=result['useful_tflops'],
    counters={k:v['mean_counters'] for k,v in measurements.items()},
    resources=measurements['issue']['samples'][0]),indent=2))
