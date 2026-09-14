"""Read-only cache/source inventory, NOT evidence of a module loaded by a service."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache-root',type=Path,default=Path('/home/pc/.cache/sglang/triton'))
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
assert not args.output.exists()
repo=Path(__file__).resolve().parents[3]
rows=[]
for path in sorted(args.cache_root.glob('*/_sparse_attn_v4_paged_prefill_kernel.json')):
    meta=json.loads(path.read_text())
    if meta.get('num_warps')!=1 or meta.get('target',{}).get('arch')!='gfx90a':continue
    ir=path.with_suffix('.ttir').read_text()
    asm=path.with_suffix('.amdgcn').read_text()
    def constant(name):
        found=re.search(r'%'+name+r' = arith.constant dense<(\d+)>',ir)
        return int(found[1]) if found else None
    if constant('h_mask')!=8 or constant('d_mask')!=512:continue
    def directive(name):
        values=re.findall(r'\.'+name+r'\s+(\d+)',asm)
        assert len(values)==1,(path,name,values)
        return int(values[0])
    agpr=re.findall(r'\.agpr_count:\s+(\d+)',asm)
    assert len(agpr)==1
    rows.append(dict(cache=str(path.parent),heads=8,dim=512,warps=1,
        launch_shared_bytes=meta.get('shared'),
        next_free_vgpr=directive('amdhsa_next_free_vgpr'),
        accum_offset=directive('amdhsa_accum_offset'),agpr_count=int(agpr[0]),
        static_lds_bytes=directive('amdhsa_group_segment_fixed_size'),
        scratch_bytes=directive('amdhsa_private_segment_fixed_size'),
        artifact_sha256={suffix:hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest()
                         for suffix in ('.json','.ttir','.amdgcn','.hsaco')}))
assert rows,'No matching artifacts; do not infer kernel resource usage'
sources=['python/sglang/srt/models/deepseek_v4.py',
    'python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py',
    'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py',
    'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py']
result=dict(scope=__doc__,artifacts=rows,
    resource_caveat='next_free_vgpr includes the accumulator mapping; do not call it354 ordinary VGPRs or infer occupancy from it alone.',
    source_sha256={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in sources})
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(matching_artifacts=len(rows),resources=[{k:v for k,v in r.items()
    if k not in ('cache','artifact_sha256')} for r in rows])))
