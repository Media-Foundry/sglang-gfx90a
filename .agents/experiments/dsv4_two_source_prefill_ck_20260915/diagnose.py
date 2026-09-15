"""Distinguish new bank-address error from inherited CK probability rounding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in owners for p in g.get('process_list',[])),owners
import torch
from torch.utils.cpp_extension import include_paths
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import _sparse_attn_v4_paged_prefill_triton
from oracle import Runner, ROOT

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--refined',action='store_true')
args=parser.parse_args()
output=ROOT/('diagnosis-refined.json' if args.refined else 'diagnosis.json');assert not output.exists()
path=ROOT/'smoke-repro.failure.pt'
f={k:v.cuda() for k,v in torch.load(path,weights_only=True).items()}
runner=Runner(*(f[k] for k in ('q','pkv','pi','pp','ekv','ei','ep','sink')),refined=args.refined)
new=runner(2).clone()
kv=torch.cat((f['pkv'],f['ekv']),dim=0)
indices=runner.combined.clone()
valid=indices>=0
indices=torch.where(valid, torch.where((indices&(1<<30))!=0,
    (indices&((1<<30)-1))+len(f['pkv']),indices),-1)
module=load_jit('gfx90a_dsv4_sparse_h8_sentinel_oracle_v1',
    cuda_files=['deepseek_v4/gfx90a_dsv4_sparse_h8_oracle.cuh'],
    cuda_wrappers=[('run','sglang::Gfx90aDsv4SparseH8Oracle::run')],
    extra_cuda_cflags=['-O3','-std=c++20','-DCK_ENABLE_BF16','-DCK_USE_XDL','-DSGLANG_DSV4_CK_SENTINEL_ORACLE=1'],
    extra_include_paths=[*include_paths(),
        '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include',
        '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/library/include'])
old=torch.empty_like(new)
module.run(f['q'],kv,indices,runner.ptr,f['sink'],old,runner.scratch,512**-.5)
# The production Triton only bounds-checks negative slots. Sanitize synthetic
# positive OOBs to -1, preserving the same valid occurrences for this comparison.
pi=torch.where((f['pi']>=0)&(f['pi']<len(f['pkv'])),f['pi'],-1)
ei=torch.where((f['ei']>=0)&(f['ei']<len(f['ekv'])),f['ei'],-1)
triton=_sparse_attn_v4_paged_prefill_triton(f['q'],f['pkv'],pi,f['pp'],f['ekv'],ei,f['ep'],f['sink'],512**-.5)
index=(1,0,237)
result=dict(fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),refined=args.refined,
    new_equals_original_ck_bits=torch.equal(new.view(torch.uint8),old.view(torch.uint8)),
    global_max_abs={name:float((value.float()-f['expected']).abs().max()) for name,value in [('new',new),('original_ck',old),('production_triton',triton)]},
    failed_element={name:float(value[index]) for name,value in [('new',new),('original_ck',old),('production_triton',triton),('reference',f['expected'])]},
    prefix_lengths=(f['pp'][1:]-f['pp'][:-1]).cpu().tolist(),
    extend_lengths=(f['ep'][1:]-f['ep'][:-1]).cpu().tolist())
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
