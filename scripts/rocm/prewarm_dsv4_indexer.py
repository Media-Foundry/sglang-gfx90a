"""Explicit compile-only wide-C4 priming; run before launching a TP8 service.

Example: HIP_VISIBLE_DEVICES=4 python scripts/rocm/prewarm_dsv4_indexer.py
  --signature 4096:64:64 --signature 8192:128:128 --output prewarm.json
Use the same TRITON_CACHE_DIR and Python/compiler environment as the service.
No default profile, KV selection, model weight or steady execution is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signature',action='append',required=True,metavar='WIDTH:PAGE_COLUMNS:PAGE_STRIDE')
    parser.add_argument('--shuffle',type=int,choices=(0,8,16),default=16)
    parser.add_argument('--fp8-fnuz',action='store_true')
    parser.add_argument('--fp16-dot',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite output')
    from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_prewarm import (
        prewarm_query_reuse4, validate_signature,
    )
    specs=[]
    for value in args.signature:
        try:
            width,columns,stride=map(int,value.split(':'))
            validate_signature(width,columns,stride,args.shuffle)
        except ValueError as error:parser.error(str(error))
        specs.append((width,columns,stride))
    import torch
    import triton
    torch.cuda.set_device(0)
    records=[]
    for width,columns,stride in dict.fromkeys(specs):
        start=time.perf_counter()
        code=prewarm_query_reuse4(width=width,page_columns=columns,page_stride=stride,
            preshuffle_tile=args.shuffle,fp8_fnuz=args.fp8_fnuz,dot_fp16=args.fp16_dot)
        record=dict(width=width,page_columns=columns,page_stride=stride,
                    wall_s=time.perf_counter()-start,artifact=code.hash,
                    hsaco_sha256=hashlib.sha256(code.asm['hsaco']).hexdigest())
        records.append(record);print(json.dumps(record),flush=True)
    args.output.write_text(json.dumps(dict(compile_only=True,records=records,
        torch_version=torch.__version__,triton_version=triton.__version__,
        cache_dir_env=os.environ.get('TRITON_CACHE_DIR'),
        cache_note='Unset means the Triton default cache; same environment required at service launch.',
        shuffle=args.shuffle,fp8_fnuz=args.fp8_fnuz,fp16_dot=args.fp16_dot,
        arch=torch.cuda.get_device_properties(0).gcnArchName),indent=2)+'\n')


if __name__=='__main__':main()
