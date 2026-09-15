#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/Code/sglang
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export SGLANG_USE_AITER=1
export PYTHONPATH=/home/pc/Code/sglang/python:/home/pc/Code/sglang/python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:/home/pc/Code/sglang/python/sglang/kernels/aot/python
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
out=.agents/experiments/dsv4_c16_wide_owner_20260915
test ! -e "$out/result.json"
test ! -e "$out/run.log"
/home/pc/anaconda3/envs/DS/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$out/oracle.py" --output "$out/result.json" 2>&1 | tee "$out/run.log"
