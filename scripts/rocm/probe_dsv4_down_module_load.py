#!/usr/bin/env python3
"""Measure cached JIT module loading separately from graph/tensor allocation.

No model forward. By default no kernel launch; --first-launch adds real-shape
temporary fixtures to measure the separate first-use stage and check outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--physical-gpu', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-launch', action='store_true',
                        help='also compare real-shape first launches with bounded temporary fixtures')
    args = parser.parse_args()
    assert not args.output.exists()
    assert os.environ.get('HIP_VISIBLE_DEVICES') == str(args.physical_gpu)
    owner = psutil.Process(args.owner_pid)
    assert owner.cmdline()[1:3] == ['-m', 'sglang.launch_server']
    raw = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    seen = {int(x['process_info']['pid']) for g in raw for x in g.get('process_list', [])
            if isinstance(x['process_info'], dict)}
    allowed = {owner.pid, os.getpid(), *(p.pid for p in owner.children(recursive=True))}
    assert not seen - allowed, ('foreign GPU PIDs', sorted(seen - allowed))
    torch.cuda.init()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx90a'
    from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
        _jit_down_grouped, _jit_down_grouped_uniform,
    )
    shape = (256, 32, 6, 4096, 256, 4, 2, 8, 832, 2)
    def snapshot():
        torch.cuda.synchronize()
        return dict(free=torch.cuda.mem_get_info()[0],
                    allocated=torch.cuda.memory_allocated(),
                    reserved=torch.cuda.memory_reserved())
    snapshots = [dict(stage='context_and_imports', **snapshot())]
    modules = []
    for name, load in [('baseline', _jit_down_grouped), ('candidate', _jit_down_grouped_uniform)]:
        start = time.monotonic()
        modules.append(load(*shape))
        snapshots.append(dict(stage=name, wall_s=time.monotonic()-start, **snapshot()))
    launch_snapshots = []
    if args.first_launch:
        from bench_dsv4_gfx90a_occupancy_bucket_oracle import make_metadata
        from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import gfx90a_fp4_expert_down_grouped
        assert torch.cuda.mem_get_info()[0] >= 2 * 1024**3
        torch.manual_seed(2090801)
        x = torch.randint(-127, 128, (32, 6, 256), device='cuda', dtype=torch.int8)
        scales = torch.rand((32, 6, 8), device='cuda') * .01
        weights = torch.randint(0, 256, (256, 4096, 128), device='cuda', dtype=torch.uint8)
        weight_scales = torch.randint(122, 128, (256, 4096, 8), device='cuda', dtype=torch.uint8)
        router_weights = torch.rand((32, 6), device='cuda')
        ids = torch.rand((32, 256), device='cuda').topk(6, dim=1).indices.int()
        metadata = make_metadata(ids, assignments=4)
        outputs = []
        launch_snapshots.append(dict(stage='fixtures_ready', **snapshot()))
        for enabled in (False, True):
            start = time.monotonic()
            outputs.append(gfx90a_fp4_expert_down_grouped(
                x, scales, weights, weight_scales, metadata.sorted_ids,
                metadata.sorted_experts, metadata.valid, router_weights,
                assignments=4, rows=2, waves=8, blocks=832, use_lds_lut=True,
                uniform_metadata=enabled))
            memory = snapshot()
            launch_snapshots.append(dict(stage=f'first_launch_{enabled}',
                                         wall_s=time.monotonic()-start, **memory))
        assert torch.isfinite(outputs[1]).all()
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and ('gfx90a_fp4_expert_down_grouped_256_32' in parts[5]
                               or 'gfx90a_fp4_tp8_down_uniform_service_256_32' in parts[5]):
            paths.add(parts[5])
    inventory = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(paths)}
    assert len(inventory) == 2, inventory
    result = dict(physical_gpu=args.physical_gpu, owner_pid=owner.pid,
                  snapshots=snapshots, module_sha256=inventory,
                  additional_loader_device_bytes=snapshots[0]['free']-snapshots[-1]['free'],
                  launch_snapshots=launch_snapshots,
                  first_launch_bit_exact=True if args.first_launch else None,
                  caveat=('First-launch wall times include host dispatch and allocation; not steady performance'
                          if args.first_launch else 'No kernel launch: lazy device instantiation cost is not covered'))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
