"""Frozen full service inputs: isolate row-position dependence of QKV GEMM."""
import json
import os
import argparse
from pathlib import Path
import statistics

import torch
from woa_tiles import project


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--service-kernel', action='store_true')
args = parser.parse_args()
if args.service_kernel:
    from sglang.kernels.ops.debug.dsv4_prefill_qkv import project as service_project
assert os.environ.get("HIP_VISIBLE_DEVICES") == "4"
target = root / ("qkv-service-oracle.json" if args.service_kernel else "qkv-order-oracle.json")
assert not target.exists()
run = root / "layer0-changed-rows"
assert (run / "complete.json").exists()
load = lambda arm, name: torch.load(
    run / ("trace-" + arm) / ("layer_0_rank_0_" + name + ".pt"), weights_only=True)
x = [load(arm, "prepare_full_input").cuda() for arm in ("A1", "B1")]
y_service = [load(arm, "prepare_full_qkv_a").cuda() for arm in ("A1", "B1")]
w = torch.load(root / "all-ranks/trace-A1/layer_0_rank_0_projection_wqkv_a.pt",
               weights_only=True).cuda()
assert w.shape == (1536, 4096) and w.dtype == torch.bfloat16
assert x[0].shape == (32768, 4096) and x[1].shape == (32767, 4096)
assert torch.equal(x[0][-8192:], x[1][-8192:])


def differences(a, b):
    d = (a.float() - b.float()).abs()
    return dict(changed=int(torch.count_nonzero(d)), max_abs=float(d.max()))


def measure(fn):
    for _ in range(3): fn()
    times = []
    for _ in range(12):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(); fn(); end.record(); end.synchronize()
        times.append(begin.elapsed_time(end))
    return dict(median_ms=statistics.median(times), samples_ms=times)


baseline = [torch.nn.functional.linear(t, w) for t in x]
result = dict(baseline_vs_service=[differences(a, b) for a, b in zip(baseline, y_service)],
              baseline_shift=differences(baseline[0][-8192:], baseline[1][-8192:]), candidates=[])
for tile in ((128, 128, 128, 8), (64, 128, 128, 4), (128, 128, 64, 8)):
    if args.service_kernel and tile != (128, 128, 128, 8):
        continue
    fn = (lambda t: service_project(t, w)) if args.service_kernel else (lambda t: project(t, w, tile))
    output = [fn(t) for t in x]
    if args.service_kernel:
        assert all(torch.equal(v, project(t, w, tile)) for t, v in zip(x, output))
    record = dict(tile=tile, shift=differences(output[0][-8192:], output[1][-8192:]),
                  vs_service=[differences(a, b) for a, b in zip(output, y_service)],
                  timing=measure(lambda: fn(x[0])))
    assert record['shift']['changed'] == 0, record
    result['candidates'].append(record)
result['baseline_timing'] = measure(lambda: torch.nn.functional.linear(x[0], w))
# Probe additional offsets and actual-input perturbations, not just one shape pair.
xa = torch.zeros_like(x[0]); xb = torch.zeros_like(x[1]); prior_a = prior_b = 0
trials = []
for i in range(100):
    row_a = 24576 + 512 + (i % 4) * 128
    row_b = row_a - (1, 63, 64, 127, 128, 129, 255, 256)[i % 8]
    source = (x[0][24576 + (512, 2048, 3328, 3840)[i % 4]].float()
              * (1 + (i % 9 - 4) / 64)).bfloat16()
    xa[prior_a].zero_(); xb[prior_b].zero_()
    xa[row_a].copy_(source); xb[row_b].copy_(source)
    fn = service_project if args.service_kernel else lambda t, weight: project(t, weight, (128, 128, 128, 8))
    a = fn(xa, w)[row_a]
    b = fn(xb, w)[row_b]
    assert torch.equal(a, b), i
    trials.append(dict(iteration=i, row_a=row_a, row_b=row_b, exact=True))
    prior_a, prior_b = row_a, row_b
result['mutations'] = trials
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'mutations'}, indent=2))
print('shifted mutations exact', len(trials))
