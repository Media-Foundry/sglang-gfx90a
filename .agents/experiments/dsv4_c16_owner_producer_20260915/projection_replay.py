"""Bounded real-input projection replay; verifies weights against live-object hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
from safetensors import safe_open


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def diff(a, b):
    d = (a.float() - b.float()).abs()
    return dict(changed_elements=int((a != b).sum()), changed_rows=int((a != b).any(1).sum()),
                max_abs=float(d.max()), finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blas', choices=['cublas', 'cublaslt'])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    source = root / 'capture/oracle'
    report = json.loads((source / 'report.json').read_text())
    assert hashlib.sha256((source / 'inputs.pt').read_bytes()).hexdigest() == report['inputs_file_sha256']
    data = torch.load(source / 'inputs.pt', map_location='cpu', weights_only=True)
    assert sha(data['q_lora']) == report['q_lora_sha256']
    assert sha(data['x']) == report['x_sha256']
    model = Path('/home/pc/models/modelscope')
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    key = 'layers.20.attn.indexer.wq_b.weight'
    scale_key = 'layers.20.attn.indexer.wq_b.scale'
    with safe_open(model / index[key], framework='pt', device='cpu') as f:
        packed = f.get_tensor(key)
        scale = f.get_tensor(scale_key)
    weight = (packed.float().reshape(64, 128, 8, 128) * scale.float()[:, None, :, None]).reshape(8192, 1024).bfloat16()
    assert sha(weight) == report['parameters']['wq_b']['weight']['sha256'], 'Not the actual runtime BF16 weight'
    from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan
    _, rows, valid, _ = host_plan(report['extend_lens'], report['prefix_lens'], 0)
    assert valid.all()
    device = torch.device('cuda:0')
    x, w = data['q_lora'].to(device), weight.to(device)
    ids = torch.from_numpy(rows).to(device)
    torch.backends.cuda.preferred_blas_library('default')
    default_baseline = F.linear(x, w).index_select(0, ids)
    if args.blas:
        # PyTorch keeps CUDA names for the corresponding ROCm BLAS backends.
        torch.backends.cuda.preferred_blas_library(args.blas)
    part = x.index_select(0, ids)
    reverse = torch.arange(len(part) - 1, -1, -1, device=device)
    full_reverse = torch.arange(len(x) - 1, -1, -1, device=device)
    variants = {
        'full': (x, lambda y: y.index_select(0, ids)),
        'compact': (part, lambda y: y),
        'compact_reverse': (part.index_select(0, reverse), lambda y: y.index_select(0, reverse)),
        'full_reverse': (x.index_select(0, full_reverse), lambda y: y.index_select(0, len(x) - 1 - ids)),
        'compact_padded4096': (F.pad(part, (0, 0, 0, 4096 - len(part))), lambda y: y[:len(part)]),
    }
    result = dict(diagnostic_only=True, gpu_name=torch.cuda.get_device_name(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  visible_devices=__import__('os').environ.get('HIP_VISIBLE_DEVICES'),
                  weight_sha256=sha(weight), input_sha256=sha(data['q_lora']),
                  torch_version=torch.__version__, hip_version=torch.version.hip,
                  preferred_blas=str(torch.backends.cuda.preferred_blas_library()), variants={})
    baseline = F.linear(x, w).index_select(0, ids)
    selected_outputs = {}
    for name, (operand, select) in variants.items():
        for _ in range(3):
            y = F.linear(operand, w)
        selected = select(y)
        selected_outputs[name] = selected.clone()
        entry = dict(shape=list(operand.shape), versus_full=diff(baseline, selected),
                     versus_default_full=diff(default_baseline, selected),
                     repeat=diff(selected, select(F.linear(operand, w))))
        samples = []
        for _ in range(6):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                F.linear(operand, w)
            b.record(); b.synchronize()
            samples.append(a.elapsed_time(b) / 5)
        entry['serial_diagnostic_ms'] = samples
        entry['median_ms'] = statistics.median(samples)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            F.linear(operand, w)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / f'{name}.trace.json'))
        entry['gpu_events'] = sorted(set(e.name for e in prof.events()
                                         if e.device_type == torch.autograd.DeviceType.CUDA))
        result['variants'][name] = entry
        (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print(name, json.dumps({k: v for k, v in entry.items() if k != 'gpu_events'}), flush=True)
    result['status'] = 'complete'
    result['compact_row_permutation'] = diff(selected_outputs['compact'], selected_outputs['compact_reverse'])
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
