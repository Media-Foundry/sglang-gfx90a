"""Replay captured HC coefficients/post and Engram gates at original M shapes.

Single-device component only. A replay is evidence only when both service
outputs are reproduced; identical repeated-row inputs isolate M-dependence.
"""

import argparse
import json
from pathlib import Path

import torch

from compare_dsv41_row_trace import row_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats, hc_split_sinkhorn
    from sglang.srt.layers.engram import engram_gate

    grouped = {}
    for file in sorted(args.trace_dir.glob("*.pt")):
        r = torch.load(file, weights_only=True, map_location="cpu")
        n = 1 if r["tag"] == "engram_gate" else 2
        forward, sub = divmod(r["call"], n)
        if forward not in (1, 3):
            continue
        module = file.name.split("-module")[1].split("-")[0]
        grouped.setdefault((r["tag"], module, sub), {})[forward] = r
    assert grouped
    reports = []
    for key, entries in sorted(grouped.items()):
        a, b = entries[1], entries[3]
        aa = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in a['args'].items()}
        bb = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in b['args'].items()}

        def run(data):
            if key[0] == "engram_gate":
                return [engram_gate(**data)]
            if key[0] == "hc_post":
                x, residual, post, comb = (data[k] for k in ("x", "residual", "post", "comb"))
                return [(post.unsqueeze(-1) * x.unsqueeze(1)
                         + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)).to(x.dtype)]
            mix = hc_mix_stats(data['x'].flatten(1), data['hc_fn'], 1e-20).unsqueeze(1)
            pre, post, comb = hc_split_sinkhorn(mix, data['hc_scale'], data['hc_base'], 4, 20, 1e-6)
            return [pre.squeeze(1), post.squeeze(1), comb.squeeze(1)]

        with torch.no_grad():
            y1, y204 = run(aa), run(bb)
            # Identical input rows in a larger M: isolate shape-dependent
            # arithmetic from differences already present upstream.
            repeated = {k: v for k, v in aa.items()}
            for k in ('x', 'kv', 'residual', 'post', 'comb', 'apply_pre'):
                if isinstance(repeated.get(k), torch.Tensor):
                    v = repeated[k]
                    assert v.shape[0] == 1
                    repeated[k] = v.expand(204, *v.shape[1:]).contiguous()
            repeated_out = run(repeated)
        observed1 = a['output'][1:] if key[0] == '_hc_mix_and_combine' else [a['output']]
        observed204 = b['output'][1:] if key[0] == '_hc_mix_and_combine' else [b['output']]
        row = {'key': key, 'layer_id': a['layer_id'], 'prefix': a['prefix'],
               'service_decode': [row_metrics(x.cpu(), y) for x, y in zip(y1, observed1)],
               'service_prefill': [row_metrics(x.cpu(), y) for x, y in zip(y204, observed204)],
               'identical_input_M1_M204': [row_metrics(x[0].cpu(), y[-1].cpu()) for x,y in zip(y1,repeated_out)]}
        def intermediates(data):
            if key[0] == '_hc_mix_and_combine':
                x = data['x'].flatten(1).float()
                return {'dot':x @ data['hc_fn'].float().t(),
                        'rstd':torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-20)}
            if key[0] == 'engram_gate':
                h = data['x'].float()
                key_t, value = data['kv'].split([4*5120,5120],-1)
                key_t = key_t.float().unflatten(-1,(4,5120))
                h2 = h.square().mean(-1)
                k2 = key_t.square().mean(-1)
                dot = (h * (data['q_weight'].float()*data['k_weight'].float()) * key_t).sum(-1)
                rstd = torch.rsqrt(h2+data['eps'])*torch.rsqrt(k2+data['eps'])
                normalized = dot*rstd*5120**-0.5
                gate = torch.sigmoid(torch.copysign(normalized.abs().clamp_min(data['clamp_value']).sqrt(),normalized))
                return {'h_mean_square':h2,'key_mean_square':k2,'raw_dot':dot,'rstd':rstd,
                        'normalized_dot':normalized,'gate':gate}
            return {}
        inter1, inter204 = intermediates(aa), intermediates(repeated)
        row['intermediates_same_input'] = {k:row_metrics(v[0].cpu(),inter204[k][-1].cpu()) for k,v in inter1.items()}
        reports.append(row)
        print(json.dumps({'key':key, 'layer':a['layer_id'],
                          'decode_reproduced':all(r['exact'] for r in row['service_decode']),
                          'prefill_reproduced':all(r['exact'] for r in row['service_prefill']),
                          'same_input_drift':[r.get('max_abs') for r in row['identical_input_M1_M204']]}),flush=True)
        if inter1:
            print(json.dumps({k:v.get('max_abs') for k,v in row['intermediates_same_input'].items()}),flush=True)
    with args.output.open('x') as f:
        json.dump(reports,f,indent=2)


if __name__ == '__main__':
    main()
