"""Compare captured compressor stages by request identity, not physical slots."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from analyze import digest

torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument('--run', default='stable-layer2-compressor')
args = parser.parse_args()
root = Path(__file__).resolve().parent / args.run
assert (root / 'complete.json').exists()
cases = json.loads((root / 'inputs.json').read_text())['requests']
lookup = {digest(r['input_ids']): i for i, r in enumerate(cases)}
sources = {}

def read(arm, name):
    p = root / ('trace-' + arm) / ('layer_2_rank_0_' + name + '.pt')
    sources[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return torch.load(p, weights_only=True)

def load(arm):
    ids, pos = read(arm, 'input_ids').tolist(), read(arm, 'positions').tolist()
    starts = [i for i, p in enumerate(pos) if p == 0]
    assert starts[0] == 0
    keys = []
    for a, b in zip(starts, starts[1:] + [len(pos)]):
        case = lookup[digest(ids[a:b])]
        assert pos[a:b] == list(range(b-a))
        keys.extend((case, p) for p in pos[a:b])
    plan = read(arm, 'prepare_compressor_core_plan').long()
    ragged = plan[:, 1] & 0xFFFFFFFF
    seq_len = plan[:, 0] & ((1 << 23)-1)
    assert ragged.min() >= 0 and ragged.max() < len(keys)
    compressed_keys = [keys[r] for r in ragged.tolist()]
    assert all(k[1] == s-1 for k, s in zip(compressed_keys, seq_len.tolist()))
    assert len(set(compressed_keys)) == len(compressed_keys)
    sampled_keys = [keys[r] for r in read(arm, 'sample_rows').tolist()]
    stages = {}
    for name in ('core_input', 'core_projection', 'index_input', 'index_projection',
                 'core_pooled', 'core_result'):
        tensor = read(arm, 'prepare_compressor_' + name)
        kk = compressed_keys if name in ('core_pooled','core_result') else keys
        assert tensor.shape[0] == len(kk)
        stages[name] = (kk, tensor)
    for name in ('q', 'attn_core'):
        stages[name] = (sampled_keys, read(arm, name))
    assert torch.equal(stages['core_input'][1], read(arm, 'prepare_full_input'))
    weights = {name: read(arm, 'prepare_compressor_'+name+'_weight') for name in ('core','index')}
    return stages, weights, dict(m=len(keys), compressed_rows=len(compressed_keys))

def compare(a, b):
    out = {}
    for stage in a:
        ak, ax = a[stage]; bk, bx = b[stage]
        ai, bi = {k:i for i,k in enumerate(ak)}, {k:i for i,k in enumerate(bk)}
        common = sorted(ai.keys() & bi.keys()); assert common
        x = ax[[ai[k] for k in common]]; y = bx[[bi[k] for k in common]]
        neq = (x != y).reshape(len(common), -1)
        changed_rows = neq.any(dim=1).nonzero().flatten().tolist()
        details = []
        for row in changed_rows:
            details.append(dict(case=common[row][0], position=common[row][1],
                changed=int(neq[row].sum()), max_abs=float((x[row]-y[row]).abs().max()),
                first_columns=neq[row].nonzero().flatten()[:16].tolist()))
        out[stage] = dict(rows=len(common), changed_rows=len(changed_rows),
            changed_elements=int(neq.sum()), max_abs=float((x.float()-y.float()).abs().max()), details=details)
    return out

loaded = {arm: load(arm) for arm in ('A1','B1','A2')}
for arm in ('B1','A2'):
    for name in ('core','index'):
        assert torch.equal(loaded['A1'][1][name], loaded[arm][1][name])
result = dict(shapes={a:v[2] for a,v in loaded.items()}, comparisons={}, sources=sources)
for arm in ('B1','A2'):
    comparison = compare(loaded['A1'][0], loaded[arm][0])
    result['comparisons']['A1-'+arm] = comparison
    print('A1-'+arm, json.dumps({k:{**v,'details':v['details'][:3]} for k,v in comparison.items()}), flush=True)
(root/'compressor-summary.json').write_text(json.dumps(result,indent=2)+'\n')
