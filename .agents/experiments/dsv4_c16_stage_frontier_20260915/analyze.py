"""Compare complete stage fingerprints; selected numeric rows are supplemental."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def load_values(root, record):
    path = root / record['values_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['file_sha256']
    return torch.load(path, weights_only=True, map_location='cpu')


def spans(indices, metadata):
    """Losslessly compact row IDs into request-local absolute-position runs."""
    bounds = np.cumsum([0, *metadata['metadata']['extend_lens']])
    positions = metadata['positions'].numpy()
    result = []
    for row in indices:
        row = int(row)
        case = int(np.searchsorted(bounds[1:], row, side='right'))
        pos = int(positions[row])
        if (result and result[-1]['case'] == case
                and result[-1]['last_row'] + 1 == row
                and result[-1]['last_position'] + 1 == pos):
            result[-1].update(last_row=row, last_position=pos)
        else:
            result.append(dict(case=case, first_row=row, last_row=row,
                               first_position=pos, last_position=pos))
    return result


def main():
    roots = [ROOT / arm / 'data' for arm in ('A', 'B')]
    for arm in ('A', 'B'):
        assert (ROOT / arm / 'complete.json').exists(), arm
    plans = [read_json(ROOT / arm / 'plan.json') for arm in ('A', 'B')]
    assert plans[0]['sources'] == plans[1]['sources']
    assert plans[0]['input_sha256'] == plans[1]['input_sha256']
    records = []
    for layer in range(20, 25):
        for rank in range(8):
            stem = f'layer-{layer}-rank-{rank}'
            metas = [torch.load(p / (stem + '-metadata.pt'), weights_only=True,
                                map_location='cpu') for p in roots]
            assert metas[0]['metadata'] == metas[1]['metadata']
            for field in ('input_ids', 'positions'):
                assert torch.equal(metas[0][field], metas[1][field]), (stem, field)
            files = [set(p.name for p in root.glob(stem + '-*.json')) for root in roots]
            assert files[0] == files[1], stem
            pairs = [(read_json(roots[0] / name), read_json(roots[1] / name))
                     for name in files[0]]
            for a, b in sorted(pairs, key=lambda pair: pair[0]['sequence']):
                for field in ('layer', 'rank', 'stage', 'sequence', 'is_none', 'shape', 'dtype', 'row_tensor'):
                    assert a.get(field) == b.get(field), (stem, field)
                item = dict(layer=layer, rank=rank, stage=a['stage'], sequence=a['sequence'],
                            equal=a.get('sha256') == b.get('sha256'), is_none=a['is_none'])
                if a.get('row_tensor'):
                    av, bv = [load_values(root, record) for root, record in zip(roots, (a, b))]
                    changed = torch.any(av['row_hashes'] != bv['row_hashes'], dim=1).nonzero().flatten().numpy()
                    assert (len(changed) == 0) == item['equal'], (stem, a['stage'])
                    item.update(changed_rows=len(changed), spans=spans(changed, metas[0]))
                    dtype = getattr(torch, a['dtype'].removeprefix('torch.'))
                    x = av['samples'].contiguous().view(dtype).float()
                    y = bv['samples'].contiguous().view(dtype).float()
                    item['sampled_rows_only'] = dict(
                        max_abs=float((x-y).abs().max()),
                        relative_l2=float(torch.linalg.vector_norm(x-y) /
                                          torch.linalg.vector_norm(x).clamp_min(1e-30)),
                        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
                records.append(item)
    groups = {}
    for item in records:
        key = (item['layer'], item['sequence'], item['stage'])
        group = groups.setdefault(key, dict(layer=key[0], sequence=key[1], stage=key[2],
                                           different_ranks=[], row_counts={}))
        if not item['equal']:
            group['different_ranks'].append(item['rank'])
            group['row_counts'][item['rank']] = item.get('changed_rows')
    summary = [groups[key] for key in sorted(groups)]
    result = dict(diagnostic_only=True, same_sources=True, same_full_inputs_positions_and_layout=True,
                  warning='Earliest observed state is not necessarily the causal first operation. '
                          'ffn_routed may include fused shared output. Samples are not full numeric oracles.',
                  summary=summary, records=records)
    (ROOT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([item for item in summary if item['different_ranks']], indent=2))


if __name__ == '__main__':
    main()
