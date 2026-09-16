"""Align the same request's logical positions; never compare padded batch hashes."""
import hashlib
import json
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
target = root / 'trace-analysis.json'
assert not target.exists()
arms = ('mixed', 'serial')
prompt = json.loads((root/'mixed/teacher-prompts.json').read_text())['input_ids'][0]


def load_metadata(directory, layer, rank):
    data = torch.load(directory/f'layer-{layer}-rank-{rank}-metadata.pt', weights_only=False)
    meta = data['metadata']
    ids, positions = data['input_ids'].tolist(), data['positions'].tolist()
    offset = 0
    matches = []
    for count, prefix in zip(meta['extend_lens'], meta['prefix_lens'], strict=True):
        if count and prefix >= 32768 and ids[offset:offset+count] == prompt[prefix:prefix+count]:
            assert positions[offset:offset+count] == list(range(prefix, prefix+count))
            matches.append({positions[row]: row for row in range(offset, offset+count)})
        offset += count
    assert offset == meta['logical_rows'] <= meta['rows']
    assert len(matches) == 1, (directory, layer, rank, meta, len(matches))
    return meta, matches[0]


def records(directory, layer, rank):
    result = {}
    for path in directory.glob(f'layer-{layer}-rank-{rank}-*.json'):
        item = json.loads(path.read_text())
        if 'values_file' in item:
            payload_path = directory/item['values_file']
            assert hashlib.sha256(payload_path.read_bytes()).hexdigest() == item['file_sha256']
            item['payload'] = torch.load(payload_path, weights_only=False)
        result[item['stage']] = item
    for name in ('attn_residual','attn_norm','q','attn_core','attn_out','ffn_input','ffn_out'):
        assert name in result, (directory, name)
    return result


for arm in arms:
    directory = root/f'trace-{arm}'
    assert json.loads((directory/'complete.json').read_text())['status'] == 'diagnostic_complete'
    assert not json.loads((directory/f'P32k-tail-policy-trace-{arm}.stop.json').read_text())['remaining']

results = []
for rank in range(8):
    for layer in range(3):
        directories = [root/f'trace-{arm}'/'trace' for arm in arms]
        left, right = [load_metadata(d, layer, rank) for d in directories]
        positions = sorted(set(left[1]) & set(right[1]))
        assert len(positions) >= 32, (rank, layer, len(positions))
        a, b = [records(d, layer, rank) for d in directories]
        stages = []
        for name in sorted(a.keys() & b.keys(), key=lambda key: a[key]['sequence']):
            x, y = a[name], b[name]
            if x.get('row_tensor') and y.get('row_tensor'):
                assert x['dtype'] == y['dtype'] and x['shape'][1:] == y['shape'][1:]
                ix = [left[1][p] for p in positions]
                iy = [right[1][p] for p in positions]
                same = (x['payload']['row_hashes'][ix] == y['payload']['row_hashes'][iy]).all(dim=1)
                differing = [p for p, exact in zip(positions, same.tolist(), strict=True) if not exact]
                stages.append(dict(stage=name, compared_rows=len(positions), different_rows=len(differing),
                                   first_different_position=differing[0] if differing else None))
            elif name not in ('input_ids','positions'):
                assert x.get('is_none') == y.get('is_none'), (name, rank, layer)
                if not x.get('is_none'):
                    assert x['sha256'] == y['sha256'], ('non-row input changed', name, rank, layer)
        results.append(dict(rank=rank, layer=layer, shapes=[left[0]['rows'],right[0]['rows']],
            logical_shapes=[left[0]['logical_rows'],right[0]['logical_rows']],
            first_different_stage=next((s['stage'] for s in stages if s['different_rows']), None), stages=stages))
output = dict(status='complete', diagnostic_only=True, request=0,
              comparison='same logical token positions, mixed versus serial, common FP32/20', results=results)
target.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps([{k:v for k,v in item.items() if k!='stages'} for item in results], indent=2))
