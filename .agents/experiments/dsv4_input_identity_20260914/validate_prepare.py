"""Bounded assertions: repaired QKV/KV fixture, not whole-model determinism."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
def read(path):
    return json.loads((root / path).read_text())

for name in ('layer1-prepare', 'layer0-changed-rows', 'layer0-stable-qkv'):
    assert read(name + '/complete.json')['diagnostic_only']
    identities = read(name + '/identity.json')
    assert all(r['echo_exact'] == 16 and r['hashes'] == identities[0]['hashes'] for r in identities)
fixed = read('layer0-stable-qkv/prepare-summary.json')
assert all(r['changed_elements'] == 0 for rows in fixed['comparisons'].values() for r in rows)
baseline = read('layer0-changed-rows/prepare-summary.json')
qkv = [r for r in baseline['comparisons']['A1-B1']
       if r['rank'] == 0 and r['stage'] == 'prepare_full_qkv_a' and r['case'] == 15]
assert len(qkv) == 1 and qkv[0]['changed_elements'] == 4
for name, attn, ffn in [('layer0-changed-rows', 141, 147), ('layer0-stable-qkv', 13, 16)]:
    data = read(name + '/all-rank-summary.json')
    assert all(v['exact_rows'] == v['rows'] for rank in data['control_repeat'].values() for v in rank.values())
    for stage, expected in [('attn_out', attn), ('ffn_out', ffn)]:
        assert sum(not r['exact'] for r in data['per_rank_comparison']['0'][stage]['details']) == expected
for name in ('qkv-order-oracle.json', 'qkv-service-oracle.json'):
    data = read(name)
    assert all(r['changed'] == 0 for r in data['baseline_vs_service'])
    assert data['baseline_shift']['changed'] == 4
    assert len(data['mutations']) == 100 and all(r['exact'] for r in data['mutations'])
for row in read('qkv-shape-row-axes.json'):
    assert row['row_shift_m32768']['changed'] == row['row_shift_m32767']['changed'] == 1
    assert row['shape_change_at_row_a']['changed'] == row['shape_change_at_row_b']['changed'] == 0
print('PASS: input identity, projection reproduction, fixed-QKV/KV, bounded service reduction; whole-output drift remains.')
