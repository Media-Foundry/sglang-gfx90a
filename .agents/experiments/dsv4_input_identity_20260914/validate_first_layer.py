"""Validate the bounded first-layer claims; never assert whole-model stability."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    for name in ('combined-l0','combined-ffn','both-ar-l0'):
        assert (ROOT/name/'complete.json').exists()
        rows=json.loads((ROOT/name/'identity.json').read_text())
        assert all(x['echo_exact']==16 and x['hashes']==rows[0]['hashes'] for x in rows)
    d=json.loads((ROOT/'both-ar-l0/all-rank-summary.json').read_text())
    boundary=('attn_out','ffn_input','ffn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb')
    for rank in range(8):
        for stage in boundary:
            r=d['per_rank_comparison'][str(rank)][stage]
            assert r['exact_rows']==r['rows']==15 and r['max_abs']==0,(rank,stage)
        for stage,r in d['control_repeat'][str(rank)].items():
            assert r['exact_rows']==r['rows'] and r['max_abs']==0,(rank,stage,'repeat')
    for row in d['ffn_reductions']:
        assert row['all_rank_partials_unchanged']
        for arm in ('A1','B1'):
            assert row[arm]['changed_vs_sum']==0 and row[arm]['all_ranks_agree']
    for name in ('tree','ring','tree1'):
        d=json.loads((ROOT/f'rccl-{name}.json').read_text())
        assert all(r['all_ranks_agree'] for r in d['summary'])
    d=json.loads((ROOT/'woa-service-mutation.json').read_text())
    assert len(d)==2 and all(r['shifted_exact']==100 and len(r['trials'])==100 for r in d)
    print('Verified input identity, sampled layer0 boundary closure, repeat controls, collective rank agreement, and wo_a mutations.')
    print('Whole-model output repeatability and performance acceptance remain UNPROVEN.')


if __name__=='__main__':main()
