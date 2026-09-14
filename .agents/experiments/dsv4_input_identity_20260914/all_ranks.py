"""Distinguish changed rank partials from a changed collective reduction."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
import analyze

ROOT=Path(__file__).resolve().parent/'all-ranks'


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='all-ranks')
    args = parser.parse_args()
    assert '/' not in args.run_name and args.run_name not in ('.', '..')
    ROOT = Path(__file__).resolve().parent / args.run_name
    assert (ROOT/'complete.json').exists()
    analyze.ROOT=ROOT
    identities=json.loads((ROOT/'identity.json').read_text())
    assert all(row['hashes']==identities[0]['hashes'] and row['echo_exact']==16 for row in identities)
    lookup={analyze.digest(row['input_ids']):i for i,row in enumerate(
        json.loads((ROOT/'inputs.json').read_text())['requests'])}
    data={};meta={};comp={}
    for name in ('A1','B1','A2'):
        data[name]=[];meta[name]=[]
        for rank in range(8):
            m,d=analyze.load_trace(name,lookup,rank)
            data[name].append(d);meta[name].append(m)
    for rank in range(8):
        comp[rank]=analyze.compare(data['A1'][rank],data['B1'][rank])
    common=sorted(set(data['A1'][0]['wo_b_partial'])&set(data['B1'][0]['wo_b_partial']))
    reductions=[]
    for key in common:
        item=dict(case=key[0],position=key[1])
        item['all_rank_partials_unchanged']=all(torch.equal(
            data['A1'][r]['wo_b_partial'][key],data['B1'][r]['wo_b_partial'][key]) for r in range(8))
        for name in ('A1','B1'):
            partial=[data[name][r]['wo_b_partial'][key] for r in range(8)]
            actual=data[name][0]['wo_b'][key]
            ref=sum(x.double() for x in partial).bfloat16()
            f32=sum(x.float() for x in partial).bfloat16()
            cyclic=[]
            for start in range(8):
                acc=partial[start].clone()
                for j in range(1,8): acc=(acc.float()+partial[(start+j)%8].float()).bfloat16()
                cyclic.append(int(torch.count_nonzero(acc!=actual)))
            item[name]=dict(
                all_ranks_agree=all(torch.equal(actual,data[name][r]['wo_b'][key]) for r in range(8)),
                changed_from_fp64_sum=int(torch.count_nonzero(actual!=ref)),
                max_abs_from_fp64_sum=float((actual.float()-ref.float()).abs().max()),
                f32_sum_matches_fp64=torch.equal(f32,ref),
                cyclic_bf16_changed_counts=cyclic)
        reductions.append(item)
    result=dict(metadata=meta,per_rank_comparison=comp,reductions=reductions,
        control_repeat={r:analyze.compare(data['A1'][r],data['A2'][r]) for r in range(8)})
    old=ROOT.parent.parent/'dsv4_ck_drift_20260914/analyze.py'
    spec=importlib.util.spec_from_file_location('old_analysis',old)
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
    result['output_comparisons']={f'{a}-{b}':analysis.compare(
        json.loads((ROOT/f'{a}-by-case.json').read_text()),
        json.loads((ROOT/f'{b}-by-case.json').read_text()))
        for a,b in [('A1','A2'),('A1','B1')]}
    result['input_identity_exact']=True
    (ROOT/'all-rank-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('all rank partials unchanged',all(r['all_rank_partials_unchanged'] for r in reductions))
    for r,stages in comp.items():
        print(r,{s:(v['exact_rows'],v['rows'],v['max_abs']) for s,v in stages.items()})
    for row in reductions:
        if row['case']==15 and row['position']==0: print('boundary',row)
    print('output exact',{k:v['exact'] for k,v in result['output_comparisons'].items()})


if __name__=='__main__':main()
