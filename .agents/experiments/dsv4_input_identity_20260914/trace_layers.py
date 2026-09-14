"""All-layer sampled first-propagation analysis, with request identity alignment."""
import argparse
import json
from pathlib import Path
import analyze
import torch


def main():
    torch.set_num_threads(4)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True)
    p.add_argument('--layers',type=int,default=43)
    args=p.parse_args()
    assert '/' not in args.run_name and args.run_name not in ('.','..')
    root=Path(__file__).resolve().parent/args.run_name
    assert (root/'complete.json').exists()
    ids=json.loads((root/'identity.json').read_text())
    assert all(r['echo_exact']==16 and r['hashes']==ids[0]['hashes'] for r in ids)
    lookup={analyze.digest(r['input_ids']):i for i,r in enumerate(json.loads((root/'inputs.json').read_text())['requests'])}
    analyze.ROOT=root;layers=[]
    for layer in range(args.layers):
        comparisons={};controls={};metadata=[]
        for rank in range(8):
            ma,a=analyze.load_trace('A1',lookup,rank,layer)
            mb,b=analyze.load_trace('B1',lookup,rank,layer)
            mc,c=analyze.load_trace('A2',lookup,rank,layer)
            required={'attn_residual','q','attn_core','wo_a','wo_b_partial','wo_b',
                      'attn_out','ffn_input','ffn_topk_ids','ffn_topk_weights',
                      'ffn_routed','ffn_shared','ffn_partial','ffn_out'}
            for label,stages in (('A1',a),('B1',b),('A2',c)):
                assert required<=stages.keys(),(layer,rank,label,required-stages.keys())
            metadata.append(dict(rank=rank,A1=ma,B1=mb,A2=mc))
            comparisons[rank]=analyze.compare(a,b)
            controls[rank]=analyze.compare(a,c)
            assert all(v['rows']>0 for v in (*comparisons[rank].values(),*controls[rank].values()))
        summary={}
        for stage in comparisons[0]:
            rows=[comparisons[r][stage] for r in range(8)]
            summary[stage]=dict(exact_rows=sum(r['exact_rows'] for r in rows),
                rows=sum(r['rows'] for r in rows),max_abs=max(r['max_abs'] for r in rows),
                changed_ranks=[r for r in range(8) if comparisons[r][stage]['exact_rows']!=comparisons[r][stage]['rows']])
        layers.append(dict(layer=layer,metadata=metadata,summary=summary,
                           comparison=comparisons,control=controls))
        print('LAYER',layer,{s:(v['exact_rows'],v['rows'],v['max_abs']) for s,v in summary.items() if v['changed_ranks']},flush=True)
    boundary_stages=('attn_residual','attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','ffn_input','ffn_out')
    first=next((dict(layer=x['layer'],stage=s,**x['summary'][s]) for x in layers
                for s in boundary_stages if s in x['summary'] and x['summary'][s]['changed_ranks']),None)
    first_control=next((dict(layer=x['layer'],rank=rank,stage=s,
                            exact_rows=stages[s]['exact_rows'],rows=stages[s]['rows'],
                            max_abs=stages[s]['max_abs']) for x in layers
                        for s in boundary_stages for rank,stages in x['control'].items()
                        if s in stages and stages[s]['exact_rows']!=stages[s]['rows']),None)
    result=dict(input_identity_exact=True,sampled_only=True,first_boundary_difference=first,
                first_control_boundary_difference=first_control,layers=layers)
    (root/'all-layer-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FIRST BOUNDARY',first,flush=True)
    print('FIRST CONTROL BOUNDARY',first_control,flush=True)


if __name__=='__main__':main()
