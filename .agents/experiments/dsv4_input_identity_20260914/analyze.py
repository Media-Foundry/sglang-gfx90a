"""Key responses and sampled GPU rows by verified request identity, not batch slot."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parent


def digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def load_trace(name,lookup,rank=0):
    root=ROOT/('trace-'+name)
    def read(key): return torch.load(root/f'layer_0_rank_{rank}_{key}.pt',weights_only=True)
    ids=read('input_ids').tolist();positions=read('positions').tolist()
    starts=[i for i,p in enumerate(positions) if p==0]
    assert starts and starts[0]==0
    ends=starts[1:]+[len(ids)]
    row_to_key={};sequence_cases=[]
    for start,end in zip(starts,ends,strict=True):
        sequence=ids[start:end]
        case=lookup[digest(sequence)]
        assert positions[start:end]==list(range(end-start))
        sequence_cases.append(case)
        for row in range(start,end):row_to_key[row]=(case,positions[row])
    rows=read('sample_rows').tolist()
    keys=[row_to_key[row] for row in rows]
    assert len(set(keys))==len(keys)
    stages={}
    for stage in ('attn_residual','attn_pre_norm','attn_norm','q','attn_core',
                  'attn_inverse_rope','wo_a','wo_b_partial','wo_b','attn_out',
                  'ffn_input','ffn_out'):
        path=root/f'layer_0_rank_{rank}_{stage}.pt'
        if path.exists():
            value=torch.load(path,weights_only=True)
            assert value.shape[0]==len(keys),(stage,value.shape,len(keys))
            stages[stage]={key:value[j] for j,key in enumerate(keys)}
    return dict(actual_m=len(ids),cases=sequence_cases,sampled_rows=len(rows)),stages


def compare(a,b):
    result={}
    for stage in a:
        if stage not in b:continue
        common=sorted(set(a[stage])&set(b[stage]))
        rows=[]
        for key in common:
            x,y=a[stage][key],b[stage][key]
            delta=(x.float()-y.float()).abs()
            rows.append(dict(case=key[0],position=key[1],exact=torch.equal(x,y),
                             max_abs=float(delta.max()),changed=int(torch.count_nonzero(delta))))
        result[stage]=dict(exact_rows=sum(r['exact'] for r in rows),rows=len(rows),
                          max_abs=max(r['max_abs'] for r in rows),details=rows)
    return result


def main():
    assert (ROOT/'complete.json').exists()
    cases=json.loads((ROOT/'inputs.json').read_text())['requests']
    lookup={digest(row['input_ids']):i for i,row in enumerate(cases)}
    identities=json.loads((ROOT/'identity.json').read_text())
    assert all(row['hashes']==identities[0]['hashes'] and row['echo_exact']==16 for row in identities)
    metadata={};traces={}
    for name in ('A1','B1','B2','A2'):
        metadata[name],traces[name]=load_trace(name,lookup)
    result=dict(input_identity_exact=True,gpu_trace_metadata=metadata,
        stage_comparisons={f'{a}-{b}':compare(traces[a],traces[b])
                           for a,b in [('A1','A2'),('B1','B2'),('A1','B1')]})
    old=ROOT.parent/'dsv4_ck_drift_20260914/analyze.py'
    spec=importlib.util.spec_from_file_location('old_analysis',old)
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
    result['output_comparisons']={f'{a}-{b}':analysis.compare(
        json.loads((ROOT/f'{a}-by-case.json').read_text()),
        json.loads((ROOT/f'{b}-by-case.json').read_text()))
        for a,b in [('A1','A2'),('B1','B2'),('A1','B1')]}
    result['trace_inventory']={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for root in ROOT.glob('trace-*') if root.is_dir() for p in root.glob('*.pt')}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(metadata)
    for pair,stages in result['stage_comparisons'].items():
        print(pair,{stage:(v['exact_rows'],v['rows'],v['max_abs']) for stage,v in stages.items()})
    print('output exact',{k:v['exact'] for k,v in result['output_comparisons'].items()})


if __name__=='__main__':main()
