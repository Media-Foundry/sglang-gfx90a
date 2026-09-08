#!/usr/bin/env python3
"""TP8 layout cost screen using real occupancy; not a model-output oracle."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from bench_dsv4_tp4_m32_noa2a_ep2_oracle import (
    make_stage, balanced_owners, owner_metadata, full_metadata,
    reconstruct_topk_from_counts,
)
from bench_dsv4_tp4_m32_paged_decode_geometry import capture, time_graph
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import _jit_gate_up_grouped_row_prefetch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recorder',required=True)
    p.add_argument('--pass-index',type=int,default=80)
    p.add_argument('--layers',default='0,20,40')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    payload=torch.load(a.recorder,map_location='cpu',weights_only=False)
    reports=[]
    for layer in map(int,a.layers.split(',')):
        counts=payload['logical_count'][a.pass_index,layer]
        assert (counts%8==0).all()
        counts=counts//8
        assert counts.sum()==192 and counts.max()<=32
        ids=reconstruct_topk_from_counts(counts).cuda()
        assert torch.equal(torch.bincount(ids.cpu().flatten().long(),minlength=256),counts)
        owners=balanced_owners(counts,2)
        assert len(owners[0])==len(owners[1])==128
        assert set(owners[0]).isdisjoint(owners[1])
        torch.manual_seed(20260908+layer)
        xq=torch.randint(-127,128,(32,4096),device='cuda',dtype=torch.int8)
        xs=torch.rand((32,128),device='cuda')*.01
        weights=torch.rand((32,6),device='cuda')
        stages={'A':make_stage('A',256,256,full_metadata(ids),xq,xs,weights,832,832,False)}
        for owner,experts in enumerate(owners):
            for grid in (832,1664):
                name=f'B{owner}-G{grid}'
                stages[name]=make_stage(name,128,512,owner_metadata(ids,experts),
                    xq,xs,weights,grid,grid,True)
        graphs={}
        for name,stage in stages.items():
            stage.s13.random_(122,128);stage.s2.random_(122,128)
            stage.intermediate.zero_()
            module=_jit_gate_up_grouped_row_prefetch(stage.e,32,6,stage.i,4096,4,2,8,stage.gate_blocks,2)
            def gate(stage=stage,module=module):
                module.run(stage.xq,stage.xs,stage.w13,stage.s13,
                    stage.metadata.sorted_ids,stage.metadata.sorted_experts,
                    stage.metadata.valid,stage.intermediate,10.)
            # Validate new prefetched geometry against generic gate for this shape.
            for mutation in range(100):
                xq.random_(-127,128);xs.uniform_(.001,.01)
                if mutation%10==0:stage.s13.random_(122,128)
                stage.gate();reference=stage.intermediate.clone()
                gate();torch.cuda.synchronize()
                assert torch.equal(reference,stage.intermediate),(layer,name,mutation)
            def full(stage=stage,gate=gate):
                gate();stage.quant();stage.down();stage.reduce()
                return stage.output
            full();torch.cuda.synchronize()
            graphs[name],out=capture(full)
            assert torch.isfinite(out).all()
            expected=out.clone()
            for _ in range(1000):graphs[name].replay()
            torch.cuda.synchronize();assert torch.equal(expected,out)
        samples={name:[] for name in stages}
        order=list(stages)
        for _ in range(5):
            for name in order+list(reversed(order)):
                samples[name].append(time_graph(graphs[name],20,100))
        report=dict(layer=layer,pass_index=a.pass_index,counts=counts.tolist(),owners=owners,
            samples_us=samples,trimmed_us={n:statistics.mean(sorted(v)[1:-1]) for n,v in samples.items()},
            gate_exact_mutations_per_shape=100,stable_replays=1000,
            scans={n:st.metadata.sorted_experts.numel() for n,st in stages.items()},
            note='Synthetic independent weights; no cross-layout numerical or E2E claim; sorter/input quant/global collective excluded')
        reports.append(report)
        if a.output:
            a.output.write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        for g in graphs.values():g.reset()
        del graphs,stages


if __name__=='__main__':main()
