#!/usr/bin/env python3
"""Standalone TP8 I256 gate screen; no production selector or weight cache."""
import json
import statistics
import argparse

import torch
from bench_dsv4_gfx90a_occupancy_bucket_oracle import make_metadata
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_gate_up_grouped, _jit_gate_up_grouped_row_prefetch, _jit_down_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8
from sglang.kernels.jit.utils import load_jit, make_cpp_args


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--down-prefetch',action='store_true',
                        help='Keep prefetched gate fixed; test isolated subgroup8 down candidate')
    parser.add_argument('--breakdown', action='store_true',
                        help='Measure prefetched gate, quant, down and reducer separately; diagnostic only')
    parser.add_argument('--gate-row-stripe', type=int, choices=(8,16,32,64,128),
                        help='Standalone task-order oracle against accepted prefetched gate')
    parser.add_argument('--gate-blocks', type=int, choices=(416,624,832,1040,1248),
                        help='Only change accepted prefetched gate CTA count; full chain comparison')
    parser.add_argument('--paired-scale-load', action='store_true',
                        help='Standalone gate/up scale uint16 load; unchanged scale layout')
    parser.add_argument('--lut-replicas', type=int, choices=(8,16,32),
                        help='Standalone lane-replicated LDS LUT, no new accumulators')
    parser.add_argument('--uniform-metadata', action='store_true',
                        help='Standalone readfirstlane expert/token metadata; unchanged arithmetic')
    parser.add_argument('--uniform-waves', type=int, choices=(4,8), default=8,
                        help='Uniform candidate wave count, preserving total grid waves')
    parsed=parser.parse_args()
    if parsed.uniform_metadata:
        assert not (parsed.lut_replicas or parsed.paired_scale_load or parsed.gate_blocks or parsed.gate_row_stripe or parsed.down_prefetch or parsed.breakdown)
        parsed.full=True
    if parsed.lut_replicas:
        assert not (parsed.paired_scale_load or parsed.gate_blocks or parsed.gate_row_stripe or parsed.down_prefetch or parsed.breakdown)
        parsed.full=True
    full=parsed.full or parsed.down_prefetch or parsed.breakdown or bool(parsed.gate_row_stripe) or bool(parsed.gate_blocks) or parsed.paired_scale_load
    assert not (parsed.gate_row_stripe and parsed.down_prefetch)
    assert not (parsed.gate_blocks and (parsed.gate_row_stripe or parsed.down_prefetch or parsed.breakdown))
    assert not (parsed.paired_scale_load and (parsed.gate_blocks or parsed.gate_row_stripe or parsed.down_prefetch or parsed.breakdown))
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
    torch.manual_seed(20908)
    m,t,e,i,k=32,6,256,256,4096
    x=torch.randint(-127,128,(m,k),dtype=torch.int8,device='cuda')
    scale=torch.full((m,k//32),.01,device='cuda')
    w=torch.randint(0,256,(e,2*i,k//2),dtype=torch.uint8,device='cuda')
    ws=torch.full((e,2*i,k//32),127,dtype=torch.uint8,device='cuda')
    args=(e,m,t,i,k,4,2,8,832,2)
    mods=[_jit_gate_up_grouped(*args),_jit_gate_up_grouped_row_prefetch(*args)]
    if parsed.uniform_metadata:
        uniform_args=(*args[:7],parsed.uniform_waves,832*8//parsed.uniform_waves,args[9])
        cpp=make_cpp_args(*uniform_args)
        mods=[mods[1],load_jit(
            'gfx90a_fp4_gate_uniform_metadata_oracle',*uniform_args,
            cuda_files=['deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh'],
            cuda_wrappers=[('run',f'sglang::Gfx90aFp4ExpertGateRowPrefetchOracle<{cpp}>::run')],
            extra_cuda_cflags=['-O3','-DSGLANG_FP4_GATE_UNIFORM_METADATA_ORACLE=1'])]
    if parsed.down_prefetch:mods=[mods[1],mods[1]]
    if parsed.gate_blocks:
        candidate_args=(*args[:8],parsed.gate_blocks,args[9])
        mods=[mods[1],_jit_gate_up_grouped_row_prefetch(*candidate_args)]
    if parsed.paired_scale_load:
        cpp=make_cpp_args(*args)
        mods=[mods[1],load_jit(
            'gfx90a_fp4_gate_paired_scale_oracle',*args,
            cuda_files=['deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh'],
            cuda_wrappers=[('run',f'sglang::Gfx90aFp4ExpertGateRowPrefetchOracle<{cpp}>::run')],
            extra_cuda_cflags=['-O3','-DSGLANG_FP4_GATE_PAIRED_SCALE_LOAD_ORACLE=1'])]
    if parsed.lut_replicas:
        cpp=make_cpp_args(*args)
        mods=[mods[1],load_jit(
            'gfx90a_fp4_gate_lut_replicas_oracle',*args,parsed.lut_replicas,
            cuda_files=['deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh'],
            cuda_wrappers=[('run',f'sglang::Gfx90aFp4ExpertGateRowPrefetchOracle<{cpp}>::run')],
            extra_cuda_cflags=['-O3',f'-DSGLANG_FP4_GATE_LUT_REPLICAS_ORACLE={parsed.lut_replicas}'])]
    if parsed.gate_row_stripe:
        cpp=make_cpp_args(*args)
        mods=[mods[1],load_jit(
            'gfx90a_fp4_gate_row_stripe_oracle',*args,parsed.gate_row_stripe,
            cuda_files=['deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh'],
            cuda_wrappers=[('run',f'sglang::Gfx90aFp4ExpertGateRowPrefetchOracle<{cpp}>::run')],
            extra_cuda_cflags=['-O3',f'-DSGLANG_FP4_GATE_ROW_STRIPE_ORACLE={parsed.gate_row_stripe}'])]
    if full:
        down=_jit_down_grouped(e,m,t,k,i,4,2,8,832,2)
        w2=torch.randint(0,256,(e,k,i//2),dtype=torch.uint8,device='cuda')
        s2=torch.randint(122,128,(e,k,i//32),dtype=torch.uint8,device='cuda')
        router_weights=torch.rand((m,t),device='cuda')
        downs=[down,down]
        if parsed.down_prefetch:
            cpp=make_cpp_args(e,m,t,k,i,4,8,832,2)
            downs[1]=load_jit('gfx90a_fp4_tp8_down_prefetch_oracle',*cpp,
                             cuda_files=['deepseek_v4/gfx90a_fp4_tp8_down_prefetch_oracle.cuh'],
                             cuda_wrappers=[('run_partial',f'sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{cpp}>::run_partial'),
                                            ('reduce',f'sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{cpp}>::reduce')],
                             extra_cuda_cflags=['-O3'])
    for expert_pool in (256,128,32):
        ids=torch.rand((m,expert_pool),device='cuda').topk(t,dim=1).indices.int()
        meta=make_metadata(ids,assignments=4)
        outputs=[torch.empty((m,t,i),dtype=torch.bfloat16,device='cuda') for _ in mods]
        finals=[torch.empty((m,k),dtype=torch.bfloat16,device='cuda') for _ in mods] if full else outputs
        partials=[torch.empty((m,t,k),dtype=torch.float32,device='cuda') for _ in mods] if full else []
        graphs=[]
        for arm,(mod,out) in enumerate(zip(mods,outputs)):
            def run():
                mod.run(x,scale,w,ws,meta.sorted_ids,meta.sorted_experts,meta.valid,out,10.)
                if full:
                    q,qs=per_token_group_quant_int8(out,32)
                    downs[arm].run_partial(q,qs,w2,s2,meta.sorted_ids,meta.sorted_experts,
                                     meta.valid,router_weights,partials[arm])
                    downs[arm].reduce(partials[arm],finals[arm])
            run();torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):run()
            graphs.append(g)
        exact=0;partial_exact=0;max_abs=0.;stable=True
        for n in range(100):
            x.random_(-127,128)
            scale.uniform_(.001,.02)
            if n%25==0:ws.random_(122,128)
            for g in graphs:g.replay()
            torch.cuda.synchronize()
            exact+=int(torch.equal(*outputs) and torch.equal(*finals))
            if full:partial_exact+=int(torch.equal(*partials))
            max_abs=max(max_abs,float((finals[0]-finals[1]).abs().max()))
            assert all(torch.isfinite(o).all() for o in finals)
            expected=finals[1].clone()
            for _ in range(10):graphs[1].replay()
            torch.cuda.synchronize()
            stable=stable and torch.equal(expected,finals[1])
        samples=[[],[]]
        for _ in range(5):
            for arm in (0,1,1,0):
                for _ in range(20):graphs[arm].replay()
                torch.cuda.synchronize()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):graphs[arm].replay()
                end.record();end.synchronize()
                samples[arm].append(begin.elapsed_time(end)*10)
        print(json.dumps(dict(full_stage=full,uniform_metadata=parsed.uniform_metadata,uniform_waves=parsed.uniform_waves,down_prefetch=parsed.down_prefetch,gate_row_stripe=parsed.gate_row_stripe,gate_blocks=parsed.gate_blocks,paired_scale_load=parsed.paired_scale_load,lut_replicas=parsed.lut_replicas,expert_pool=expert_pool,active_experts=ids.unique().numel(),
                              scans=meta.sorted_experts.numel(),exact_mutations=exact,
                              partial_exact_mutations=partial_exact if full else None,
                              max_abs=max_abs,stable=stable,samples_us=samples,
                              trimmed_us=[statistics.mean(sorted(v)[1:-1]) for v in samples])),flush=True)
        if parsed.breakdown:
            # Match the accepted prefetched gate, not the old unprefetched arm.
            arm = 1
            graphs[arm].replay()
            torch.cuda.synchronize()
            expected_partial = partials[arm].clone()
            expected_final = finals[arm].clone()
            stage_graphs = []
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mods[arm].run(x,scale,w,ws,meta.sorted_ids,meta.sorted_experts,
                              meta.valid,outputs[arm],10.)
            stage_graphs.append(('gate',g))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                quant, quant_scale = per_token_group_quant_int8(outputs[arm],32)
            stage_graphs.append(('intermediate_quant',g))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                downs[arm].run_partial(quant,quant_scale,w2,s2,meta.sorted_ids,
                                      meta.sorted_experts,meta.valid,router_weights,partials[arm])
            stage_graphs.append(('down_partial',g))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                downs[arm].reduce(partials[arm],finals[arm])
            stage_graphs.append(('fixed_reducer',g))
            for _,g in stage_graphs:g.replay()
            torch.cuda.synchronize()
            assert torch.equal(partials[arm],expected_partial)
            assert torch.equal(finals[arm],expected_final)
            stage_samples={name:[] for name,_ in stage_graphs}
            # Alternate measurement order to limit monotonic drift. Isolated
            # graph durations do not add up to the multistream service path.
            for cycle in range(10):
                order=stage_graphs if cycle%2==0 else list(reversed(stage_graphs))
                for name,g in order:
                    for _ in range(20):g.replay()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):g.replay()
                    end.record();end.synchronize()
                    stage_samples[name].append(start.elapsed_time(end)*10)
            print(json.dumps(dict(diagnostic='isolated_prefetched_chain',
                                  expert_pool=expert_pool,exact_split_chain=True,
                                  samples_us=stage_samples,
                                  trimmed_us={name:statistics.mean(sorted(v)[1:-1])
                                              for name,v in stage_samples.items()})),flush=True)


if __name__=='__main__':main()
