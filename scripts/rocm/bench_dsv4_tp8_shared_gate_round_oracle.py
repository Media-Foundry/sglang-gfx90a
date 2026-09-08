#!/usr/bin/env python3
"""Standalone TP8 C1 gate+activation; production selector untouched."""
import json
import statistics
import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args
from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import gfx90a_wave64_bf16_gemv


def main():
    torch.manual_seed(20908)
    x=torch.randn(1,4096,device='cuda',dtype=torch.bfloat16)
    weights=[torch.randn(512,4096,device='cuda',dtype=torch.bfloat16)*0.02 for _ in range(43)]
    outputs=[torch.empty(1,256,device='cuda',dtype=torch.bfloat16) for _ in weights]
    def reference(w):
        z=gfx90a_wave64_bf16_gemv(x,w)
        g,u=z.chunk(2,-1)
        return torch.nn.functional.silu(g.clamp(max=10))*u.clamp(min=-10,max=10)
    for rows,waves in [(1,4),(1,8),(2,4),(2,8)]:
        args=make_cpp_args(512,4096,rows,2,waves,True)
        module=load_jit('dsv4_tp8_shared_gate_torch_round_oracle',*args,
            cuda_files=['gemm/gfx90a_bf16_gated_gemv.cuh'],
            cuda_wrappers=[('run',f'sglang::Gfx90aBf16GatedGemvKernel<{args}>::run')],
            extra_cuda_cflags=['-O3'])
        exact=0;max_abs=0.
        for i in range(100):
            x.normal_()
            if i%25==0:weights[0].normal_(0,0.02)
            expected=reference(weights[0]);module.run(x,weights[0],outputs[0],10.)
            exact+=int(torch.equal(expected,outputs[0]))
            max_abs=max(max_abs,float((expected.float()-outputs[0].float()).abs().max()))
        report={'rows':rows,'waves':waves,'exact':exact,'mutations':100,'max_abs':max_abs}
        if exact!=100:
            print(json.dumps(report),flush=True);continue
        graphs=[]
        for fused in [False,True]:
            def run():
                for w,y in zip(weights,outputs):
                    if fused:module.run(x,w,y,10.)
                    else:reference(w)
            run();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
            graphs.append(graph)
        graph=graphs[1];graph.replay();expected=[y.clone() for y in outputs]
        for _ in range(1000):graph.replay()
        assert all(torch.equal(y,z) for y,z in zip(outputs,expected))
        times={'A':[],'B':[]}
        for _ in range(5):
            for name,graph in [('A',graphs[0]),('B',graphs[1]),('B',graphs[1]),('A',graphs[0])]:
                for _ in range(5):graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):graph.replay()
                end.record();end.synchronize()
                times[name].append(start.elapsed_time(end)*1000/(20*43))
        report['median_us']={k:statistics.median(v) for k,v in times.items()}
        report['graph_replay_exact']=True
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
