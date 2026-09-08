#!/usr/bin/env python3
"""Check the opt-in projection kernel, not whole-model correctness."""
import itertools
from types import SimpleNamespace
from unittest.mock import patch
import torch
import sglang.srt.layers.quantization.dsv4_projection_experiment as experiment

from sglang.srt.layers.quantization.dsv4_projection_experiment import (
    eligible, maybe_row_stable_linear, projection_scope,
)
from sglang.kernels.ops.quantization.gfx90a_row_stable_linear import row_stable_linear


def main():
    for v in itertools.product([False, True], repeat=7):
        e, h, a, native, ext, tp, ep = v
        assert eligible(enabled=e, hip=h, arch='gfx90a' if a else 'gfx942',
                        native=native, extend=ext, tp=8 if tp else 4,
                        ep=1 if ep else 2) == all(v)
    with projection_scope(None, None):
        assert maybe_row_stable_linear(None, None) is None
    batch = SimpleNamespace(spec_algorithm=None, forward_mode=SimpleNamespace(
        is_extend_without_speculative=lambda: True))
    with patch.object(experiment.envs.SGLANG_DSV4_GFX90A_ROW_STABLE_PREFILL,
                      'get', return_value=True), patch(
        'sglang.srt.runtime_context.get_parallel', return_value=SimpleNamespace(
            tp_size=8, attn_tp_size=8, moe_ep_size=1)), patch.object(
        torch.cuda, 'get_device_properties', return_value=SimpleNamespace(
            gcnArchName='gfx90a:sramecc+')):
        try:
            with projection_scope(batch, 0):
                assert experiment._active.get()
                batch.forward_mode = SimpleNamespace(
                    is_extend_without_speculative=lambda: False)
                with projection_scope(batch, 0):
                    assert not experiment._active.get()
                assert experiment._active.get()
                raise RuntimeError('exercise cleanup')
        except RuntimeError as exc:
            assert str(exc) == 'exercise cleanup'
        assert not experiment._active.get()
    print('128 selector combinations and default-off scope passed', flush=True)
    torch.manual_seed(20908)
    for m, n, k in [(736,1536,4096), (65,128,256), (5,64,64), (0,64,64)]:
        w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        x = torch.randn(1,k,device='cuda',dtype=torch.bfloat16).repeat(m,1)
        y = row_stable_linear(x,w)
        assert torch.equal(y, y[:1].expand_as(y))
        if not m:
            continue
        ref = (x[:1].double() @ w.double().t()).bfloat16()
        torch.testing.assert_close(y[:1],ref,atol=0.02,rtol=0.01)
        for _ in range(100):
            x.copy_(torch.randn_like(x[:1]).expand_as(x))
            y = row_stable_linear(x,w)
            assert torch.equal(y,y[:1].expand_as(y))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = row_stable_linear(x,w)
        graph.replay()
        expected = y.clone()
        for _ in range(1000):
            graph.replay()
        assert torch.equal(y,expected)
        print((m,n,k),'100 mutations and 1000 graph replays passed',flush=True)


if __name__ == '__main__':
    main()
