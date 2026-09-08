#!/usr/bin/env python3
"""CPU dispatch tests for the default-off TP8 M32 wo_a experiment."""
import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.layers.quantization import dsv4_woa_experiment as m


def main():
    keys = ['enabled','hip','arch','native','decode','batch_size','tp','attn_tp','ep']
    choices = [(True,False),(True,False),('gfx90a','gfx942'),(True,False),
               (True,False),(32,1),(8,4),(8,4),(1,2)]
    for values in itertools.product(*choices):
        assert m.eligible(**dict(zip(keys,values))) == all(
            v == c[0] for v,c in zip(values,choices))
    flag = m.envs.SGLANG_DSV4_GFX90A_TP8_M32_WOA_HIPBLASLT
    with patch.object(flag,'get',return_value=False):
        assert m.maybe_woa(None,None,None,None) is None

    def tensor(shape):
        t = MagicMock()
        t.shape=shape; t.dtype=torch.bfloat16; t.device=torch.device('cuda:0')
        t.is_cuda=True; t.is_contiguous.return_value=True
        t.__getitem__.return_value=t; t.t.return_value=t
        return t
    x,w = tensor((32,1,4096)),tensor((1,1024,4096))
    batch = SimpleNamespace(spec_algorithm=None,batch_size=32,
                            forward_mode=SimpleNamespace(is_decode=lambda:True))
    m._ready_devices.clear()
    with patch.object(flag,'get',return_value=True), patch.object(torch.version,'hip','mock'), \
         patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(gcnArchName='gfx90a')), \
         patch('sglang.srt.runtime_context.get_parallel',return_value=SimpleNamespace(tp_size=8,moe_ep_size=1)), \
         patch('torch.cuda.is_current_stream_capturing',return_value=True) as capturing, \
         patch('aiter.ops.gradlib.hipb_create_extension') as init, \
         patch('aiter.ops.gradlib.hipb_findallsols',return_value=[]) as supported, \
         patch('aiter.tuned_gemm.extensions_created',False), \
         patch('aiter.tuned_gemm.hipb_gemm') as gemm:
        try:
            m.maybe_woa(x,w,batch,8)
            raise AssertionError('first-use capture must fail')
        except RuntimeError as e:
            assert 'before graph capture' in str(e)
        init.assert_not_called();gemm.assert_not_called()
        capturing.return_value=False
        try:
            m.maybe_woa(x,w,batch,8)
            raise AssertionError('missing solution must fail')
        except RuntimeError as e:
            assert 'unavailable' in str(e)
        assert not m._ready_devices
        supported.return_value=[4429]
        gemm.return_value.unsqueeze.return_value='output'
        assert m.maybe_woa(x,w,batch,8)=='output'
        assert gemm.call_args.args[2]==4429
        count=supported.call_count
        capturing.return_value=True
        assert m.maybe_woa(x,w,batch,8)=='output'
        assert supported.call_count==count
        assert init.call_count==1, 'global workspace must not be initialized twice'
        w.shape=(2,1024,4096)
        assert m.maybe_woa(x,w,batch,8) is None
    m._ready_devices.clear()
    print('PASS: 512 conditions, disabled no-op, eager initialization, capture guard, '
          'unsupported solution failure, cached capture, shape rejection')


if __name__ == '__main__': main()
