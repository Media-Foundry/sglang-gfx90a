#!/usr/bin/env python3
"""CPU-only execution of scope and actual attention selector AST."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from sglang.srt.distributed.device_communicators import dsv4_ar_experiment as scope


def main():
    flag=scope.envs.SGLANG_DSV4_GFX90A_TP8_DECODE_ATTN_WARPS2
    batch=NS(spec_algorithm=None,batch_size=1,forward_mode=NS(is_decode=lambda:True))
    path=Path('python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py')
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sparse_attn_v4_paged_decode_triton')
    blocks=[n for n in fn.body if isinstance(n,ast.If) and 'native_attention_active' in ast.unparse(n)]
    assert len(blocks)==1
    code=compile(ast.Module(body=blocks,type_ignores=[]),str(path),'exec')
    base=dict(_oracle_num_warps=None,_oracle_num_stages=None,quant_kv=False,
              fuse_inverse_rope=False,q=NS(dtype=torch.bfloat16),T=1,H=8,D=512,
              block_h=16,block_k=16,kv_splits=64,num_warps=4,num_stages=2,
              torch=torch,logging=logging,__name__=__name__,_tp8_warps2_logged=set())
    parallel=NS(tp_size=8,attn_tp_size=8,moe_ep_size=1)
    with (patch.object(flag,'get',return_value=True),patch.object(torch.version,'hip','test'),patch.object(
        torch.cuda,'get_device_properties',return_value=NS(gcnArchName='gfx90a')),
        patch('sglang.srt.runtime_context.get_parallel',return_value=parallel),patch(
        'sglang.srt.utils.common.is_gfx90a_supported',return_value=True)):
        for m in (1,32):
            batch.batch_size=m
            with scope.dsv4_ar_scope(batch,None):
                ns={**base,'T':m};exec(code,ns);assert ns['num_warps']==2
                for key,value in [('T',2),('H',16),('D',256),('block_h',32),
                                  ('block_k',32),('kv_splits',1),('quant_kv',True),
                                  ('fuse_inverse_rope',True),('q',NS(dtype=torch.float16)),
                                  ('_oracle_num_warps',4),('_oracle_num_stages',3)]:
                    ns={**base,key:value};exec(code,ns);assert ns['num_warps']==4,key
                for key,value in [('tp_size',4),('attn_tp_size',4),('moe_ep_size',2)]:
                    with patch.object(parallel,key,value):
                        ns=dict(base);exec(code,ns);assert ns['num_warps']==4
            assert not scope.native_attention_active()
        for b in (NS(**{**vars(batch),'spec_algorithm':NS(is_none=lambda:False)}),
                  NS(**{**vars(batch),'forward_mode':NS(is_decode=lambda:False)}),
                  NS(**{**vars(batch),'batch_size':16})):
            with scope.dsv4_ar_scope(b,None):
                assert not scope.native_attention_active()
        try:
            with scope.dsv4_ar_scope(batch,None):raise ValueError('cleanup')
        except ValueError:pass
        assert not scope.native_attention_active()
    ns=dict(base);exec(code,ns);assert ns['num_warps']==4
    c1flag=scope.envs.SGLANG_DSV4_GFX90A_TP8_C1_ATTN_WARPS2
    with (patch.object(flag,'get',return_value=False),
          patch.object(c1flag,'get',return_value=True),
          patch.object(torch.version,'hip','test'),
          patch.object(torch.cuda,'get_device_properties',return_value=NS(gcnArchName='gfx90a')),
          patch('sglang.srt.runtime_context.get_parallel',return_value=parallel),
          patch('sglang.srt.utils.common.is_gfx90a_supported',return_value=True)):
        fused={**base,'fuse_inverse_rope':True}
        batch.batch_size=1
        with scope.dsv4_ar_scope(batch,None):
            ns=dict(fused);exec(code,ns);assert ns['num_warps']==2
            for key,value in [('T',32),('T',2),('H',16),('D',256),('block_h',32),
                              ('block_k',32),('kv_splits',1),('quant_kv',True),
                              ('fuse_inverse_rope',False),('q',NS(dtype=torch.float16)),
                              ('_oracle_num_warps',4),('_oracle_num_stages',3)]:
                ns={**fused,key:value};exec(code,ns);assert ns['num_warps']==4,key
            for key,value in [('tp_size',4),('attn_tp_size',4),('moe_ep_size',2)]:
                with patch.object(parallel,key,value):
                    ns=dict(fused);exec(code,ns);assert ns['num_warps']==4
        for overrides in ({'batch_size':32}, {'batch_size':2},
                          {'spec_algorithm':NS(is_none=lambda:False)},
                          {'forward_mode':NS(is_decode=lambda:False)}):
            with scope.dsv4_ar_scope(NS(**{**vars(batch),**overrides}),None):
                assert not scope.native_attention_active()
                ns=dict(fused);exec(code,ns);assert ns['num_warps']==4
        assert not scope.native_attention_active()
    ns={**base,'fuse_inverse_rope':True};exec(code,ns);assert ns['num_warps']==4
    print('PASS: independent fused C1 flag; C32/nonfused/speculative/prefill remain excluded')
    print('PASS: actual selector M1/M32, shape/dtype/TP/EP exclusions, speculative/prefill exclusion and cleanup')


if __name__=='__main__':main()
