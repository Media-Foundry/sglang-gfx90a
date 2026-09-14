"""CPU tests for the opt-in scope and sampled debug input contract."""
import os
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from sglang.kernels.ops.debug.dsv4_prefill_attention_ar import eligible, enabled_for
from sglang.kernels.ops.debug.dsv4_sampled_stage_dump import (
    sampled_stage_value, should_dump_stage, stage_dump_scope,
)
from sglang.kernels.ops.debug.dsv4_prepare_dump import make_prepare_dump


class Contract(unittest.TestCase):
    def test_core_compressor_scope(self):
        from sglang.kernels.ops.debug.dsv4_prefill_compressor import enabled, project, FLAG
        with patch.dict(os.environ,{FLAG:'0'}):
            self.assertFalse(enabled(None,None,None))
        comp=SimpleNamespace(_debug_original_v4=True,layer_id=2,is_in_indexer=False,
                             ratio=4,head_dim=512)
        parallel=SimpleNamespace(attn_cp_size=1,attn_tp_size=8)
        x=SimpleNamespace(device='fixture',shape=(32768,4096))
        with patch.dict(os.environ,{FLAG:'1'}), patch(
                'sglang.srt.runtime_context.get_parallel',return_value=parallel), patch(
                'sglang.kernels.ops.debug.dsv4_prefill_attention_ar.enabled_for',return_value=True):
            self.assertTrue(enabled(comp,x,None))
            for key,value in [('_debug_original_v4',False),('layer_id',3),('is_in_indexer',True),
                              ('ratio',128),('head_dim',128)]:
                with self.subTest(key=key):
                    self.assertFalse(enabled(SimpleNamespace(**{**vars(comp),key:value}),x,None))
            parallel.attn_cp_size=2
            self.assertFalse(enabled(comp,x,None))
        with self.assertRaises(ValueError):
            project(torch.empty(32,4096),torch.empty(2048,4096))

    def test_compressor_capture_preserves_values(self):
        from sglang.kernels.ops.debug import dsv4_compressor_dump as module
        x=torch.zeros(4,8);w=torch.ones(8,8);y=torch.ones(4,8)
        comp=SimpleNamespace(is_in_indexer=False,wkv_gate=SimpleNamespace(weight=w))
        with patch.object(module,'_dump',return_value=None):
            self.assertIs(module.projection(comp,x,None,y),y)
        seen=[]
        with patch.object(module,'_dump',return_value=lambda n,v,**k:seen.append((n,v,k))):
            self.assertIs(module.projection(comp,x,None,y),y)
        self.assertEqual([n for n,_,_ in seen],[f'prepare_compressor_core_{s}' for s in
                         ('input','weight','projection')])
        self.assertTrue(all(k==dict(full=True,rank0_only=True) for _,_,k in seen))

    def test_compressor_cache_capture_uses_plan_locations(self):
        from sglang.kernels.ops.debug import dsv4_compressor_dump as module
        raw=torch.tensor([[0,0],[0,65538]],dtype=torch.int32)
        class Plan:
            is_decode=False
            def __getitem__(self,i):
                assert i==1
                return raw
        cache=torch.arange(3*512).reshape(3,512).bfloat16()
        seen={}
        locations=torch.zeros(65539,dtype=torch.int64)
        locations[0]=2
        locations[2]=0  # A stale uint16 mask would select the wrong row.
        locations[65538]=1
        with patch('sglang.srt.runtime_context.get_parallel',return_value=SimpleNamespace(attn_tp_rank=0)), patch.object(
                module,'_dump',return_value=lambda n,v,**k:seen.update({n:v.clone()})):
            module.storage(SimpleNamespace(is_in_indexer=False),None,torch.zeros(2,512),
                           cache.view(torch.uint8),locations,Plan(),bf16_store=True)
        self.assertTrue(torch.equal(seen['prepare_compressor_core_result'],cache[[2,1]]))
        self.assertEqual(seen['prepare_compressor_core_locations'].tolist(),[2,1])

    def test_shared_guard_and_forward_batch_scope(self):
        from sglang.kernels.ops.debug.dsv4_prefill_shared import shared
        with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_PREFILL_SHARED_STABLE':'0'}):
            self.assertFalse(enabled_for(None,None,None,None,
                             flag='SGLANG_DSV4_DEBUG_PREFILL_SHARED_STABLE'))
        with self.assertRaises(ValueError):
            shared(torch.empty(32,4096),torch.empty(512,4096),torch.empty(4096,256),10.)
        source=Path(__file__).resolve().parents[3]/'python/sglang/srt/models/deepseek_v2.py'
        tree=ast.parse(source.read_text())
        callers=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef):
                for call in ast.walk(node):
                    if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                            and call.func.attr=='_forward_shared_experts'
                            and any(k.arg=='forward_batch' for k in call.keywords)):
                        callers.append(node.name)
        # ast.walk also visits the nested SBO hook from its owning method.
        self.assertEqual(callers,['forward_normal']*3+['_pre_combine_hook'])

    def test_stable_wob_disabled_and_geometry_contract(self):
        source=Path(__file__).resolve().parents[3]/'python/sglang/srt/models/deepseek_v4.py'
        tree=ast.parse(source.read_text())
        methods=[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                 and any(isinstance(c,ast.Constant) and c.value=='SGLANG_DSV4_DEBUG_PREFILL_WOB_STABLE'
                         for c in ast.walk(n))]
        self.assertEqual(methods,['forward'])
        with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_PREFILL_WOB_STABLE':'0'}):
            self.assertFalse(enabled_for(None,None,None,None,
                             flag='SGLANG_DSV4_DEBUG_PREFILL_WOB_STABLE'))
        from sglang.kernels.ops.debug.dsv4_prefill_wqb import project
        with self.assertRaises(ValueError):
            project(torch.empty(32,1024),torch.empty(4096,1024),projection_name='wo_b')
        with self.assertRaises(ValueError):
            project(None,None,projection_name='unknown')

    def test_stable_wqb_wiring_is_normal_prepare_only(self):
        source=Path(__file__).resolve().parents[3]/'python/sglang/srt/models/deepseek_v4.py'
        tree=ast.parse(source.read_text())
        methods=[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                 and any(isinstance(c,ast.Constant) and c.value=='SGLANG_DSV4_DEBUG_PREFILL_WQB_STABLE'
                         for c in ast.walk(n))]
        self.assertEqual(methods,['_forward_prepare'])
        with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_PREFILL_WQB_STABLE':'0'}):
            self.assertFalse(enabled_for(None,None,None,None,
                             flag='SGLANG_DSV4_DEBUG_PREFILL_WQB_STABLE'))

    def test_stable_qkv_wiring_is_normal_prepare_only(self):
        source=Path(__file__).resolve().parents[3]/'python/sglang/srt/models/deepseek_v4.py'
        tree=ast.parse(source.read_text())
        methods=[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                 and any(isinstance(c,ast.Constant) and c.value=='SGLANG_DSV4_DEBUG_PREFILL_QKV_STABLE'
                         for c in ast.walk(n))]
        self.assertEqual(methods,['_forward_prepare'])
        with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_PREFILL_QKV_STABLE':'0'}):
            self.assertFalse(enabled_for(None,None,None,None,
                             flag='SGLANG_DSV4_DEBUG_PREFILL_QKV_STABLE'))

    def test_prepare_dump_contract(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_PREPARE_DUMP": "0"}):
            self.assertIsNone(make_prepare_dump(None,None,None,None))
        batch=SimpleNamespace(spec_algorithm=None,forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda:True))
        positions=torch.tensor([0,1,2,0,1,2]);x=torch.arange(24).view(6,4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "SGLANG_DSV4_DEBUG_PREPARE_DUMP":"1", "SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR":directory,
                "SGLANG_DSV4_DEBUG_STAGE_LAYER":"1", "SGLANG_DSV4_DEBUG_STAGE_RANK":"-1",
                "SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS":"0,2"}):
            self.assertIsNone(make_prepare_dump(0,0,batch,positions))
            dump=make_prepare_dump(1,0,batch,positions)
            dump("prepare_q",x);dump("prepare_full_kv",x,full=True)
            self.assertTrue(torch.equal(torch.load(Path(directory)/'layer_1_rank_0_prepare_q.pt',weights_only=True),x[[0,2,3,5]]))
            self.assertTrue(torch.equal(torch.load(Path(directory)/'layer_1_rank_0_prepare_full_kv.pt',weights_only=True),x))
            make_prepare_dump(1,1,batch,positions)("not_saved",x,full=True,rank0_only=True)
            self.assertFalse((Path(directory)/'layer_1_rank_1_not_saved.pt').exists())

    def test_disabled_does_not_touch_gpu_or_batch(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32": "0"}):
            self.assertFalse(enabled_for(None, None, None, None))
        with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32": "1",
                                    "SGLANG_DSV4_DEBUG_PREFILL_WOA_STABLE": "0"}):
            self.assertFalse(enabled_for(None, None, None, None,
                                        flag="SGLANG_DSV4_DEBUG_PREFILL_WOA_STABLE"))

    def test_callback_ownership(self):
        module=SimpleNamespace()
        with stage_dump_scope(module, "outer"):
            self.assertEqual(module._dsv4_stage_dump, "outer")
            with self.assertRaises(ValueError):
                with stage_dump_scope(module, "inner"):
                    raise ValueError("fixture")
            self.assertEqual(module._dsv4_stage_dump, "outer")
        self.assertFalse(hasattr(module, "_dsv4_stage_dump"))

    def test_scope(self):
        base = dict(enabled=True, native=True, extend=True, hip=True,
                    arch="gfx90a", tp=8, attn_tp=8, ep=1, rows=32768)
        self.assertTrue(eligible(**base))
        for key, value in [("enabled", False), ("native", False), ("extend", False),
                           ("hip", False), ("arch", "gfx942"), ("tp", 4),
                           ("attn_tp", 4), ("ep", 2), ("rows", 8191),
                           ("rows", 36865), ("rows", 32), ("rows", 128)]:
            with self.subTest(key=key, value=value):
                self.assertFalse(eligible(**{**base, key: value}))
        self.assertTrue(eligible(**{**base, "rows": 8192}))
        self.assertTrue(eligible(**{**base, "rows": 36864}))

    def test_samples_keep_full_inputs(self):
        positions = torch.tensor([0, 1, 2, 0, 1, 2])
        x = torch.arange(24).view(6, 4)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS": "0,2"}):
                actual = sampled_stage_value("x", x, positions, directory, "p")
                self.assertTrue(torch.equal(actual, x[[0, 2, 3, 5]]))
                self.assertTrue(torch.equal(torch.load(Path(directory) / "p_sample_rows.pt",
                                                      weights_only=True), torch.tensor([0, 2, 3, 5])))
                for name in ("input_ids", "positions"):
                    self.assertTrue(torch.equal(sampled_stage_value(
                        name, positions, positions, directory, "p"), positions))
                # Parameters must not be row-sampled merely because N == M.
                for name in ("projection_wqkv_a", "hc_attn_fn", "router_weight"):
                    self.assertTrue(torch.equal(sampled_stage_value(
                        name, x, positions, directory, "p"), x))
                    with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_STAGE_SKIP_WEIGHTS": "1"}):
                        self.assertFalse(should_dump_stage(name))
                        self.assertTrue(should_dump_stage("ffn_topk_weights"))
            with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS": ""}):
                self.assertTrue(torch.equal(sampled_stage_value("x", x, positions, directory, "p"), x))


if __name__ == "__main__":
    unittest.main()
